#ifndef GeneratorInterface_LHEInterface_WeightHelper_h
#define GeneratorInterface_LHEInterface_WeightHelper_h

#include "SimDataFormats/GeneratorProducts/interface/GenWeightProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenWeightInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/WeightGroupInfo.h"
#include "SimDataFormats/GeneratorProducts/interface/PdfWeightGroupInfo.h"
#include "SimDataFormats/GeneratorProducts/interface/WeightsInfo.h"
#include "SimDataFormats/GeneratorProducts/interface/UnknownWeightGroupInfo.h"
#include "SimDataFormats/GeneratorProducts/interface/PdfWeightGroupInfo.h"
#include "SimDataFormats/GeneratorProducts/interface/PartonShowerWeightGroupInfo.h"
#include "SimDataFormats/GeneratorProducts/interface/ScaleWeightGroupInfo.h"
#include "SimDataFormats/GeneratorProducts/interface/MEParamWeightGroupInfo.h"
#include "LHAPDF/LHAPDF.h"
#include <boost/algorithm/string.hpp>
#include <bits/stdc++.h>
#include <fstream>
#include <memory>

namespace gen {
  struct ParsedWeight {
    std::string id;
    int index;
    std::string groupname;
    std::string content;
    std::unordered_map<std::string, std::string> attributes;
    int wgtGroup_idx;
  };

  class WeightHelper {
  public:
    WeightHelper();

    template <typename T>
    std::unique_ptr<GenWeightProduct> weightProduct(const GenWeightInfoProduct& weightsInfo,
                                                    std::vector<T> weights,
                                                    float w0) const;

    void setGuessPSWeightIdx(bool guessPSWeightIdx) {
      PartonShowerWeightGroupInfo::setGuessPSWeightIdx(guessPSWeightIdx);
    }
    void addUnassociatedGroup(std::vector<std::unique_ptr<gen::WeightGroupInfo>>& weightGroups) const {
      gen::UnknownWeightGroupInfo unassoc("unassociated");
      unassoc.setDescription("Weights with missing or invalid header meta data");
      weightGroups.push_back(std::make_unique<gen::UnknownWeightGroupInfo>(unassoc));
    }
    int addWeightToProduct(GenWeightProduct& product, double weight, std::string name, int weightNum, int groupIndex);
    void setDebug(bool value) { debug_ = value; }

  protected:
    bool debug_ = false;
    const unsigned int FIRST_PSWEIGHT_ENTRY = 2;
    const unsigned int DEFAULT_PSWEIGHT_LENGTH = 46;
    std::map<std::string, std::string> currWeightAttributeMap_;
    std::map<std::string, std::string> currGroupAttributeMap_;
    bool isScaleWeightGroup(const ParsedWeight& weight) const;
    bool isMEParamWeightGroup(const ParsedWeight& weight) const;
    bool isPdfWeightGroup(const ParsedWeight& weight) const;
    bool isPartonShowerWeightGroup(const ParsedWeight& weight) const;
    bool isOrphanPdfWeightGroup(ParsedWeight& weight) const;
    void updateScaleInfo(gen::ScaleWeightGroupInfo& scaleGroup, const ParsedWeight& weight) const;
    void updateMEParamInfo(const ParsedWeight& weight, int index) const;
    void updatePdfInfo(gen::PdfWeightGroupInfo& pdfGroup, const ParsedWeight& weight) const;
    void updatePartonShowerInfo(gen::PartonShowerWeightGroupInfo& psGroup, const ParsedWeight& weight) const;
    void cleanupOrphanCentralWeight(WeightGroupInfoContainer& weightGroups) const;
    bool splitPdfWeight(ParsedWeight& weight, WeightGroupInfoContainer& weightGroups) const;

    int lhapdfId(const ParsedWeight& weight, gen::PdfWeightGroupInfo& pdfGroup) const;
    std::string searchAttributes(const std::string& label, const ParsedWeight& weight) const;
    std::string searchAttributesByTag(const std::string& label, const ParsedWeight& weight) const;
    std::string searchAttributesByRegex(const std::string& label, const ParsedWeight& weight) const;

    // Possible names for the same thing
    const std::unordered_map<std::string, std::vector<std::string>> attributeNames_ = {
        {"muf", {"muF", "MUF", "muf", "facscfact"}},
        {"mur", {"muR", "MUR", "mur", "renscfact"}},
        {"pdf", {"PDF", "PDF set", "lhapdf", "pdf", "pdf set", "pdfset"}},
        {"dyn", {"DYN_SCALE"}},
        {"dyn_name", {"dyn_scale_choice"}},
        {"up", {"_up", "Hi"}},
        {"down", {"_dn", "Lo"}},
        {"me_variation", {"mass", "sthw2", "width"}},
    };
    void printWeights(const WeightGroupInfoContainer& weightGroups) const;
    std::unique_ptr<WeightGroupInfo> buildGroup(ParsedWeight& weight) const;
    WeightGroupInfoContainer buildGroups(std::vector<ParsedWeight>& parsedWeights, bool addUnassociatedGroup) const;
    std::string searchString(const std::string& label, const std::string& name) const;
  };

  template <typename T>
  std::unique_ptr<GenWeightProduct> WeightHelper::weightProduct(const GenWeightInfoProduct& weightsInfo,
                                                                std::vector<T> weights,
                                                                float w0) const {
    auto weightProduct = std::make_unique<GenWeightProduct>(w0);
    weightProduct->setNumWeightSets(weightsInfo.numberOfGroups());
    gen::WeightGroupData groupData = {0, nullptr};
    int i = 0;
    // size=1 happens if there are no PS weights, so the weights vector contains only the central GEN weight.
    if (weights.size() > 1) {
      // This gets remade every event to avoid having state-dependence in the helper class
      // could think about doing caching instead
      int unassociatedIdx = weightsInfo.unassociatedIdx();
      std::unique_ptr<gen::UnknownWeightGroupInfo> unassociatedGroup;
      if (unassociatedIdx != -1)
        unassociatedGroup = std::make_unique<gen::UnknownWeightGroupInfo>("unassociated");
      for (const auto& weight : weights) {
        double wgtval;
        std::string wgtid;
        if constexpr (std::is_same<T, gen::WeightsInfo>::value) {
          wgtid = weight.id;
          wgtval = weight.wgt;
        } else if (std::is_same<T, double>::value) {
          wgtid = std::to_string(i);
          wgtval = weight;
        }
        try {
          groupData = weightsInfo.containingWeightGroupInfo(i, groupData.index);
        } catch (const cms::Exception& e) {
          if (unassociatedIdx == -1)
            throw e;
          if (debug_) {
            std::cout << "WARNING: " << e.what() << std::endl;
          }
          // Access the unassociated group separately so it can be modified
          unassociatedGroup->addContainedId(i, wgtid, wgtid);
          groupData = {static_cast<size_t>(unassociatedIdx), unassociatedGroup.get()};
        }
        int entry = groupData.group->weightVectorEntry(wgtid, i);

        // TODO: is this too slow?
        if (debug_)
          std::cout << "Adding weight num " << i << " EntryNum " << entry << " to group " << groupData.index
                    << std::endl;
        weightProduct->addWeight(wgtval, groupData.index, entry);
        i++;
      }
    }
    return weightProduct;
  }
}  // namespace gen

#endif
