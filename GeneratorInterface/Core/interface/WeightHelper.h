#ifndef GeneratorInterface_LHEInterface_WeightHelper_h
#define GeneratorInterface_LHEInterface_WeightHelper_h

#include "DataFormats/Common/interface/OwnVector.h"
#include "SimDataFormats/GeneratorProducts/interface/GenWeightProduct.h"
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
    edm::OwnVector<gen::WeightGroupInfo> weightGroups() { return weightGroups_; }

    template <typename T>
    std::unique_ptr<GenWeightProduct> weightProduct(std::vector<T> weights, float w0);

    void setfillEmptyIfWeightFails(bool value) { fillEmptyIfWeightFails_ = value; }
    void setModel(std::string model) { model_ = model; }
    void setGuessPSWeightIdx(bool guessPSWeightIdx) {
      PartonShowerWeightGroupInfo::setGuessPSWeightIdx(guessPSWeightIdx);
    }
    void addUnassociatedGroup() {
      weightGroups_.push_back(std::make_unique<UnknownWeightGroupInfo>("unassociated"));
      weightGroups_.back().setDescription("Weights with missing or invalid header meta data");
    }
    int addWeightToProduct(
        std::unique_ptr<GenWeightProduct>& product, double weight, std::string name, int weightNum, int groupIndex);
    int findContainingWeightGroup(std::string wgtId, int weightIndex, int previousGroupIndex);
    void setDebug(bool value) { debug_ = value; }
    bool fillEmptyIfWeightFails() { return fillEmptyIfWeightFails_; }

  protected:
    // TODO: Make this only print from one thread a la
    // https://github.com/kdlong/cmssw/blob/master/PhysicsTools/NanoAOD/plugins/GenWeightsTableProducer.cc#L1069
    bool debug_ = false;
    bool fillEmptyIfWeightFails_ = false;
    const unsigned int FIRST_PSWEIGHT_ENTRY = 2;
    const unsigned int DEFAULT_PSWEIGHT_LENGTH = 46;
    std::string model_;
    std::vector<ParsedWeight> parsedWeights_;
    std::map<std::string, std::string> currWeightAttributeMap_;
    std::map<std::string, std::string> currGroupAttributeMap_;
    edm::OwnVector<gen::WeightGroupInfo> weightGroups_;
    bool isScaleWeightGroup(const ParsedWeight& weight);
    bool isMEParamWeightGroup(const ParsedWeight& weight);
    bool isPdfWeightGroup(const ParsedWeight& weight);
    bool isPartonShowerWeightGroup(const ParsedWeight& weight);
    bool isOrphanPdfWeightGroup(ParsedWeight& weight);
    void updateScaleInfo(gen::ScaleWeightGroupInfo& scaleGroup, const ParsedWeight& weight);
    void updateMEParamInfo(const ParsedWeight& weight, int index);
    void updatePdfInfo(gen::PdfWeightGroupInfo& pdfGroup, const ParsedWeight& weight);
    void updatePartonShowerInfo(gen::PartonShowerWeightGroupInfo& psGroup, const ParsedWeight& weight);
    void cleanupOrphanCentralWeight();
    bool splitPdfWeight(ParsedWeight& weight);

    int lhapdfId(const ParsedWeight& weight, gen::PdfWeightGroupInfo& pdfGroup);
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
    void printWeights();
    std::unique_ptr<WeightGroupInfo> buildGroup(ParsedWeight& weight);
    void buildGroups();
    std::string searchString(const std::string& label, const std::string& name);
  };

  // Templated function (needed here because of plugins)
  template <typename T>
  std::unique_ptr<GenWeightProduct> WeightHelper::weightProduct(std::vector<T> weights, float w0) {
    auto weightProduct = std::make_unique<GenWeightProduct>(w0);
    weightProduct->setNumWeightSets(weightGroups_.size());
    int weightGroupIndex = 0;
    int i = 0;
    // This happens if there are no PS weights, so the weights vector contains only the central GEN weight.
    // Just add an empty product (need for all cases or...?)
    if (weights.size() > 1) {
      for (const auto& weight : weights) {
        try {
          if constexpr (std::is_same<T, gen::WeightsInfo>::value) {
            weightGroupIndex = addWeightToProduct(weightProduct, weight.wgt, weight.id, i, weightGroupIndex);
          } else if (std::is_same<T, double>::value)
            weightGroupIndex = addWeightToProduct(weightProduct, weight, std::to_string(i), i, weightGroupIndex);

        } catch (cms::Exception& e) {
          if (fillEmptyIfWeightFails_) {
            std::cerr << "WARNING: " << e.what() << std::endl;
            std::cerr << "fillEmptyIfWeightFails_ is set to True, so variations will be empty!!" << std::endl;
            weightProduct->setNumWeightSets(1);  // Only central weight
            return weightProduct;
          } else {
            throw cms::Exception("ERROR: " + std::string(e.what()) +
                                 "\nfillEmptyIfWeightFails_ is set to False, so exiting code");
          }
        }
        i++;
      }
    }
    return weightProduct;
  }
}  // namespace gen

#endif
