#include "GeneratorInterface/Core/interface/WeightHelper.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <regex>

namespace gen {
  WeightHelper::WeightHelper() {}

  bool WeightHelper::isScaleWeightGroup(const ParsedWeight& weight) const {
    return (weight.groupname.find("scale_variation") != std::string::npos ||
            weight.groupname.find("Central scale variation") != std::string::npos);
  }

  bool WeightHelper::isPdfWeightGroup(const ParsedWeight& weight) const {
    const std::string& name = weight.groupname;
    if (name.find("PDF_variation") != std::string::npos)
      return true;
    return LHAPDF::lookupLHAPDFID(name) != -1;
  }

  bool WeightHelper::isPartonShowerWeightGroup(const ParsedWeight& weight) const {
    const std::string& groupname = boost::to_lower_copy(weight.groupname);
    std::vector<std::string> psNames = {"isr", "fsr", "nominal", "baseline", "emission"};
    for (const auto& name : psNames) {
      if (groupname.find(name) != std::string::npos)
        return true;
    }
    return false;
  }

  bool WeightHelper::isOrphanPdfWeightGroup(ParsedWeight& weight) const {
    std::pair<std::string, int> pairLHA;
    try {
      pairLHA = LHAPDF::lookupPDF(stoi(searchAttributes("pdf", weight)));
    } catch (...) {
      return false;
    }

    if (!pairLHA.first.empty() && pairLHA.second == 0) {
      weight.groupname = std::string(pairLHA.first);
      return true;
    } else {
      return false;
    }
  }

  bool WeightHelper::isMEParamWeightGroup(const ParsedWeight& weight) const {
    return (weight.groupname.find("mg_reweighting") != std::string::npos ||
            weight.groupname.find("variation") != std::string::npos);
    // variation used for blanket of all variations, might need to change
  }

  std::string WeightHelper::searchAttributes(const std::string& label, const ParsedWeight& weight) const {
    std::string attribute = searchAttributesByTag(label, weight);
    return attribute.empty() ? searchAttributesByRegex(label, weight) : attribute;
  }

  std::string WeightHelper::searchAttributesByTag(const std::string& label, const ParsedWeight& weight) const {
    auto& attributes = weight.attributes;
    for (const auto& lab : attributeNames_.at(label)) {
      if (attributes.find(lab) != attributes.end()) {
        return boost::algorithm::trim_copy_if(attributes.at(lab), boost::is_any_of("\""));
      }
    }
    return "";
  }

  std::string WeightHelper::searchAttributesByRegex(const std::string& label, const ParsedWeight& weight) const {
    auto& content = weight.content;
    std::smatch match;
    for (const auto& lab : attributeNames_.at(label)) {
      std::regex floatExpr(lab + "\\s*=\\s*([0-9.]+(?:[eE][+-]?[0-9]+)?)");
      std::regex strExpr(lab + "\\s*=\\s*([^=]+)");
      if (std::regex_search(content, match, floatExpr)) {
        return boost::algorithm::trim_copy(match.str(1));
      } else if (std::regex_search(content, match, strExpr)) {
        return boost::algorithm::trim_copy(match.str(1));
      }
    }
    return "";
  }

  void WeightHelper::updateScaleInfo(gen::ScaleWeightGroupInfo& scaleGroup, const ParsedWeight& weight) const {
    std::string muRText = searchAttributes("mur", weight);
    std::string muFText = searchAttributes("muf", weight);
    std::string dynNumText = searchAttributes("dyn", weight);
    float muR, muF;
    try {
      muR = std::stof(muRText);
      muF = std::stof(muFText);
    } catch (std::invalid_argument& e) {
      if (debug_)
        std::cout << "Tried to convert (" << muR << ", " << muF << ") to a int" << std::endl;
      scaleGroup.setWeightIsCorrupt();
      return;
      /// do something
    }

    if (dynNumText.empty()) {
      scaleGroup.setMuRMuFIndex(weight.index, weight.id, muR, muF);
    } else {
      std::string dynType = searchAttributes("dyn_name", weight);
      try {
        int dynNum = std::stoi(dynNumText);
        scaleGroup.setDyn(weight.index, weight.id, muR, muF, dynNum, dynType);
      } catch (std::invalid_argument& e) {
        scaleGroup.setWeightIsCorrupt();
        /// do something here
      }
    }

    if (scaleGroup.lhaid() == -1) {
      std::string lhaidText = searchAttributes("pdf", weight);
      try {
        scaleGroup.setLhaid(std::stoi(lhaidText));
      } catch (std::invalid_argument& e) {
        scaleGroup.setLhaid(-1);
        // do something here
      }
    }
  }

  int WeightHelper::lhapdfId(const ParsedWeight& weight, gen::PdfWeightGroupInfo& pdfGroup) const {
    std::string lhaidText = searchAttributes("pdf", weight);

    if (debug_)
      std::cout << "Looking for LHAPDF info in ID " << lhaidText << std::endl;

    if (!lhaidText.empty()) {
      try {
        return std::stoi(lhaidText);
      } catch (std::invalid_argument& e) {
        pdfGroup.setIsWellFormed(false);
      }
    } else if (!pdfGroup.lhaIds().empty()) {
      return pdfGroup.lhaIds().back() + 1;
    } else {
      if (debug_)
        std::cout << "Looking up LHAPDF ID from name" << weight.groupname << std::endl;
      return LHAPDF::lookupLHAPDFID(weight.groupname);
    }
    return -1;
  }

  void WeightHelper::updatePdfInfo(gen::PdfWeightGroupInfo& pdfGroup, const ParsedWeight& weight) const {
    int lhaid = lhapdfId(weight, pdfGroup);
    if (debug_)
      std::cout << "LHAID identified as " << lhaid << std::endl;
    if (pdfGroup.parentLhapdfId() < 0) {
      int parentId = lhaid - LHAPDF::lookupPDF(lhaid).second;
      pdfGroup.setParentLhapdfInfo(parentId);
      pdfGroup.setUncertaintyType(gen::kUnknownUnc);

      std::string description = "";
      if (pdfGroup.uncertaintyType() == gen::kHessianUnc)
        description += "Hessian ";
      else if (pdfGroup.uncertaintyType() == gen::kMonteCarloUnc)
        description += "Monte Carlo ";
      description += "Uncertainty sets for LHAPDF set " + LHAPDF::lookupPDF(parentId).first;
      description += " with LHAID = " + std::to_string(parentId);
      description += "; ";

      pdfGroup.appendDescription(description);
    }
    // after setup parent info, add lhaid
    pdfGroup.addLhaid(lhaid);
  }

  void WeightHelper::updatePartonShowerInfo(gen::PartonShowerWeightGroupInfo& psGroup,
                                            const ParsedWeight& weight) const {
    if (psGroup.nIdsContained() == DEFAULT_PSWEIGHT_LENGTH) {
      psGroup.setIsWellFormed(true);
      psGroup.cacheWeightIndicesByLabel();
    }
    if (weight.content.find(':') != std::string::npos && weight.content.find('=') != std::string::npos)
      psGroup.setNameIsPythiaSyntax(true);
  }

  bool WeightHelper::splitPdfWeight(ParsedWeight& weight, WeightGroupInfoContainer& weightGroups) const {
    if (weightGroups[weight.wgtGroup_idx]->weightType() == gen::WeightType::kPdfWeights) {
      auto& pdfGroup = *static_cast<gen::PdfWeightGroupInfo*>(weightGroups[weight.wgtGroup_idx].get());
      int lhaid = lhapdfId(weight, pdfGroup);
      if (lhaid > 0 && !pdfGroup.isIdInParentSet(lhaid) && pdfGroup.parentLhapdfId() > 0) {
        weightGroups.push_back(buildGroup(weight));
        weight.wgtGroup_idx++;
        return true;
      }
    }
    return false;
  }

  void WeightHelper::cleanupOrphanCentralWeight(WeightGroupInfoContainer& weightGroups) const {
    auto centralIt = std::find_if(std::begin(weightGroups), std::end(weightGroups), [](auto& entry) {
      return entry->weightType() == gen::WeightType::kScaleWeights &&
             static_cast<ScaleWeightGroupInfo*>(entry.get())->containsCentralWeight();
    });
    if (centralIt == std::end(weightGroups))
      return;

    auto& centralWeight = *static_cast<gen::ScaleWeightGroupInfo*>(centralIt->get());

    std::vector<size_t> toRemove;
    for (size_t i = 0; i < weightGroups.size(); i++) {
      auto& group = weightGroups[i];
      if (group->weightType() == gen::WeightType::kPdfWeights) {
        auto& pdfGroup = *static_cast<gen::PdfWeightGroupInfo*>(group.get());
        // These are weights that contain nothing but a single central weight, because
        // some versions of madgraph write the central weight separately
        if (pdfGroup.nIdsContained() == 1 && pdfGroup.parentLhapdfId() == centralWeight.lhaid()) {
          toRemove.push_back(i);
          const auto& weightInfo = pdfGroup.weightMetaInfo(0);
          centralWeight.addContainedId(weightInfo.globalIndex, weightInfo.id, weightInfo.label, 1, 1);
        }
      }
    }
    // Indices are guaranteed to be unique, delete from high to low to avoid changing indices
    std::sort(std::begin(toRemove), std::end(toRemove), std::greater<size_t>());
    for (auto i : toRemove) {
      weightGroups.erase(std::begin(weightGroups) + i);
    }
  }

  void WeightHelper::printWeights(const WeightGroupInfoContainer& weightGroups) const {
    // checks
    for (const auto& group : weightGroups) {
      std::cout << std::boolalpha << group->name() << " (" << group->firstId() << "-" << group->lastId()
                << "): " << group->isWellFormed() << std::endl;
      if (group->weightType() == gen::WeightType::kScaleWeights) {
        const auto& groupScale = *static_cast<gen::ScaleWeightGroupInfo*>(group.get());
        std::cout << groupScale.centralIndex() << " ";
        std::cout << groupScale.muR1muF2Index() << " ";
        std::cout << groupScale.muR1muF05Index() << " ";
        std::cout << groupScale.muR2muF1Index() << " ";
        std::cout << groupScale.muR2muF2Index() << " ";
        std::cout << groupScale.muR2muF05Index() << " ";
        std::cout << groupScale.muR05muF1Index() << " ";
        std::cout << groupScale.muR05muF2Index() << " ";
        std::cout << groupScale.muR05muF05Index() << " \n";
        for (auto& name : groupScale.dynNames()) {
          std::cout << name << ": ";
          std::cout << groupScale.scaleIndex(1.0, 1.0, name) << " ";
          std::cout << groupScale.scaleIndex(1.0, 2.0, name) << " ";
          std::cout << groupScale.scaleIndex(1.0, 0.5, name) << " ";
          std::cout << groupScale.scaleIndex(2.0, 1.0, name) << " ";
          std::cout << groupScale.scaleIndex(2.0, 2.0, name) << " ";
          std::cout << groupScale.scaleIndex(2.0, 0.5, name) << " ";
          std::cout << groupScale.scaleIndex(0.5, 1.0, name) << " ";
          std::cout << groupScale.scaleIndex(0.5, 2.0, name) << " ";
          std::cout << groupScale.scaleIndex(0.5, 0.5, name) << "\n";
        }

      } else if (group->weightType() == gen::WeightType::kPdfWeights) {
        std::cout << group->description() << "\n";
      } else if (group->weightType() == gen::WeightType::kPartonShowerWeights) {
        const auto& groupPS = *static_cast<gen::PartonShowerWeightGroupInfo*>(group.get());
        std::vector<std::string> labels = groupPS.weightLabels();
        groupPS.printVariables();
      }
    }
  }

  std::unique_ptr<WeightGroupInfo> WeightHelper::buildGroup(ParsedWeight& weight) const {
    if (debug_) {
      std::cout << "Building group for weight group " << weight.groupname << " weight content is " << weight.content
                << std::endl;
    }
    if (isScaleWeightGroup(weight)) {
      if (debug_)
        std::cout << "Weight type is scale\n";
      return std::make_unique<ScaleWeightGroupInfo>(weight.groupname);
    } else if (isPdfWeightGroup(weight)) {
      if (debug_)
        std::cout << "Weight type is PDF\n";
      return std::make_unique<PdfWeightGroupInfo>(weight.groupname);
    } else if (isMEParamWeightGroup(weight)) {
      if (debug_)
        std::cout << "Weight type is MEParam\n";
      return std::make_unique<MEParamWeightGroupInfo>(weight.groupname);
    } else if (isPartonShowerWeightGroup(weight)) {
      if (debug_)
        std::cout << "Weight type is parton shower\n";
      return std::make_unique<PartonShowerWeightGroupInfo>("shower");
    } else if (isOrphanPdfWeightGroup(weight)) {
      if (debug_)
        std::cout << "Weight type is PDF\n";
      return std::make_unique<PdfWeightGroupInfo>(weight.groupname);
    }
    if (debug_)
      std::cout << "Weight type is unknown\n";

    std::cout << "Group name is " << weight.groupname << std::endl;

    return std::make_unique<UnknownWeightGroupInfo>(weight.groupname);
  }

  WeightGroupInfoContainer WeightHelper::buildGroups(std::vector<ParsedWeight>& parsedWeights,
                                                     bool addUnassociated) const {
    WeightGroupInfoContainer weightGroups;
    int groupOffset = 0;
    for (auto& weight : parsedWeights) {
      weight.wgtGroup_idx += groupOffset;
      if (debug_)
        std::cout << "Building group for weight " << weight.content << " group " << weight.groupname << " group index "
                  << weight.wgtGroup_idx << std::endl;

      int numGroups = static_cast<int>(weightGroups.size());
      if (weight.wgtGroup_idx == numGroups) {
        weightGroups.push_back(buildGroup(weight));
      } else if (weight.wgtGroup_idx >= numGroups)
        throw cms::Exception("Invalid group index " + std::to_string(weight.wgtGroup_idx));

      // split PDF groups
      if (splitPdfWeight(weight, weightGroups))
        groupOffset++;

      auto& group = weightGroups[weight.wgtGroup_idx];
      group->addContainedId(weight.index, weight.id, weight.content);
      if (group->weightType() == gen::WeightType::kScaleWeights)
        updateScaleInfo(*static_cast<gen::ScaleWeightGroupInfo*>(group.get()), weight);
      else if (group->weightType() == gen::WeightType::kPdfWeights)
        updatePdfInfo(*static_cast<gen::PdfWeightGroupInfo*>(group.get()), weight);
      else if (group->weightType() == gen::WeightType::kPartonShowerWeights)
        updatePartonShowerInfo(*static_cast<gen::PartonShowerWeightGroupInfo*>(group.get()), weight);
    }
    cleanupOrphanCentralWeight(weightGroups);
    if (addUnassociated) {
      addUnassociatedGroup(weightGroups);
    }
    return weightGroups;
  }

}  // namespace gen
