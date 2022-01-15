#include "GeneratorInterface/Core/interface/WeightHelper.h"
#include <regex>

namespace gen {
  WeightHelper::WeightHelper() { model_ = ""; }

  bool WeightHelper::isScaleWeightGroup(const ParsedWeight& weight) {
    return (weight.groupname.find("scale_variation") != std::string::npos ||
            weight.groupname.find("Central scale variation") != std::string::npos);
  }

  bool WeightHelper::isPdfWeightGroup(const ParsedWeight& weight) {
    const std::string& name = weight.groupname;

    if (name.find("PDF_variation") != std::string::npos)
      return true;
    return LHAPDF::lookupLHAPDFID(name) != -1;
  }

  bool WeightHelper::isPartonShowerWeightGroup(const ParsedWeight& weight) {
    const std::string& name = boost::to_lower_copy(weight.groupname);
    // But "Nominal" and "Baseline" weights in the PS group
    return name.find("isr") != std::string::npos || name.find("fsr") != std::string::npos ||
           name.find("nominal") != std::string::npos || name.find("baseline") != std::string::npos;
  }

  bool WeightHelper::isOrphanPdfWeightGroup(ParsedWeight& weight) {
    std::string lhaidText = searchAttributes("pdf", weight);
    try {
      auto pairLHA = LHAPDF::lookupPDF(stoi(lhaidText));
      // require pdf set to exist and it to be the first entry (ie 0)
      // possibly change this requirement
      if (!pairLHA.first.empty() && pairLHA.second == 0) {
        weight.groupname = std::string(pairLHA.first);
        return true;
      }
    } catch (...) {
      return false;
    }
    return false;
  }

  bool WeightHelper::isMEParamWeightGroup(const ParsedWeight& weight) {
    return (weight.groupname.find("mg_reweighting") != std::string::npos ||
            weight.groupname.find("variation") != std::string::npos);
    // variation used for blanket of all variations, might need to change
  }

  std::string WeightHelper::searchAttributes(const std::string& label, const ParsedWeight& weight) const {
    std::string attribute = searchAttributesByTag(label, weight);
    return attribute.empty() ? searchAttributesByRegex(label, weight) : attribute;
    attribute = searchAttributesByRegex(label, weight);
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

  std::string WeightHelper::searchString(const std::string& label, const std::string& name) {
    for (const auto& lab : attributeNames_.at(label)) {
      if (name.find(lab) != std::string::npos)
        return name.substr(0, name.find(lab));
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

  void WeightHelper::updateScaleInfo(const ParsedWeight& weight, int index) {
    auto& group = weightGroups_[index];
    auto& scaleGroup = dynamic_cast<gen::ScaleWeightGroupInfo&>(group);
    std::string muRText = searchAttributes("mur", weight);
    std::string muFText = searchAttributes("muf", weight);

    if (muRText.empty() || muFText.empty()) {
      scaleGroup.setIsWellFormed(false);
      return;
    }

    try {
      float muR = std::stof(muRText);
      float muF = std::stof(muFText);
      std::string dynNumText = searchAttributes("dyn", weight);
      if (dynNumText.empty()) {
        scaleGroup.setMuRMuFIndex(weight.index, weight.id, muR, muF);
      } else {
        std::string dynType = searchAttributes("dyn_name", weight);
        int dynNum = std::stoi(dynNumText);
        scaleGroup.setMuRMuFIndex(weight.index, weight.id, muR, muF, dynNum, dynType);
      }
    } catch (std::invalid_argument& e) {
      scaleGroup.setIsWellFormed(false);
    }
    if (scaleGroup.lhaid() == -1) {
      std::string lhaidText = searchAttributes("pdf", weight);
      try {
        scaleGroup.setLhaid(std::stoi(lhaidText));
      } catch (std::invalid_argument& e) {
        scaleGroup.setLhaid(-2);
      }
    }
  }

  int WeightHelper::lhapdfId(const ParsedWeight& weight, gen::PdfWeightGroupInfo& pdfGroup) {
    std::string lhaidText = searchAttributes("pdf", weight);
    if (!lhaidText.empty()) {
      try {
        return std::stoi(lhaidText);
      } catch (std::invalid_argument& e) {
        pdfGroup.setIsWellFormed(false);
      }
    } else if (pdfGroup.lhaIds().size() > 0) {
      return pdfGroup.lhaIds().back() + 1;
    } else {
      return LHAPDF::lookupLHAPDFID(weight.groupname);
    }
    return -1;
  }

  void WeightHelper::updatePdfInfo(const ParsedWeight& weight, int index) {
    auto& pdfGroup = dynamic_cast<gen::PdfWeightGroupInfo&>(weightGroups_[index]);
    int lhaid = lhapdfId(weight, pdfGroup);
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

  // TODO: Could probably recycle this code better
  std::unique_ptr<GenWeightProduct> WeightHelper::weightProduct(std::vector<double> weights, float w0) {
    auto weightProduct = std::make_unique<GenWeightProduct>(w0);
    weightProduct->setNumWeightSets(weightGroups_.size());
    int weightGroupIndex = 0;
    // This happens if there are no PS weights, so the weights vector contains only the central GEN weight.
    // Just add an empty product
    if (weights.size() > 1) {
      for (unsigned int i = 0; i < weights.size(); i++) {
        std::string id = std::to_string(i);
        addWeightToProduct(weightProduct, weights.at(i), id, i, weightGroupIndex);
      }
    }
    return std::move(weightProduct);
  }

  void WeightHelper::cleanupOrphanCentralWeight() {
    std::vector<int> removeList;
    for (auto it = weightGroups_.begin(); it < weightGroups_.end(); it++) {
      if (it->weightType() != WeightType::kScaleWeights)
        continue;
      auto& baseWeight = dynamic_cast<gen::ScaleWeightGroupInfo&>(*it);
      if (baseWeight.containsCentralWeight())
        continue;
      for (auto subIt = weightGroups_.begin(); subIt < it; subIt++) {
        if (subIt->weightType() != WeightType::kPdfWeights)
          continue;
        auto& subWeight = dynamic_cast<gen::PdfWeightGroupInfo&>(*subIt);
        if (subWeight.nIdsContained() == 1 && subWeight.parentLhapdfId() == baseWeight.lhaid()) {
          removeList.push_back(subIt - weightGroups_.begin());
          auto info = subWeight.idsContained().at(0);
          baseWeight.addContainedId(info.globalIndex, info.id, info.label, 1, 1);
        }
      }
    }
    std::sort(removeList.begin(), removeList.end(), std::greater<int>());
    for (auto idx : removeList) {
      weightGroups_.erase(weightGroups_.begin() + idx);
    }
  }

  std::unique_ptr<GenWeightProduct> WeightHelper::weightProduct(std::vector<gen::WeightsInfo> weights, float w0) {
    auto weightProduct = std::make_unique<GenWeightProduct>(w0);
    weightProduct->setNumWeightSets(weightGroups_.size());
    int weightGroupIndex = 0;
    int i = 0;
    for (const auto& weight : weights) {
      weightGroupIndex = addWeightToProduct(weightProduct, weight.wgt, weight.id, i++, weightGroupIndex);
    }
    return std::move(weightProduct);
  }

  int WeightHelper::addWeightToProduct(
      std::unique_ptr<GenWeightProduct>& product, double weight, std::string name, int weightNum, int groupIndex) {
    groupIndex = findContainingWeightGroup(name, weightNum, groupIndex);
    auto group = weightGroups_[groupIndex];
    int entry = group.weightVectorEntry(name, weightNum);
    if (debug_)
        std::cout << "Adding weight " << entry << " to group " << groupIndex << std::endl;
    product->addWeight(weight, groupIndex, entry);
    return groupIndex;
  }

  int WeightHelper::findContainingWeightGroup(std::string wgtId, int weightIndex, int previousGroupIndex) {
    // Start search at previous index, under expectation of ordered weights
    previousGroupIndex = previousGroupIndex >= 0 ? previousGroupIndex : 0;
    for (int index = previousGroupIndex; index < std::min(index + 1, static_cast<int>(weightGroups_.size())); index++) {
      const gen::WeightGroupInfo& weightGroup = weightGroups_[index];
      if (weightGroup.indexInRange(weightIndex) && weightGroup.containsWeight(wgtId, weightIndex)) {
        return static_cast<int>(index);
      }
    }

    // Fall back to unordered search
    int counter = 0;
    for (auto weightGroup : weightGroups_) {
      if (weightGroup.containsWeight(wgtId, weightIndex))
        return counter;
      counter++;
    }
    // Needs to be properly handled
    throw std::range_error("Unmatched Generator weight! ID was " + wgtId + " index was " + std::to_string(weightIndex) +
                           "\nNot found in any of " + std::to_string(weightGroups_.size()) + " weightGroups.");
  }

  void WeightHelper::printWeights() {
    // checks
    for (auto& wgt : weightGroups_) {
      if (!wgt.isWellFormed())
        std::cout << "\033[1;31m";
      std::cout << std::boolalpha << wgt.name() << " (" << wgt.firstId() << "-" << wgt.lastId()
                << "): " << wgt.isWellFormed() << std::endl;
      if (wgt.weightType() == gen::WeightType::kScaleWeights) {
        auto& wgtScale = dynamic_cast<gen::ScaleWeightGroupInfo&>(wgt);
        std::cout << wgtScale.centralIndex() << " ";
        std::cout << wgtScale.muR1muF2Index() << " ";
        std::cout << wgtScale.muR1muF05Index() << " ";
        std::cout << wgtScale.muR2muF1Index() << " ";
        std::cout << wgtScale.muR2muF2Index() << " ";
        std::cout << wgtScale.muR2muF05Index() << " ";
        std::cout << wgtScale.muR05muF1Index() << " ";
        std::cout << wgtScale.muR05muF2Index() << " ";
        std::cout << wgtScale.muR05muF05Index() << " \n";
        for (auto name : wgtScale.getDynNames()) {
          std::cout << name << ": ";
          std::cout << wgtScale.getScaleIndex(1.0, 1.0, name) << " ";
          std::cout << wgtScale.getScaleIndex(1.0, 2.0, name) << " ";
          std::cout << wgtScale.getScaleIndex(1.0, 0.5, name) << " ";
          std::cout << wgtScale.getScaleIndex(2.0, 1.0, name) << " ";
          std::cout << wgtScale.getScaleIndex(2.0, 2.0, name) << " ";
          std::cout << wgtScale.getScaleIndex(2.0, 0.5, name) << " ";
          std::cout << wgtScale.getScaleIndex(0.5, 1.0, name) << " ";
          std::cout << wgtScale.getScaleIndex(0.5, 2.0, name) << " ";
          std::cout << wgtScale.getScaleIndex(0.5, 0.5, name) << "\n";
        }

      } else if (wgt.weightType() == gen::WeightType::kPdfWeights) {
        std::cout << wgt.description() << "\n";
      } else if (wgt.weightType() == gen::WeightType::kPartonShowerWeights) {
        auto& wgtPS = dynamic_cast<gen::PartonShowerWeightGroupInfo&>(wgt);
        for (auto group : wgtPS.getWeightNames()) {
          std::cout << group << ": up " << wgtPS.upIndex(group);
          std::cout << " - down " << wgtPS.downIndex(group) << std::endl;
        }
      }
      if (!wgt.isWellFormed())
        std::cout << "\033[0m";
    }
  }

  std::unique_ptr<WeightGroupInfo> WeightHelper::buildGroup(ParsedWeight& weight) {
    if (debug_) {
        std::cout << "Building group for weight group " << weight.groupname 
                  << " weight content is " << weight.content << std::endl;
    }
    if (isScaleWeightGroup(weight))
      return std::make_unique<ScaleWeightGroupInfo>(weight.groupname);
    else if (isPdfWeightGroup(weight))
      return std::make_unique<PdfWeightGroupInfo>(weight.groupname);
    else if (isMEParamWeightGroup(weight))
      return std::make_unique<MEParamWeightGroupInfo>(weight.groupname);
    else if (isPartonShowerWeightGroup(weight))
      return std::make_unique<PartonShowerWeightGroupInfo>("shower");
    else if (isOrphanPdfWeightGroup(weight))
      return std::make_unique<PdfWeightGroupInfo>(weight.groupname);

    return std::make_unique<UnknownWeightGroupInfo>(weight.groupname);
  }

  void WeightHelper::buildGroups() {
    weightGroups_.clear();
    int currentGroupIdx = 0;
    int groupOffset = 0;
    for (auto& weight : parsedWeights_) {
      weight.wgtGroup_idx += groupOffset;
      currentGroupIdx = weight.wgtGroup_idx;
      if (debug_)
        std::cout << "Building group for weight " << weight.content << " group " 
                  << weight.groupname << " group index " << weight.wgtGroup_idx << std::endl;

      int numGroups = static_cast<int>(weightGroups_.size());
      if (weight.wgtGroup_idx == numGroups) {
        weightGroups_.push_back(*buildGroup(weight));
      } else if (weight.wgtGroup_idx >= numGroups)
        throw std::range_error("Invalid group index " + currentGroupIdx);

      // split PDF groups
      if (weightGroups_[currentGroupIdx].weightType() == gen::WeightType::kPdfWeights) {
        auto& pdfGroup = dynamic_cast<gen::PdfWeightGroupInfo&>(weightGroups_[currentGroupIdx]);
        int lhaid = lhapdfId(weight, pdfGroup);
        if (lhaid > 0 && !pdfGroup.isIdInParentSet(lhaid) && pdfGroup.parentLhapdfId() > 0) {
          weightGroups_.push_back(*buildGroup(weight));
          groupOffset++;
        }
      }
      WeightGroupInfo& group = weightGroups_[currentGroupIdx];
      group.addContainedId(weight.index, weight.id, weight.content);
      if (group.weightType() == gen::WeightType::kScaleWeights)
        updateScaleInfo(weight, currentGroupIdx);
      else if (group.weightType() == gen::WeightType::kPdfWeights) {
        updatePdfInfo(weight, currentGroupIdx);
      }
    }
    cleanupOrphanCentralWeight();
  }

}  // namespace gen
