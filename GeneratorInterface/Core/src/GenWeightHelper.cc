#include "GeneratorInterface/Core/interface/GenWeightHelper.h"
#include <iostream>

using namespace tinyxml2;

namespace gen {
  GenWeightHelper::GenWeightHelper() {}

  std::vector<std::unique_ptr<gen::WeightGroupInfo>> GenWeightHelper::parseWeightGroupsFromNames(
      std::vector<std::string> weightNames, bool addUnassociated) const {
    std::vector<ParsedWeight> parsedWeights;
    int index = 0;
    int groupIndex = -1;
    int showerGroupIndex = -1;
    std::string curGroup = "";
    // If size is 1, it's just the central weight
    // Still give the opportunity to add an unassociated group
    if (weightNames.size() <= 1)
      return buildGroups(parsedWeights, addUnassociated);

    for (std::string weightName : weightNames) {
      if (weightName.find("LHE") != std::string::npos) {
        // Parse as usual, this is the SUSY workflow
        std::vector<std::string> info;
        boost::split(info, weightName, boost::is_any_of(","));
        std::unordered_map<std::string, std::string> attributes;
        std::string text = info.back();
        info.pop_back();
        for (auto i : info) {
          std::vector<std::string> subInfo;
          boost::split(subInfo, i, boost::is_any_of("="));
          if (subInfo.size() == 2) {
            attributes[boost::algorithm::trim_copy(subInfo[0])] = boost::algorithm::trim_copy(subInfo[1]);
          }
        }
        if (attributes["group"] != curGroup) {
          curGroup = attributes["group"];
          groupIndex++;
        }
        // Gen Weights can't have an ID, because they are just a std::vector<float> in the event
        attributes["id"] = "";
        parsedWeights.push_back({attributes["id"], index, curGroup, text, attributes, groupIndex});
      } else {
        parsedWeights.push_back(
            {"", index, weightName, weightName, std::unordered_map<std::string, std::string>(), groupIndex});
        if (isPartonShowerWeightGroup(parsedWeights.back())) {
          if (showerGroupIndex < 0) {
            showerGroupIndex = ++groupIndex;
          }
          parsedWeights.back().wgtGroup_idx = showerGroupIndex;  // all parton showers are grouped together
        }
      }
      index++;
    }
    auto groups = buildGroups(parsedWeights, addUnassociated);
    if (debug_)
      printWeights(groups);
    return groups;
  }
}  // namespace gen
