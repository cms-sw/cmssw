#include "L1Trigger/L1CaloTrigger/interface/ParametricCalibration.h"

namespace l1tp2 {
  ParametricCalibration::ParametricCalibration(const edm::ParameterSet& cpset) {
    std::vector<double> etaBins = cpset.getParameter<std::vector<double>>("etaBins");
    std::vector<double> ptBins = cpset.getParameter<std::vector<double>>("ptBins");
    std::vector<double> scale = cpset.getParameter<std::vector<double>>("scale");
    etas.insert(etas.end(), etaBins.begin(), etaBins.end());
    pts.insert(pts.end(), ptBins.begin(), ptBins.end());
    scales.insert(scales.end(), scale.begin(), scale.end());

    std::vector<double> ptMin = cpset.getParameter<std::vector<double>>("ptMin");
    if (ptMin.size() == etaBins.size()) {
      ptMins.insert(ptMins.end(), ptMin.begin(), ptMin.end());
    } else if (ptMin.size() == 1) {
      ptMins = std::vector<float>(etaBins.size(), ptMin[0]);
    } else {
      throw cms::Exception("ParametricCalibration Configuration", "Ambiguous number of ptMin values in configuration");
      ;
    }

    std::vector<double> ptMax = cpset.getParameter<std::vector<double>>("ptMax");
    if (ptMax.size() == etaBins.size()) {
      ptMaxs.insert(ptMaxs.end(), ptMax.begin(), ptMax.end());
    } else if (ptMax.size() == 1) {
      ptMaxs = std::vector<float>(etaBins.size(), ptMax[0]);
    } else {
      throw cms::Exception("ParametricCalibration Configuration", "Ambiguous number of ptMax values in configuration");
      ;
    }

    if (pts.size() * etas.size() != scales.size())
      throw cms::Exception("Configuration",
                           "Bad number of calibration scales, pts.size() * etas.size() != scales.size()");
  }
  // ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
  void ParametricCalibration::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.setComment("");
    desc.addUntracked<std::vector<double>>("ptMin", std::vector<double>{});
    desc.addUntracked<std::vector<double>>("ptMax", std::vector<double>{});
    descriptions.add("createIdealTkAlRecords", desc);
  }

};  // namespace l1tp2
