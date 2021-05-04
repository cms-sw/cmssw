#include "L1Trigger/L1CaloTrigger/interface/ParametricCalibration.h"

namespace l1tp2 {
  ParametricCalibration::ParametricCalibration(const edm::ParameterSet& cpset) {
    std::vector<double> etaBins = cpset.getParameter<std::vector<double>>("etaBins");
    std::vector<double> ptBins = cpset.getParameter<std::vector<double>>("ptBins");
    std::vector<double> scale = cpset.getParameter<std::vector<double>>("scale");
    etas.insert(etas.end(), etaBins.begin(), etaBins.end());
    pts.insert(pts.end(), ptBins.begin(), ptBins.end());
    scales.insert(scales.end(), scale.begin(), scale.end());

    if (pts.size() * etas.size() != scales.size())
      throw cms::Exception("Configuration",
                           "Bad number of calibration scales, pts.size() * etas.size() != scales.size()");
  }

  // ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
  void ParametricCalibration::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.setComment("");
    desc.addUntracked<std::vector<double>>("etaBins", std::vector<double>{});
    desc.addUntracked<std::vector<double>>("ptBins", std::vector<double>{});
    desc.addUntracked<std::vector<double>>("scale", std::vector<double>{});
    descriptions.add("createIdealTkAlRecords", desc);
  }

  float ParametricCalibration::operator()(const float pt, const float abseta) const {
    int ptBin = -1;
    for (unsigned int i = 0, n = pts.size(); i < n; ++i) {
      if (pt < pts[i]) {
        ptBin = i;
        break;
      }
    }
    int etaBin = -1;
    for (unsigned int i = 0, n = etas.size(); i < n; ++i) {
      if (abseta < etas[i]) {
        etaBin = i;
        break;
      }
    }

    if (ptBin == -1 || etaBin == -1)
      return 1;
    else
      return scales[ptBin * etas.size() + etaBin];
  }

};  // namespace l1tp2
