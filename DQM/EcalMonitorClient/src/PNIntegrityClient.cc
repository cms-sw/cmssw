#include "DQM/EcalMonitorClient/interface/PNIntegrityClient.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"

#include "DataFormats/EcalDetId/interface/EcalPnDiodeDetId.h"

#include "CondFormats/EcalObjects/interface/EcalDQMStatusHelper.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace ecaldqm {
  PNIntegrityClient::PNIntegrityClient() : DQWorkerClient(), errFractionThreshold_(0.) {
    qualitySummaries_.insert("QualitySummary");
  }

  void PNIntegrityClient::setParams(edm::ParameterSet const& _params) {
    errFractionThreshold_ = _params.getUntrackedParameter<double>("errFractionThreshold");
  }

  void PNIntegrityClient::producePlots(ProcessType) {
    uint32_t mask(0x1 << EcalDQMStatusHelper::TT_SIZE_ERROR | 0x1 << EcalDQMStatusHelper::TT_ID_ERROR |
                  0x1 << EcalDQMStatusHelper::CH_ID_ERROR | 0x1 << EcalDQMStatusHelper::CH_GAIN_ZERO_ERROR);

    MESet& meQualitySummary(MEs_.at("QualitySummary"));

    MESet const& sOccupancy(sources_.at("Occupancy"));
    MESet const& sMEMChId(sources_.at("MEMChId"));
    MESet const& sMEMGain(sources_.at("MEMGain"));
    MESet const& sMEMBlockSize(sources_.at("MEMBlockSize"));
    MESet const& sMEMTowerId(sources_.at("MEMTowerId"));

    for (unsigned iDCC(0); iDCC < nDCC; ++iDCC) {
      if (memDCCIndex(iDCC + 1) == unsigned(-1))
        continue;
      for (unsigned iPN(0); iPN < 10; ++iPN) {
        int subdet(0);
        if (iDCC >= kEBmLow && iDCC <= kEBpHigh)
          subdet = EcalBarrel;
        else
          subdet = EcalEndcap;

        EcalPnDiodeDetId id(subdet, iDCC + 1, iPN + 1);

        bool doMask(meQualitySummary.maskMatches(id, mask, statusManager_, GetTrigTowerMap()));

        float entries(sOccupancy.getBinContent(getEcalDQMSetupObjects(), id));

        float chid(sMEMChId.getBinContent(getEcalDQMSetupObjects(), id));
        float gain(sMEMGain.getBinContent(getEcalDQMSetupObjects(), id));

        float blocksize(sMEMBlockSize.getBinContent(getEcalDQMSetupObjects(), id));
        float towerid(sMEMTowerId.getBinContent(getEcalDQMSetupObjects(), id));

        if (entries + gain + chid + blocksize + towerid < 1.) {
          meQualitySummary.setBinContent(getEcalDQMSetupObjects(), id, doMask ? kMUnknown : kUnknown);
          continue;
        }

        float chErr((gain + chid + blocksize + towerid) / (entries + gain + chid + blocksize + towerid));

        if (chErr > errFractionThreshold_)
          meQualitySummary.setBinContent(getEcalDQMSetupObjects(), id, doMask ? kMBad : kBad);
        else
          meQualitySummary.setBinContent(getEcalDQMSetupObjects(), id, doMask ? kMGood : kGood);
      }
    }
  }

  DEFINE_ECALDQM_WORKER(PNIntegrityClient);
}  // namespace ecaldqm
