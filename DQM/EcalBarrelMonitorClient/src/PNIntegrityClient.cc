#include "../interface/PNIntegrityClient.h"
#include "../interface/EcalDQMClientUtils.h"

#include "DataFormats/EcalDetId/interface/EcalPnDiodeDetId.h"

#include "CondFormats/EcalObjects/interface/EcalDQMStatusHelper.h"

namespace ecaldqm
{
  PNIntegrityClient::PNIntegrityClient(edm::ParameterSet const& _workerParams, edm::ParameterSet const& _commonParams) :
    DQWorkerClient(_workerParams, _commonParams, "PNIntegrityClient"),
    errFractionThreshold_(_workerParams.getUntrackedParameter<double>("errFractionThreshold"))
  {
    qualitySummaries_.insert("QualitySummary");
  }

  void
  PNIntegrityClient::producePlots()
  {
    uint32_t mask(0x1 << EcalDQMStatusHelper::TT_SIZE_ERROR |
                  0x1 << EcalDQMStatusHelper::TT_ID_ERROR |
                  0x1 << EcalDQMStatusHelper::CH_ID_ERROR |
                  0x1 << EcalDQMStatusHelper::CH_GAIN_ZERO_ERROR);

    MESet* meQualitySummary(MEs_["QualitySummary"]);

    MESet const* sOccupancy(sources_["Occupancy"]);
    MESet const* sMEMChId(sources_["MEMChId"]);
    MESet const* sMEMGain(sources_["MEMGain"]);
    MESet const* sMEMBlockSize(sources_["MEMBlockSize"]);
    MESet const* sMEMTowerId(sources_["MEMTowerId"]);

    for(unsigned iDCC(0); iDCC < BinService::nDCC; ++iDCC){
      if(memDCCIndex(iDCC + 1) == unsigned(-1)) continue;
      for(unsigned iPN(0); iPN < 10; ++iPN){
        int subdet(0);
        if(iDCC >= kEBmLow && iDCC <= kEBpHigh) subdet = EcalBarrel;
        else subdet = EcalEndcap;

        EcalPnDiodeDetId id(subdet, iDCC + 1, iPN + 1);

        bool doMask(applyMask(meQualitySummary->getBinType(), id, mask));

        float entries(sOccupancy->getBinContent(id));

        float chid(sMEMChId->getBinContent(id));
        float gain(sMEMGain->getBinContent(id));

        float blocksize(sMEMBlockSize->getBinContent(id));
        float towerid(sMEMTowerId->getBinContent(id));

        if(entries + gain + chid + blocksize + towerid < 1.){
          meQualitySummary->setBinContent(id, doMask ? kMUnknown : kUnknown);
          continue;
        }

        float chErr((gain + chid + blocksize + towerid) / (entries + gain + chid + blocksize + towerid));

        if(chErr > errFractionThreshold_)
          meQualitySummary->setBinContent(id, doMask ? kMBad : kBad);
        else
          meQualitySummary->setBinContent(id, doMask ? kMGood : kGood);
      }
    }
  }

  DEFINE_ECALDQM_WORKER(PNIntegrityClient);
}
