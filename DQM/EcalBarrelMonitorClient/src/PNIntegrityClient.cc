#include "../interface/PNIntegrityClient.h"

#include "DataFormats/EcalDetId/interface/EcalPnDiodeDetId.h"

#include "CondFormats/EcalObjects/interface/EcalDQMStatusHelper.h"

namespace ecaldqm
{
  PNIntegrityClient::PNIntegrityClient(edm::ParameterSet const& _workerParams, edm::ParameterSet const& _commonParams) :
    DQWorkerClient(_workerParams, _commonParams, "PNIntegrityClient"),
    errFractionThreshold_(_workerParams.getUntrackedParameter<double>("errFractionThreshold"))
  {
  }

  void
  PNIntegrityClient::beginRun(edm::Run const&, edm::EventSetup const&)
  {
    MEs_[kQualitySummary]->resetAll(-1.);
    MEs_[kQualitySummary]->reset(kUnknown);
  }

  void
  PNIntegrityClient::producePlots()
  {
    uint32_t mask(0x1 << EcalDQMStatusHelper::TT_SIZE_ERROR |
                  0x1 << EcalDQMStatusHelper::TT_ID_ERROR |
                  0x1 << EcalDQMStatusHelper::CH_ID_ERROR |
                  0x1 << EcalDQMStatusHelper::CH_GAIN_ZERO_ERROR);

    for(unsigned iDCC(0); iDCC < BinService::nDCC; ++iDCC){
      if(memDCCIndex(iDCC + 1) == unsigned(-1)) continue;
      for(unsigned iPN(0); iPN < 10; ++iPN){
        int subdet(0);
        if(iDCC >= kEBmLow && iDCC <= kEBpHigh) subdet = EcalBarrel;
        else subdet = EcalEndcap;

        EcalPnDiodeDetId id(subdet, iDCC + 1, iPN + 1);

        bool doMask(applyMask_(kQualitySummary, id, mask));

        float entries(sources_[kOccupancy]->getBinContent(id));

        float chid(sources_[kMEMChId]->getBinContent(id));
        float gain(sources_[kMEMGain]->getBinContent(id));

        float blocksize(sources_[kMEMBlockSize]->getBinContent(id));
        float towerid(sources_[kMEMTowerId]->getBinContent(id));

        if(entries + gain + chid + blocksize + towerid < 1.){
          MEs_[kQualitySummary]->setBinContent(id, doMask ? kMUnknown : kUnknown);
          continue;
        }

        float chErr((gain + chid + blocksize + towerid) / (entries + gain + chid + blocksize + towerid));

        if(chErr > errFractionThreshold_)
          MEs_[kQualitySummary]->setBinContent(id, doMask ? kMBad : kBad);
        else
          MEs_[kQualitySummary]->setBinContent(id, doMask ? kMGood : kGood);
      }
    }
  }

  /*static*/
  void
  PNIntegrityClient::setMEOrdering(std::map<std::string, unsigned>& _nameToIndex)
  {
    _nameToIndex["QualitySummary"] = kQualitySummary;

    _nameToIndex["Occupancy"] = kOccupancy;
    _nameToIndex["MEMChId"] = kMEMChId;
    _nameToIndex["MEMGain"] = kMEMGain;
    _nameToIndex["MEMBlockSize"] = kMEMBlockSize;
    _nameToIndex["MEMTowerId"] = kMEMTowerId;
  }

  DEFINE_ECALDQM_WORKER(PNIntegrityClient);
}
