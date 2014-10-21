#include "../interface/RawDataClient.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"
#include "DQM/EcalCommon/interface/FEFlags.h"

#include "CondFormats/EcalObjects/interface/EcalDQMStatusHelper.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <cmath>

namespace ecaldqm {

  RawDataClient::RawDataClient() :
    DQWorkerClient(),
    synchErrThresholdFactor_(0.)
  {
    qualitySummaries_.insert("QualitySummary");
  }

  void
  RawDataClient::setParams(edm::ParameterSet const& _params)
  {
    synchErrThresholdFactor_ = _params.getUntrackedParameter<double>("synchErrThresholdFactor");
  }

  void
  RawDataClient::producePlots(ProcessType)
  {
    MESet& meQualitySummary(MEs_.at("QualitySummary"));
    MESet& meErrorsSummary(MEs_.at("ErrorsSummary"));

    MESet const& sEntries(sources_.at("Entries"));
    MESet const& sL1ADCC(sources_.at("L1ADCC"));
    MESet const& sFEStatus(sources_.at("FEStatus"));

    uint32_t mask(1 << EcalDQMStatusHelper::STATUS_FLAG_ERROR);

    std::vector<int> dccStatus(nDCC, 1);

    for(unsigned iDCC(0); iDCC < nDCC; ++iDCC){
      double entries(sEntries.getBinContent(iDCC + 1));
      if(entries > 1. && sL1ADCC.getBinContent(iDCC + 1) > synchErrThresholdFactor_ * std::log(entries) / std::log(10.))
        dccStatus[iDCC] = 0;
    }

    MESet::iterator meEnd(meQualitySummary.end());
    for(MESet::iterator meItr(meQualitySummary.beginChannel()); meItr != meEnd; meItr.toNextChannel()){

      DetId id(meItr->getId());

      bool doMask(meQualitySummary.maskMatches(id, mask, statusManager_));

      int dccid(dccId(id));

      if(dccStatus[dccid - 1] == 0){
        meItr->setBinContent(doMask ? kMUnknown : kUnknown);
        continue;
      }

      int towerStatus(doMask ? kMGood : kGood);
      float towerEntries(0.);
      for(unsigned iS(0); iS < nFEFlags; iS++){
        float entries(sFEStatus.getBinContent(id, iS + 1));
        towerEntries += entries;
        if(entries > 0. &&
           iS != Enabled && iS != Disabled && iS != Suppressed &&
           iS != FIFOFull && iS != FIFOFullL1ADesync && iS != ForcedZS)
          towerStatus = doMask ? kMBad : kBad;
      }

      if(towerEntries < 1.) towerStatus = doMask ? kMUnknown : kUnknown;

      meItr->setBinContent(towerStatus);
      if(towerStatus == kBad) meErrorsSummary.fill(dccid);
    }
  }

  DEFINE_ECALDQM_WORKER(RawDataClient);
}

