#include "../interface/RawDataClient.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"
#include "DQM/EcalCommon/interface/FEFlags.h"

#include "CondFormats/EcalObjects/interface/EcalDQMStatusHelper.h"

#include <cmath>

namespace ecaldqm {

  RawDataClient::RawDataClient(edm::ParameterSet const& _workerParams, edm::ParameterSet const& _commonParams) :
    DQWorkerClient(_workerParams, _commonParams, "RawDataClient"),
    synchErrThresholdFactor_(_workerParams.getUntrackedParameter<double>("synchErrThresholdFactor"))
  {
    qualitySummaries_.insert(kQualitySummary);
  }

  void
  RawDataClient::producePlots()
  {
    uint32_t mask(1 << EcalDQMStatusHelper::STATUS_FLAG_ERROR);

    std::vector<int> dccStatus(BinService::nDCC, 1);

    for(unsigned iDCC(0); iDCC < BinService::nDCC; ++iDCC){
      double entries(sources_[kEntries]->getBinContent(iDCC + 1));
      if(entries > 1. && sources_[kL1ADCC]->getBinContent(iDCC + 1) > synchErrThresholdFactor_ * std::log(entries) / std::log(10.))
        dccStatus[iDCC] = 0;
    }

    MESet::iterator meEnd(MEs_[kQualitySummary]->end());
    for(MESet::iterator meItr(MEs_[kQualitySummary]->beginChannel()); meItr != meEnd; meItr.toNextChannel()){

      DetId id(meItr->getId());

      bool doMask(applyMask_(kQualitySummary, id, mask));

      unsigned dccid(dccId(id));

      if(dccStatus[dccid - 1] == 0){
        meItr->setBinContent(doMask ? kMUnknown : kUnknown);
        continue;
      }

      int towerStatus(doMask ? kMGood : kGood);
      float towerEntries(0.);
      for(unsigned iS(0); iS < nFEFlags; iS++){
        float entries(sources_[kFEStatus]->getBinContent(id, iS + 1));
        towerEntries += entries;
        if(entries > 0. &&
           iS != Enabled && iS != Disabled && iS != Suppressed &&
           iS != FIFOFull && iS != FIFOFullL1ADesync && iS != ForcedZS)
          towerStatus = doMask ? kMBad : kBad;
      }

      if(towerEntries < 1.) towerStatus = doMask ? kMUnknown : kUnknown;

      meItr->setBinContent(towerStatus);

    }
  }

  /*static*/
  void
  RawDataClient::setMEOrdering(std::map<std::string, unsigned>& _nameToIndex)
  {
    _nameToIndex["QualitySummary"] = kQualitySummary;

    _nameToIndex["Entries"] = kEntries;
    _nameToIndex["L1ADCC"] = kL1ADCC;
    _nameToIndex["FEStatus"] = kFEStatus;
  }

  DEFINE_ECALDQM_WORKER(RawDataClient);
}

