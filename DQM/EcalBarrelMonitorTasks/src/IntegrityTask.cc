#include "../interface/IntegrityTask.h"

#include "FWCore/Utilities/interface/Exception.h"

namespace ecaldqm {

  IntegrityTask::IntegrityTask(edm::ParameterSet const& _workerParams, edm::ParameterSet const& _commonParams) :
    DQWorkerTask(_workerParams, _commonParams, "IntegrityTask"),
    hltTaskMode_(_commonParams.getUntrackedParameter<int>("hltTaskMode"))
  {
    collectionMask_ = 
      (0x1 << kLumiSection) |
      (0x1 << kGainErrors) |
      (0x1 << kChIdErrors) |
      (0x1 << kGainSwitchErrors) |
      (0x1 << kTowerIdErrors) |
      (0x1 << kBlockSizeErrors);
  }

  void
  IntegrityTask::bookMEs()
  {
    if(hltTaskMode_ != 1){
      for(unsigned iME(0); iME < nMESets; iME++){
        if(iME == kFEDNonFatal) continue;
        if(iME == kTrendNErrors && !online) continue;
	MEs_[iME]->book();
      }
      MEs_[kByLumi]->setLumiFlag();
    }
    if(hltTaskMode_ != 0){
      MEs_[kFEDNonFatal]->book();
      MEs_[kFEDNonFatal]->getME(0)->getTH1()->GetXaxis()->SetLimits(601., 655.);
    }
  }

  void
  IntegrityTask::beginLuminosityBlock(const edm::LuminosityBlock &, const edm::EventSetup &)
  {
    MEs_[kByLumi]->reset();
  }

  void
  IntegrityTask::runOnErrors(const DetIdCollection &_ids, Collections _collection)
  {
    MESets set(nMESets);
    switch(_collection){
    case kGainErrors:
      set = kGain;
      break;
    case kChIdErrors:
      set = kChId;
      break;
    case kGainSwitchErrors:
      set = kGainSwitch;
      break;
    default:
      return;
    }

    for(DetIdCollection::const_iterator idItr(_ids.begin()); idItr != _ids.end(); ++idItr){
      MEs_[set]->fill(*idItr);
      unsigned dccid(dccId(*idItr));
      MEs_[kFEDNonFatal]->fill(dccid);
      MEs_[kByLumi]->fill(dccid);
      MEs_[kTotal]->fill(dccid);

      if(online) MEs_[kTrendNErrors]->fill(double(iLumi), 1.);
    }
  }
  
  void
  IntegrityTask::runOnErrors(const EcalElectronicsIdCollection &_ids, Collections _collection)
  {
    MESets set(nMESets);

    switch(_collection){
    case kTowerIdErrors:
      set = kTowerId;
      break;
    case kBlockSizeErrors:
      set = kBlockSize;
      break;
    default:
      return;
    }

    // 25 is not correct

    for(EcalElectronicsIdCollection::const_iterator idItr(_ids.begin()); idItr != _ids.end(); ++idItr){
      MEs_[set]->fill(*idItr);
      unsigned dccid(idItr->dccId());
      double nCrystals(0.);
      if(dccid <= kEEmHigh + 1 || dccid >= kEEpLow + 1)
        nCrystals = getElectronicsMap()->dccTowerConstituents(dccid, idItr->towerId()).size();
      else
        nCrystals = 25.;
      MEs_[kFEDNonFatal]->fill(dccid, nCrystals);
      MEs_[kByLumi]->fill(dccid, nCrystals);
      MEs_[kTotal]->fill(dccid, nCrystals);

      if(online) MEs_[kTrendNErrors]->fill(double(iLumi), nCrystals);
    }
  }

  /*static*/
  void
  IntegrityTask::setMEOrdering(std::map<std::string, unsigned>& _nameToIndex)
  {
    _nameToIndex["ByLumi"] = kByLumi;
    _nameToIndex["Total"] = kTotal;
    _nameToIndex["Gain"] = kGain;
    _nameToIndex["ChId"] = kChId;
    _nameToIndex["GainSwitch"] = kGainSwitch;
    _nameToIndex["BlockSize"] = kBlockSize;
    _nameToIndex["TowerId"] = kTowerId;
    _nameToIndex["FEDNonFatal"] = kFEDNonFatal;
    _nameToIndex["TrendNErrors"] = kTrendNErrors;
  }

  DEFINE_ECALDQM_WORKER(IntegrityTask);
}


