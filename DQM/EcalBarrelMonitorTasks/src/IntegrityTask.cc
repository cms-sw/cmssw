#include "../interface/IntegrityTask.h"

#include "FWCore/Utilities/interface/Exception.h"

namespace ecaldqm {

  IntegrityTask::IntegrityTask(edm::ParameterSet const& _workerParams, edm::ParameterSet const& _commonParams) :
    DQWorkerTask(_workerParams, _commonParams, "IntegrityTask"),
    hltTaskMode_(_commonParams.getUntrackedParameter<int>("hltTaskMode"))
  {
    collectionMask_[kLumiSection] = true;
    collectionMask_[kGainErrors] = true;
    collectionMask_[kChIdErrors] = true;
    collectionMask_[kGainSwitchErrors] = true;
    collectionMask_[kTowerIdErrors] = true;
    collectionMask_[kBlockSizeErrors] = true;
  }

  void
  IntegrityTask::bookMEs()
  {
    if(hltTaskMode_ != 1){
      for(MESetCollection::iterator mItr(MEs_.begin()); mItr != MEs_.end(); ++mItr){
        if(mItr->first == "FEDNonFatal") continue;
	mItr->second->book();
      }
      MEs_["ByLumi"]->setLumiFlag();
    }
    if(hltTaskMode_ != 0){
      MEs_["FEDNonFatal"]->book();
      MEs_["FEDNonFatal"]->getME(0)->getTH1()->GetXaxis()->SetLimits(601., 655.);
    }
  }

  void
  IntegrityTask::beginLuminosityBlock(const edm::LuminosityBlock &, const edm::EventSetup &)
  {
    MEs_["ByLumi"]->reset();
  }

  void
  IntegrityTask::runOnErrors(const DetIdCollection &_ids, Collections _collection)
  {
    if(_ids.size() == 0) return;

    MESet* set(0);
    switch(_collection){
    case kGainErrors:
      set = MEs_["Gain"];
      break;
    case kChIdErrors:
      set = MEs_["ChId"];
      break;
    case kGainSwitchErrors:
      set = MEs_["GainSwitch"];
      break;
    default:
      return;
    }

    MESet* meFEDNonFatal(MEs_["FEDNonFatal"]);
    MESet* meByLumi(MEs_["ByLumi"]);
    MESet* meTotal(MEs_["Total"]);
    MESet* meTrendNErrors(online ? MEs_["TrendNErrors"] : 0);

    for(DetIdCollection::const_iterator idItr(_ids.begin()); idItr != _ids.end(); ++idItr){
      set->fill(*idItr);
      unsigned dccid(dccId(*idItr));
      meFEDNonFatal->fill(dccid);
      meByLumi->fill(dccid);
      meTotal->fill(dccid);

      if(online) meTrendNErrors->fill(double(iLumi), 1.);
    }
  }
  
  void
  IntegrityTask::runOnErrors(const EcalElectronicsIdCollection &_ids, Collections _collection)
  {
    if(_ids.size() == 0) return;

    MESet* set(0);
    switch(_collection){
    case kTowerIdErrors:
      set = MEs_["TowerId"];
      break;
    case kBlockSizeErrors:
      set = MEs_["BlockSize"];
      break;
    default:
      return;
    }

    MESet* meFEDNonFatal(MEs_["FEDNonFatal"]);
    MESet* meByLumi(MEs_["ByLumi"]);
    MESet* meTotal(MEs_["Total"]);
    MESet* meTrendNErrors(online ? MEs_["TrendNErrors"] : 0);

    for(EcalElectronicsIdCollection::const_iterator idItr(_ids.begin()); idItr != _ids.end(); ++idItr){
      set->fill(*idItr);
      unsigned dccid(idItr->dccId());
      double nCrystals(0.);
      if(dccid <= kEEmHigh + 1 || dccid >= kEEpLow + 1)
        nCrystals = getElectronicsMap()->dccTowerConstituents(dccid, idItr->towerId()).size();
      else
        nCrystals = 25.;
      meFEDNonFatal->fill(dccid, nCrystals);
      meByLumi->fill(dccid, nCrystals);
      meTotal->fill(dccid, nCrystals);

      if(online) meTrendNErrors->fill(double(iLumi), nCrystals);
    }
  }

  DEFINE_ECALDQM_WORKER(IntegrityTask);
}


