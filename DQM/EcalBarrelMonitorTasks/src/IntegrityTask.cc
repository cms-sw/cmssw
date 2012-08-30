#include "../interface/IntegrityTask.h"

#include "FWCore/Utilities/interface/Exception.h"

namespace ecaldqm {

  IntegrityTask::IntegrityTask(edm::ParameterSet const& _workerParams, edm::ParameterSet const& _commonParams) :
    DQWorkerTask(_workerParams, _commonParams, "IntegrityTask"),
    hltTaskMode_(_commonParams.getUntrackedParameter<int>("hltTaskMode")),
    hltTaskFolder_(_commonParams.getUntrackedParameter<std::string>("hltTaskFolder"))
  {
    collectionMask_ = 
      (0x1 << kLumiSection) |
      (0x1 << kGainErrors) |
      (0x1 << kChIdErrors) |
      (0x1 << kGainSwitchErrors) |
      (0x1 << kTowerIdErrors) |
      (0x1 << kBlockSizeErrors);

    if(hltTaskMode_ != 0 && hltTaskFolder_.size() == 0)
	throw cms::Exception("InvalidConfiguration") << "HLTTask mode needs a folder name";

    if(hltTaskMode_ != 0){
      std::string path;
      std::map<std::string, std::string> replacements;
      replacements["hlttask"] = hltTaskFolder_;

      MEs_[kFEDNonFatal]->formPath(replacements);
    }
  }

  void
  IntegrityTask::bookMEs()
  {
    if(hltTaskMode_ != 1){
      for(unsigned iME(kByLumi); iME < kFEDNonFatal; iME++)
	MEs_[iME]->book();
    }
    if(hltTaskMode_ != 0){
      MEs_[kFEDNonFatal]->book();
      MEs_[kFEDNonFatal]->getME(0)->getTH1()->GetXaxis()->SetLimits(601., 655.);
    }
  }

  void
  IntegrityTask::beginLuminosityBlock(const edm::LuminosityBlock &, const edm::EventSetup &)
  {
    if(MEs_[kByLumi]->isActive()) MEs_[kByLumi]->reset();
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
      if(MEs_[set]->isActive()) MEs_[set]->fill(*idItr);
      if(MEs_[kFEDNonFatal]->isActive()) MEs_[kFEDNonFatal]->fill(*idItr);
      if(MEs_[kByLumi]->isActive()) MEs_[kByLumi]->fill(*idItr);
      if(MEs_[kTotal]->isActive()) MEs_[kTotal]->fill(*idItr);
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
      if(MEs_[set]->isActive()) MEs_[set]->fill(*idItr, 25.);
      if(MEs_[kFEDNonFatal]->isActive()) MEs_[kFEDNonFatal]->fill(*idItr, 25.);
      if(MEs_[kByLumi]->isActive()) MEs_[kByLumi]->fill(*idItr, 25.);
      if(MEs_[kTotal]->isActive()) MEs_[kTotal]->fill(*idItr, 25.);
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
  }

  DEFINE_ECALDQM_WORKER(IntegrityTask);
}


