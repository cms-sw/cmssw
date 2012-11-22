#include "../interface/IntegrityTask.h"

#include "FWCore/Utilities/interface/Exception.h"

namespace ecaldqm {

  IntegrityTask::IntegrityTask(const edm::ParameterSet &_params, const edm::ParameterSet& _paths) :
    DQWorkerTask(_params, _paths, "IntegrityTask"),
    hltTaskMode_(0),
    hltTaskFolder_("")
  {
    collectionMask_ = 
      (0x1 << kLumiSection) |
      (0x1 << kGainErrors) |
      (0x1 << kChIdErrors) |
      (0x1 << kGainSwitchErrors) |
      (0x1 << kTowerIdErrors) |
      (0x1 << kBlockSizeErrors);

    edm::ParameterSet const& commonParams(_params.getUntrackedParameterSet("Common"));

    hltTaskMode_ = commonParams.getUntrackedParameter<int>("hltTaskMode");
    hltTaskFolder_ = commonParams.getUntrackedParameter<std::string>("hltTaskFolder");

    if(hltTaskMode_ != 0 && hltTaskFolder_.size() == 0)
	throw cms::Exception("InvalidConfiguration") << "HLTTask mode needs a folder name";

    if(hltTaskMode_ != 0){
      std::string path;
      std::map<std::string, std::string> replacements;
      replacements["hlttask"] = hltTaskFolder_;

      MEs_[kFEDNonFatal]->name(replacements);
    }
  }

  IntegrityTask::~IntegrityTask()
  {
  }

  void
  IntegrityTask::bookMEs()
  {
    if(hltTaskMode_ != 1){
      for(unsigned iME(kByLumi); iME < kFEDNonFatal; iME++)
	MEs_[iME]->book();
    }
    if(hltTaskMode_ != 0)
      MEs_[kFEDNonFatal]->book();
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
  IntegrityTask::setMEData(std::vector<MEData>& _data)
  {
    _data[kByLumi] = MEData("ByLumi", BinService::kEcal2P, BinService::kDCC, MonitorElement::DQM_KIND_TH1F);
    _data[kTotal] = MEData("Total", BinService::kEcal2P, BinService::kDCC, MonitorElement::DQM_KIND_TH1F);
    _data[kGain] = MEData("Gain", BinService::kChannel, BinService::kCrystal, MonitorElement::DQM_KIND_TH1F);
    _data[kChId] = MEData("ChId", BinService::kChannel, BinService::kCrystal, MonitorElement::DQM_KIND_TH1F);
    _data[kGainSwitch] = MEData("GainSwitch", BinService::kChannel, BinService::kCrystal, MonitorElement::DQM_KIND_TH1F);
    _data[kBlockSize] = MEData("BlockSize", BinService::kChannel, BinService::kSuperCrystal, MonitorElement::DQM_KIND_TH1F);
    _data[kTowerId] = MEData("TowerId", BinService::kChannel, BinService::kSuperCrystal, MonitorElement::DQM_KIND_TH1F);
    _data[kFEDNonFatal] = MEData("FEDNonFatal", BinService::kEcal2P, BinService::kDCC, MonitorElement::DQM_KIND_TH1F);
  }

  DEFINE_ECALDQM_WORKER(IntegrityTask);
}


