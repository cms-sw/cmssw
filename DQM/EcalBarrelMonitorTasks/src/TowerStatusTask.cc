#include "../interface/TowerStatusTask.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondFormats/EcalObjects/interface/EcalDAQTowerStatus.h"
#include "CondFormats/DataRecord/interface/EcalDAQTowerStatusRcd.h"
#include "CondFormats/EcalObjects/interface/EcalDCSTowerStatus.h"
#include "CondFormats/DataRecord/interface/EcalDCSTowerStatusRcd.h"

#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"
#include "DataFormats/EcalDetId/interface/EcalScDetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"

namespace ecaldqm {

  TowerStatusTask::TowerStatusTask(const edm::ParameterSet& _params, const edm::ParameterSet& _paths) :
    DQWorkerTask(_params, _paths, "TowerStatusTask"),
    daqLumiStatus_(),
    daqRunStatus_(),
    dcsLumiStatus_(),
    dcsRunStatus_(),
    doDAQInfo_(false),
    doDCSInfo_(false)
  {
    collectionMask_ = 
      (0x1 << kRun) |
      (0x1 << kLumiSection);

    edm::ParameterSet const& taskParams(_params.getUntrackedParameterSet(name_));

    doDAQInfo_ = taskParams.getUntrackedParameter<bool>("doDAQInfo");
    doDCSInfo_ = taskParams.getUntrackedParameter<bool>("doDCSInfo");

    if(!doDAQInfo_ && !doDCSInfo_)
      throw cms::Exception("InvalidConfiguration") << "Nonthing to do in TowerStatusTask";
  }

  TowerStatusTask::~TowerStatusTask()
  {
  }

  void
  TowerStatusTask::bookMEs()
  {
  }

  void
  TowerStatusTask::beginRun(const edm::Run &, const edm::EventSetup &)
  {
    if(doDAQInfo_){
      MEs_[kDAQSummary]->book();
      MEs_[kDAQSummaryMap]->book();
      MEs_[kDAQContents]->book();

      MEs_[kDAQSummaryMap]->resetAll(-1.);
    }
    if(doDCSInfo_){
      MEs_[kDCSSummary]->book();
      MEs_[kDCSSummaryMap]->book();
      MEs_[kDCSContents]->book();

      MEs_[kDCSSummaryMap]->resetAll(-1.);
    }

    daqLumiStatus_.clear();
    daqRunStatus_.clear();
    dcsLumiStatus_.clear();
    dcsRunStatus_.clear();

    for(unsigned id(0); id < EcalTrigTowerDetId::kEBTotalTowers; id++){
      EcalTrigTowerDetId ttid(EcalTrigTowerDetId::detIdFromDenseIndex(id));
      if(doDAQInfo_){
	daqLumiStatus_[ttid.rawId()] = true;
	daqRunStatus_[ttid.rawId()] = true;
      }
      if(doDCSInfo_){
	dcsLumiStatus_[ttid.rawId()] = true;
	dcsRunStatus_[ttid.rawId()] = true;
      }
    }

    for(unsigned id(0); id < EcalScDetId::kSizeForDenseIndexing; id++){
      EcalScDetId scid(EcalScDetId::unhashIndex(id));
      if(doDAQInfo_){
	daqLumiStatus_[scid.rawId()] = true;
	daqRunStatus_[scid.rawId()] = true;
      }
      if(doDCSInfo_){
	dcsLumiStatus_[scid.rawId()] = true;
	dcsRunStatus_[scid.rawId()] = true;
      }
    }
  }

  void
  TowerStatusTask::endRun(const edm::Run &, const edm::EventSetup &)
  {
    if(doDAQInfo_) runOnTowerStatus(daqRunStatus_, 0);
    if(doDCSInfo_) runOnTowerStatus(dcsRunStatus_, 1);
  }

  void
  TowerStatusTask::beginLuminosityBlock(const edm::LuminosityBlock &, const edm::EventSetup &_es)
  {
    edm::ESHandle<EcalDAQTowerStatus> daqHndl;
    if(doDAQInfo_){
      _es.get<EcalDAQTowerStatusRcd>().get(daqHndl);
      if (!daqHndl.isValid()){
	edm::LogWarning("EventSetup") << "EcalDAQTowerStatus record not valid";
	return;
      }
    }

    edm::ESHandle<EcalDCSTowerStatus> dcsHndl;
    if(doDCSInfo_){
      _es.get<EcalDCSTowerStatusRcd>().get(dcsHndl);
      if (!dcsHndl.isValid()){
	edm::LogWarning("EventSetup") << "EcalDCSTowerStatus record not valid";
	return;
      }
    }

    for(unsigned id(0); id < EcalTrigTowerDetId::kEBTotalTowers; id++){
      EcalTrigTowerDetId ttid(EcalTrigTowerDetId::detIdFromDenseIndex(id));
      if(doDAQInfo_){
	uint16_t status(daqHndl->barrel(id).getStatusCode());
	if(status != 0){
	  daqLumiStatus_[ttid.rawId()] = false;
	  daqRunStatus_[ttid.rawId()] = false;
	}
      }
      if(doDCSInfo_){
	uint16_t status(dcsHndl->barrel(id).getStatusCode());
	if(status != 0){
	  dcsLumiStatus_[ttid.rawId()] = false;
	  dcsRunStatus_[ttid.rawId()] = false;
	}
      }
    }

    for(unsigned id(0); id < EcalScDetId::kSizeForDenseIndexing; id++){
      EcalScDetId scid(EcalScDetId::unhashIndex(id));
      if(doDAQInfo_){
	uint16_t status(daqHndl->endcap(id).getStatusCode());
	if(status != 0){
	  daqLumiStatus_[scid.rawId()] = false;
	  daqRunStatus_[scid.rawId()] = false;
	}
      }
      if(doDCSInfo_){
	uint16_t status(dcsHndl->endcap(id).getStatusCode());
	if(status != 0){
	  dcsLumiStatus_[scid.rawId()] = false;
	  dcsRunStatus_[scid.rawId()] = false;
	}
      }
    }
  }

  void
  TowerStatusTask::endLuminosityBlock(const edm::LuminosityBlock &, const edm::EventSetup &)
  {
    if(doDAQInfo_) runOnTowerStatus(daqLumiStatus_, 0);
    if(doDCSInfo_) runOnTowerStatus(dcsLumiStatus_, 1);
  }

  void
  TowerStatusTask::runOnTowerStatus(const std::map<uint32_t, bool>& _status, int _type)
  {
    if(!initialized_) return;

    std::vector<int> activeChannels(54, 0);

    unsigned summary, summaryMap, contents;
    if(_type == 0){
      summary = kDAQSummary;
      summaryMap = kDAQSummaryMap;
      contents = kDAQContents;
    }
    else{
      summary = kDCSSummary;
      summaryMap = kDCSSummaryMap;
      contents = kDCSContents;
    }

    MEs_[summaryMap]->reset();

    for(std::map<uint32_t, bool>::const_iterator stItr(_status.begin()); stItr != _status.end(); ++stItr){
      DetId id(stItr->first);
      bool status(stItr->second);

      std::cout.flush();
      MEs_[summaryMap]->setBinContent(id, status ? 1. : 0.);

      if(status){
	if(id.subdetId() == EcalTriggerTower)
	  activeChannels[dccId(id) - 1] += 25;
	else{
	  int dccid(dccId(id));
	  int towerid(towerId(id));
	  activeChannels[dccId(id) - 1] += getElectronicsMap()->dccTowerConstituents(dccid, towerid).size();
	}
      }
    }

    int totalActive(0);
    for(unsigned iDCC(0); iDCC < 54; iDCC++){
      float fraction(float(activeChannels[iDCC]) / float(getElectronicsMap()->dccConstituents(iDCC + 1).size()));
      MEs_[contents]->fill(iDCC + 1, fraction);
      totalActive += activeChannels[iDCC];
    }

    MEs_[summary]->fill(float(totalActive) / float(EBDetId::kSizeForDenseIndexing + EEDetId::kSizeForDenseIndexing));
  }

  /*static*/
  void
  TowerStatusTask::setMEData(std::vector<MEData>& _data)
  {
    _data[kDAQSummary] = MEData("DAQSummary", BinService::kEcal, BinService::kReport, MonitorElement::DQM_KIND_REAL);
    _data[kDAQSummaryMap] = MEData("DAQSummaryMap", BinService::kEcal, BinService::kSuperCrystal, MonitorElement::DQM_KIND_TH2F);
    _data[kDAQContents] = MEData("DAQContents", BinService::kSM, BinService::kReport, MonitorElement::DQM_KIND_REAL);
    _data[kDCSSummary] = MEData("DCSSummary", BinService::kEcal, BinService::kReport, MonitorElement::DQM_KIND_REAL);
    _data[kDCSSummaryMap] = MEData("DCSSummaryMap", BinService::kEcal, BinService::kSuperCrystal, MonitorElement::DQM_KIND_TH2F);
    _data[kDCSContents] = MEData("DCSContents", BinService::kSM, BinService::kReport, MonitorElement::DQM_KIND_REAL);
  }

  DEFINE_ECALDQM_WORKER(TowerStatusTask);
}
 

