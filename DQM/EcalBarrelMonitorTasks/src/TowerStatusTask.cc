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

  TowerStatusTask::TowerStatusTask(const edm::ParameterSet& _params) :
    DQWorkerTask(_params, "TowerStatusTask"),
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
  }

  void
  TowerStatusTask::beginLuminosityBlock(const edm::LuminosityBlock &, const edm::EventSetup &_es)
  {
    if(doDAQInfo_){
      std::vector<float> status(54, 1.);

      edm::ESHandle<EcalDAQTowerStatus> daqHndl;
      _es.get<EcalDAQTowerStatusRcd>().get(daqHndl);
      if (!daqHndl.isValid()){
	edm::LogWarning("EventSetup") << "EcalDAQTowerStatus record not valid";
	return;
      }

      for(unsigned id(0); id < EcalTrigTowerDetId::kEBTotalTowers; id++){
	if(daqHndl->barrel(id).getStatusCode() != 0){
          EcalTrigTowerDetId ttid(EcalTrigTowerDetId::detIdFromDenseIndex(id));
	  status[dccId(ttid) - 1] -= 25. / 1700.;
	}
      }
      for(unsigned id(0); id < EcalScDetId::kSizeForDenseIndexing; id++){
	if(daqHndl->endcap(id).getStatusCode() == 0){
          EcalScDetId scid(EcalScDetId::unhashIndex(id));
          std::pair<int, int> dccsc(getElectronicsMap()->getDCCandSC(scid));
          float nC(getElectronicsMap()->dccTowerConstituents(dccsc.first, dccsc.second).size());
          unsigned dccid(dccId(scid));
	  status[dccid - 1] -= nC / nCrystals(dccid);
	}
      }

      runOnTowerStatus(status, DAQInfo);
    }

    if(doDCSInfo_){
      std::vector<float> status(54, 1.);

      edm::ESHandle<EcalDCSTowerStatus> dcsHndl;
      _es.get<EcalDCSTowerStatusRcd>().get(dcsHndl);
      if (!dcsHndl.isValid()){
	edm::LogWarning("EventSetup") << "EcalDCSTowerStatus record not valid";
	return;
      }

      for(unsigned id(0); id < EcalTrigTowerDetId::kEBTotalTowers; id++){
	if(dcsHndl->barrel(id).getStatusCode() != 0){
          EcalTrigTowerDetId ttid(EcalTrigTowerDetId::detIdFromDenseIndex(id));
	  status[dccId(ttid) - 1] -= 25. / 1700.;
	}
      }
      for(unsigned id(0); id < EcalScDetId::kSizeForDenseIndexing; id++){
	if(dcsHndl->endcap(id).getStatusCode() == 0){
          EcalScDetId scid(EcalScDetId::unhashIndex(id));
          std::pair<int, int> dccsc(getElectronicsMap()->getDCCandSC(scid));
          float nC(getElectronicsMap()->dccTowerConstituents(dccsc.first, dccsc.second).size());
          unsigned dccid(dccId(scid));
	  status[dccid - 1] -= nC / nCrystals(dccid);
	}
      }

      runOnTowerStatus(status, DCSInfo);
    }
  }

  void
  TowerStatusTask::runOnTowerStatus(std::vector<float> const& _status, InfoType _type)
  {
    if(!initialized_) return;

    unsigned summary, summaryMap, contents;
    if(_type == DAQInfo){
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

    float totalFraction(0.);
    for(unsigned iDCC(0); iDCC < 54; iDCC++){
      MEs_[summaryMap]->setBinContent(iDCC + 1, _status[iDCC]);
      MEs_[contents]->fill(iDCC + 1, _status[iDCC]);
      totalFraction += _status[iDCC] / nCrystals(iDCC + 1);
    }

    MEs_[summary]->fill(totalFraction);
  }

  /*static*/
  void
  TowerStatusTask::setMEData(std::vector<MEData>& _data)
  {
    _data[kDAQSummary] = MEData("DAQSummary", BinService::kEcal, BinService::kReport, MonitorElement::DQM_KIND_REAL);
    _data[kDAQSummaryMap] = MEData("DAQSummaryMap", BinService::kEcal, BinService::kDCC, MonitorElement::DQM_KIND_TH2F);
    _data[kDAQContents] = MEData("DAQContents", BinService::kSM, BinService::kReport, MonitorElement::DQM_KIND_REAL);
    _data[kDCSSummary] = MEData("DCSSummary", BinService::kEcal, BinService::kReport, MonitorElement::DQM_KIND_REAL);
    _data[kDCSSummaryMap] = MEData("DCSSummaryMap", BinService::kEcal, BinService::kDCC, MonitorElement::DQM_KIND_TH2F);
    _data[kDCSContents] = MEData("DCSContents", BinService::kSM, BinService::kReport, MonitorElement::DQM_KIND_REAL);
  }

  DEFINE_ECALDQM_WORKER(TowerStatusTask);
}
 

