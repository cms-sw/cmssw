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

  TowerStatusTask::TowerStatusTask(edm::ParameterSet const& _workerParams, edm::ParameterSet const& _commonParams) :
    DQWorkerTask(_workerParams, _commonParams, "TowerStatusTask"),
    doDAQInfo_(_workerParams.getUntrackedParameter<bool>("doDAQInfo")),
    doDCSInfo_(_workerParams.getUntrackedParameter<bool>("doDCSInfo"))
  {
    collectionMask_ = 
      (0x1 << kRun) |
      (0x1 << kLumiSection);

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
  TowerStatusTask::setMEOrdering(std::map<std::string, unsigned>& _nameToIndex)
  {
    _nameToIndex["DAQSummary"] = kDAQSummary;
    _nameToIndex["DAQSummaryMap"] = kDAQSummaryMap;
    _nameToIndex["DAQContents"] = kDAQContents;
    _nameToIndex["DCSSummary"] = kDCSSummary;
    _nameToIndex["DCSSummaryMap"] = kDCSSummaryMap;
    _nameToIndex["DCSContents"] = kDCSContents;
  }

  DEFINE_ECALDQM_WORKER(TowerStatusTask);
}
 

