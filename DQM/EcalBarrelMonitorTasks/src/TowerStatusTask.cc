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
    collectionMask_[kRun] = true;
    collectionMask_[kLumiSection] = true;

    if(!doDAQInfo_ && !doDCSInfo_)
      throw cms::Exception("InvalidConfiguration") << "Nonthing to do in TowerStatusTask";
  }

  void
  TowerStatusTask::beginRun(const edm::Run &, const edm::EventSetup &)
  {
    if(doDAQInfo_){
      MEs_["DAQSummary"]->book();
      MEs_["DAQSummaryMap"]->book();
      MEs_["DAQContents"]->book();

      MEs_["DAQSummary"]->reset(-1.);
      MEs_["DAQSummaryMap"]->resetAll(-1.);
      MEs_["DAQContents"]->reset(-1.);
    }
    if(doDCSInfo_){
      MEs_["DCSSummary"]->book();
      MEs_["DCSSummaryMap"]->book();
      MEs_["DCSContents"]->book();

      MEs_["DCSSummary"]->reset(-1.);
      MEs_["DCSSummaryMap"]->resetAll(-1.);
      MEs_["DCSContents"]->reset(-1.);
    }

    initialized_ = true;
  }

  void
  TowerStatusTask::beginLuminosityBlock(const edm::LuminosityBlock &, const edm::EventSetup &_es)
  {
    if(doDAQInfo_){
      std::vector<float> status(54, 1.);

      edm::ESHandle<EcalDAQTowerStatus> daqHndl;
      _es.get<EcalDAQTowerStatusRcd>().get(daqHndl);
      if(daqHndl.isValid()){
        for(unsigned id(0); id < EcalTrigTowerDetId::kEBTotalTowers; id++){
          if(daqHndl->barrel(id).getStatusCode() != 0){
            EcalTrigTowerDetId ttid(EcalTrigTowerDetId::detIdFromDenseIndex(id));
            status[dccId(ttid) - 1] -= 25. / 1700.;
          }
        }
        for(unsigned id(0); id < EcalScDetId::kSizeForDenseIndexing; id++){
          if(daqHndl->endcap(id).getStatusCode() != 0){
            EcalScDetId scid(EcalScDetId::unhashIndex(id));
            unsigned dccid(dccId(scid));
            status[dccid - 1] -= double(scConstituents(scid).size()) / nCrystals(dccid);
          }
        }

        runOnTowerStatus(status, DAQInfo);
      }
      else
	edm::LogWarning("EventSetup") << "EcalDAQTowerStatus record not valid";
    }

    if(doDCSInfo_){
      std::vector<float> status(54, 1.);

      edm::ESHandle<EcalDCSTowerStatus> dcsHndl;
      _es.get<EcalDCSTowerStatusRcd>().get(dcsHndl);
      if(dcsHndl.isValid()){
        for(unsigned id(0); id < EcalTrigTowerDetId::kEBTotalTowers; id++){
          if(dcsHndl->barrel(id).getStatusCode() != 0){
            EcalTrigTowerDetId ttid(EcalTrigTowerDetId::detIdFromDenseIndex(id));
            status[dccId(ttid) - 1] -= 25. / 1700.;
          }
        }
        for(unsigned id(0); id < EcalScDetId::kSizeForDenseIndexing; id++){
          if(dcsHndl->endcap(id).getStatusCode() != 0){
            EcalScDetId scid(EcalScDetId::unhashIndex(id));
            unsigned dccid(dccId(scid));
            status[dccid - 1] -= double(scConstituents(scid).size()) / nCrystals(dccid);
          }
        }
        
        runOnTowerStatus(status, DCSInfo);
      }
      else
	edm::LogWarning("EventSetup") << "EcalDCSTowerStatus record not valid";
    }
  }

  void
  TowerStatusTask::runOnTowerStatus(std::vector<float> const& _status, InfoType _type)
  {
    if(!initialized_) return;

    MESet* meSummary(0);
    MESet* meSummaryMap(0);
    MESet* meContents(0);
    if(_type == DAQInfo){
      meSummary = MEs_["DAQSummary"];
      meSummaryMap = MEs_["DAQSummaryMap"];
      meContents = MEs_["DAQContents"];
    }
    else{
      meSummary = MEs_["DCSSummary"];
      meSummaryMap = MEs_["DCSSummaryMap"];
      meContents = MEs_["DCSContents"];
    }

    meSummaryMap->reset();

    float totalFraction(0.);
    for(unsigned iDCC(0); iDCC < 54; iDCC++){
      meSummaryMap->setBinContent(iDCC + 1, _status[iDCC]);
      meContents->fill(iDCC + 1, _status[iDCC]);
      totalFraction += _status[iDCC] / nCrystals(iDCC + 1);
    }

    meSummary->fill(totalFraction);
  }

  DEFINE_ECALDQM_WORKER(TowerStatusTask);
}
 

