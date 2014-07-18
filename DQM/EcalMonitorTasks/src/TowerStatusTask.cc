#include "../interface/TowerStatusTask.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

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

  TowerStatusTask::TowerStatusTask() :
    DQWorkerTask(),
    doDAQInfo_(false),
    doDCSInfo_(false)
  {
  }

  void
  TowerStatusTask::setParams(edm::ParameterSet const& _params)
  {
    doDAQInfo_ = _params.getUntrackedParameter<bool>("doDAQInfo");
    doDCSInfo_ = _params.getUntrackedParameter<bool>("doDCSInfo");

    if(doDAQInfo_ && doDCSInfo_) return;
    if(doDAQInfo_){
      MEs_.erase(std::string("DCSSummary"));
      MEs_.erase(std::string("DCSSummaryMap"));
      MEs_.erase(std::string("DCSContents"));
    }
    else if(doDCSInfo_){
      MEs_.erase(std::string("DAQSummary"));
      MEs_.erase(std::string("DAQSummaryMap"));
      MEs_.erase(std::string("DAQContents"));
    }
    else
      throw cms::Exception("InvalidConfiguration") << "Nothing to do in TowerStatusTask";
  }

  void
  TowerStatusTask::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const& _es)
  {
    if(doDAQInfo_){
      float status[nDCC];
      std::fill_n(status, nDCC, 1.);

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
      float status[nDCC];
      std::fill_n(status, nDCC, 1.);

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
  TowerStatusTask::runOnTowerStatus(float const* _status, InfoType _type)
  {
    MESet* meSummary(0);
    MESet* meSummaryMap(0);
    MESet* meContents(0);
    if(_type == DAQInfo){
      meSummary = &MEs_.at("DAQSummary");
      meSummaryMap = &MEs_.at("DAQSummaryMap");
      meContents = &MEs_.at("DAQContents");
    }
    else{
      meSummary = &MEs_.at("DCSSummary");
      meSummaryMap = &MEs_.at("DCSSummaryMap");
      meContents = &MEs_.at("DCSContents");
    }

    meSummary->reset(-1.);
    meSummaryMap->resetAll(-1.);
    meSummaryMap->reset();
    meContents->reset(-1.);

    float totalFraction(0.);
    for(int iDCC(0); iDCC < nDCC; iDCC++){
      meSummaryMap->setBinContent(iDCC + 1, _status[iDCC]);
      meContents->fill(iDCC + 1, _status[iDCC]);
      totalFraction += _status[iDCC] / nCrystals(iDCC + 1);
    }

    meSummary->fill(totalFraction);
  }

  DEFINE_ECALDQM_WORKER(TowerStatusTask);
}
 

