#include "../interface/EcalDQMonitorTask.h"

#include "../interface/DQWorkerTask.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/EcalRawData/interface/EcalRawDataCollections.h"
#include "DataFormats/EcalDetId/interface/EcalDetIdCollections.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"

#include "TStopwatch.h"

template <typename CollectionClass>
void
EcalDQMonitorTask::runOnCollection(edm::Event const& _evt, ecaldqm::Collections _col, std::set<ecaldqm::DQWorker*> const& _enabledTasks)
{
  edm::Handle<CollectionClass> hndl;
  if(!_evt.getByToken(collectionTokens_[_col], hndl)){
    if(!allowMissingCollections_)
      throw cms::Exception("ObjectNotFound") << moduleName_ << "::runOnCollection: " << ecaldqm::collectionName[_col] << " does not exist";
    edm::LogWarning("EcalDQM") << moduleName_ << "::runOnCollection: " << ecaldqm::collectionName[_col] << " does not exist";
    return;
  }

  CollectionClass const* collection(hndl.product());

  TStopwatch sw;
  sw.Reset();

  executeOnWorkers_([collection, _col, &_enabledTasks, &sw, this](ecaldqm::DQWorker* worker){
                      if(_enabledTasks.find(worker) != _enabledTasks.end()){
                        if(this->evaluateTime_) sw.Start();
                        static_cast<ecaldqm::DQWorkerTask*>(worker)->analyze(collection, _col);
                        if(this->evaluateTime_) this->taskTimes_[worker] += sw.RealTime();
                      }
                    }, "analyze");

  edm::LogInfo("EcalDQM") << moduleName_ << "::runOn" << ecaldqm::collectionName[_col] << " returning";
}

void
EcalDQMonitorTask::formSchedule(std::vector<ecaldqm::Collections> const& _preSchedule, edm::ParameterSet const& _tagPSet)
{
  for(std::vector<ecaldqm::Collections>::const_iterator colItr(_preSchedule.begin()); colItr != _preSchedule.end(); ++colItr){
    std::pair<Processor, ecaldqm::Collections> sch;

    edm::InputTag tag(_tagPSet.getUntrackedParameter<edm::InputTag>(ecaldqm::collectionName[*colItr]));

    switch(*colItr){
    case ecaldqm::kSource:
      collectionTokens_[*colItr] = edm::EDGetToken(consumes<FEDRawDataCollection>(tag));
      sch.first = &EcalDQMonitorTask::runOnCollection<FEDRawDataCollection>;
      break;
    case ecaldqm::kEcalRawData:
      collectionTokens_[*colItr] = edm::EDGetToken(consumes<EcalRawDataCollection>(tag));
      sch.first = &EcalDQMonitorTask::runOnCollection<EcalRawDataCollection>;
      break;
    case ecaldqm::kEBGainErrors:
    case ecaldqm::kEBChIdErrors:
    case ecaldqm::kEBGainSwitchErrors:
      collectionTokens_[*colItr] = edm::EDGetToken(consumes<EBDetIdCollection>(tag));
      sch.first = &EcalDQMonitorTask::runOnCollection<EBDetIdCollection>;
      break;
    case ecaldqm::kEEGainErrors:
    case ecaldqm::kEEChIdErrors:
    case ecaldqm::kEEGainSwitchErrors:
      collectionTokens_[*colItr] = edm::EDGetToken(consumes<EEDetIdCollection>(tag));
      sch.first = &EcalDQMonitorTask::runOnCollection<EEDetIdCollection>;
      break;
    case ecaldqm::kTowerIdErrors:
    case ecaldqm::kBlockSizeErrors:
    case ecaldqm::kMEMTowerIdErrors:
    case ecaldqm::kMEMBlockSizeErrors:
    case ecaldqm::kMEMChIdErrors:
    case ecaldqm::kMEMGainErrors:
      collectionTokens_[*colItr] = edm::EDGetToken(consumes<EcalElectronicsIdCollection>(tag));
      sch.first = &EcalDQMonitorTask::runOnCollection<EcalElectronicsIdCollection>;
      break;
    case ecaldqm::kEBSrFlag:
      collectionTokens_[*colItr] = edm::EDGetToken(consumes<EBSrFlagCollection>(tag));
      sch.first = &EcalDQMonitorTask::runOnCollection<EBSrFlagCollection>;
      break;
    case ecaldqm::kEESrFlag:
      collectionTokens_[*colItr] = edm::EDGetToken(consumes<EESrFlagCollection>(tag));
      sch.first = &EcalDQMonitorTask::runOnCollection<EESrFlagCollection>;
      break;
    case ecaldqm::kEBDigi:
      collectionTokens_[*colItr] = edm::EDGetToken(consumes<EBDigiCollection>(tag));
      sch.first = &EcalDQMonitorTask::runOnCollection<EBDigiCollection>;
      break;
    case ecaldqm::kEEDigi:
      collectionTokens_[*colItr] = edm::EDGetToken(consumes<EEDigiCollection>(tag));
      sch.first = &EcalDQMonitorTask::runOnCollection<EEDigiCollection>;
      break;
    case ecaldqm::kPnDiodeDigi:
      collectionTokens_[*colItr] = edm::EDGetToken(consumes<EcalPnDiodeDigiCollection>(tag));
      sch.first = &EcalDQMonitorTask::runOnCollection<EcalPnDiodeDigiCollection>;
      break;
    case ecaldqm::kTrigPrimDigi:
    case ecaldqm::kTrigPrimEmulDigi:
      collectionTokens_[*colItr] = edm::EDGetToken(consumes<EcalTrigPrimDigiCollection>(tag));
      sch.first = &EcalDQMonitorTask::runOnCollection<EcalTrigPrimDigiCollection>;
      break;
    case ecaldqm::kEBUncalibRecHit:
    case ecaldqm::kEEUncalibRecHit:
    case ecaldqm::kEBLaserLedUncalibRecHit:
    case ecaldqm::kEELaserLedUncalibRecHit:
    case ecaldqm::kEBTestPulseUncalibRecHit:
    case ecaldqm::kEETestPulseUncalibRecHit:
      collectionTokens_[*colItr] = edm::EDGetToken(consumes<EcalUncalibratedRecHitCollection>(tag));
      sch.first = &EcalDQMonitorTask::runOnCollection<EcalUncalibratedRecHitCollection>;
      break;
    case ecaldqm::kEBRecHit:
    case ecaldqm::kEBReducedRecHit:
    case ecaldqm::kEERecHit:
    case ecaldqm::kEEReducedRecHit:
      collectionTokens_[*colItr] = edm::EDGetToken(consumes<EcalRecHitCollection>(tag));
      sch.first = &EcalDQMonitorTask::runOnCollection<EcalRecHitCollection>;
      break;
    case ecaldqm::kEBBasicCluster:
    case ecaldqm::kEEBasicCluster:
      collectionTokens_[*colItr] = edm::EDGetToken(consumes<edm::View<reco::CaloCluster> >(tag));
      sch.first = &EcalDQMonitorTask::runOnCollection<edm::View<reco::CaloCluster> >;
      break;
    case ecaldqm::kEBSuperCluster:
    case ecaldqm::kEESuperCluster:
      collectionTokens_[*colItr] = edm::EDGetToken(consumes<reco::SuperClusterCollection>(tag));
      sch.first = &EcalDQMonitorTask::runOnCollection<reco::SuperClusterCollection>;
      break;
    default:
      throw cms::Exception("InvalidConfiguration") << "Undefined collection " << *colItr;
    }

    sch.second = *colItr;

    schedule_.push_back(sch);
  }
}
