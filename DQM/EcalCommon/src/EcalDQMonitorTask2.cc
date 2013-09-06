#include "DQM/EcalCommon/interface/EcalDQMonitorTask.h"

#include "DQM/EcalCommon/interface/DQWorkerTask.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/DetId/interface/DetIdCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/EcalRawData/interface/EcalRawDataCollections.h"
#include "DataFormats/EcalDetId/interface/EcalDetIdCollections.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"

#include "TStopwatch.h"

void
EcalDQMonitorTask::registerCollection(ecaldqm::Collections _collection, edm::InputTag const& _inputTag)
{
  switch(_collection){
    case ecaldqm::kSource:
      collectionTokens_[_collection] = edm::EDGetToken(consumes<FEDRawDataCollection>(_inputTag)); break;
    case ecaldqm::kEcalRawData:
      collectionTokens_[_collection] = edm::EDGetToken(consumes<EcalRawDataCollection>(_inputTag)); break;
    case ecaldqm::kGainErrors:
      collectionTokens_[_collection] = edm::EDGetToken(consumes<DetIdCollection>(_inputTag)); break;
    case ecaldqm::kChIdErrors:
      collectionTokens_[_collection] = edm::EDGetToken(consumes<DetIdCollection>(_inputTag)); break;
    case ecaldqm::kGainSwitchErrors:
      collectionTokens_[_collection] = edm::EDGetToken(consumes<DetIdCollection>(_inputTag)); break;
    case ecaldqm::kTowerIdErrors:
      collectionTokens_[_collection] = edm::EDGetToken(consumes<EcalElectronicsIdCollection>(_inputTag)); break;
    case ecaldqm::kBlockSizeErrors:
      collectionTokens_[_collection] = edm::EDGetToken(consumes<EcalElectronicsIdCollection>(_inputTag)); break;
    case ecaldqm::kMEMTowerIdErrors:
      collectionTokens_[_collection] = edm::EDGetToken(consumes<EcalElectronicsIdCollection>(_inputTag)); break;
    case ecaldqm::kMEMBlockSizeErrors:
      collectionTokens_[_collection] = edm::EDGetToken(consumes<EcalElectronicsIdCollection>(_inputTag)); break;
    case ecaldqm::kMEMChIdErrors:
      collectionTokens_[_collection] = edm::EDGetToken(consumes<EcalElectronicsIdCollection>(_inputTag)); break;
    case ecaldqm::kMEMGainErrors:
      collectionTokens_[_collection] = edm::EDGetToken(consumes<EcalElectronicsIdCollection>(_inputTag)); break;
    case ecaldqm::kEBSrFlag:
      collectionTokens_[_collection] = edm::EDGetToken(consumes<EBSrFlagCollection>(_inputTag)); break;
    case ecaldqm::kEESrFlag:
      collectionTokens_[_collection] = edm::EDGetToken(consumes<EESrFlagCollection>(_inputTag)); break;
    case ecaldqm::kEBDigi:
      collectionTokens_[_collection] = edm::EDGetToken(consumes<EBDigiCollection>(_inputTag)); break;
    case ecaldqm::kEEDigi:
      collectionTokens_[_collection] = edm::EDGetToken(consumes<EEDigiCollection>(_inputTag)); break;
    case ecaldqm::kPnDiodeDigi:
      collectionTokens_[_collection] = edm::EDGetToken(consumes<EcalPnDiodeDigiCollection>(_inputTag)); break;
    case ecaldqm::kTrigPrimDigi:
      collectionTokens_[_collection] = edm::EDGetToken(consumes<EcalTrigPrimDigiCollection>(_inputTag)); break;
    case ecaldqm::kTrigPrimEmulDigi:
      collectionTokens_[_collection] = edm::EDGetToken(consumes<EcalTrigPrimDigiCollection>(_inputTag)); break;
    case ecaldqm::kEBUncalibRecHit:
      collectionTokens_[_collection] = edm::EDGetToken(consumes<EcalUncalibratedRecHitCollection>(_inputTag)); break;
    case ecaldqm::kEEUncalibRecHit:
      collectionTokens_[_collection] = edm::EDGetToken(consumes<EcalUncalibratedRecHitCollection>(_inputTag)); break;
    case ecaldqm::kEBRecHit:
      collectionTokens_[_collection] = edm::EDGetToken(consumes<EcalRecHitCollection>(_inputTag)); break;
    case ecaldqm::kEERecHit:
      collectionTokens_[_collection] = edm::EDGetToken(consumes<EcalRecHitCollection>(_inputTag)); break;
    case ecaldqm::kEBBasicCluster:
      collectionTokens_[_collection] = edm::EDGetToken(consumes<reco::BasicClusterCollection>(_inputTag)); break;
    case ecaldqm::kEEBasicCluster:
      collectionTokens_[_collection] = edm::EDGetToken(consumes<reco::BasicClusterCollection>(_inputTag)); break;
    case ecaldqm::kEBSuperCluster:
      collectionTokens_[_collection] = edm::EDGetToken(consumes<reco::SuperClusterCollection>(_inputTag)); break;
    case ecaldqm::kEESuperCluster:
      collectionTokens_[_collection] = edm::EDGetToken(consumes<reco::SuperClusterCollection>(_inputTag)); break;
    default:
      throw cms::Exception("InvalidConfiguration") << "Undefined collection " << _collection;
  }
}

template <class C>
void
EcalDQMonitorTask::runOnCollection(edm::Event const& _evt, ecaldqm::Collections _collection)
{
  edm::Handle<C> hndl;
  if(_evt.getByToken(collectionTokens_[_collection], hndl)){

    TStopwatch* sw(0);
    if(evaluateTime_){
      sw = new TStopwatch;
      sw->Stop();
    }

    ecaldqm::DQWorkerTask* task(0);

    for(std::vector<ecaldqm::DQWorkerTask *>::iterator wItr(taskLists_[_collection].begin()); wItr != taskLists_[_collection].end(); ++wItr){
      task = *wItr;
      if(evaluateTime_) sw->Start();
      if(enabled_[task]) task->analyze(hndl.product(), _collection);
      if(evaluateTime_){
	sw->Stop();
	taskTimes_[task] += sw->RealTime();
      }
    }

    delete sw;
  }
  else if(!allowMissingCollections_)
    throw cms::Exception("ObjectNotFound") << ecaldqm::collectionName[_collection] << " does not exist";
}

template <>
void
EcalDQMonitorTask::runOnCollection<DetIdCollection>(edm::Event const& _evt, ecaldqm::Collections _collection)
{
  edm::Handle<EBDetIdCollection> ebHndl;
  edm::Handle<EEDetIdCollection> eeHndl;
  if(_evt.getByToken(collectionTokens_[_collection], ebHndl) && _evt.getByToken(collectionTokens_[_collection], eeHndl)){
    unsigned nEB(ebHndl->size());
    unsigned nEE(eeHndl->size());

    if(nEB == 0 && nEE == 0) return;

    DetIdCollection ids;
    for(unsigned iId(0); iId < nEB; iId++) ids.push_back(DetId(ebHndl->at(iId)));
    for(unsigned iId(0); iId < nEE; iId++) ids.push_back(DetId(eeHndl->at(iId)));

    TStopwatch* sw(0);
    if(evaluateTime_){
      sw = new TStopwatch;
      sw->Stop();
    }

    ecaldqm::DQWorkerTask* task(0);

    for(std::vector<ecaldqm::DQWorkerTask *>::iterator wItr(taskLists_[_collection].begin()); wItr != taskLists_[_collection].end(); ++wItr){
      task = *wItr;
      if(evaluateTime_) sw->Start();
      if(enabled_[task]) task->analyze(const_cast<const DetIdCollection*>(&ids), _collection);
      if(evaluateTime_){
	sw->Stop();
	taskTimes_[task] += sw->RealTime();
      }
    }

    delete sw;
  }
  else if(!allowMissingCollections_)
    throw cms::Exception("ObjectNotFound") << ecaldqm::collectionName[_collection] << " does not exist";
}

void
EcalDQMonitorTask::formSchedule_(std::vector<ecaldqm::Collections> const& _usedCollections, std::multimap<ecaldqm::Collections, ecaldqm::Collections> const& _dependencies)
{
  typedef std::multimap<ecaldqm::Collections, ecaldqm::Collections>::const_iterator mmiter;

  std::vector<ecaldqm::Collections> preSchedule;
  std::vector<ecaldqm::Collections>::iterator insertPoint, findPoint;

  for(std::vector<ecaldqm::Collections>::const_iterator colItr(_usedCollections.begin()); colItr != _usedCollections.end(); ++colItr){

    bool inserted(true);
    if((insertPoint = find(preSchedule.begin(), preSchedule.end(), *colItr)) == preSchedule.end()) inserted = false;

    std::pair<mmiter, mmiter> range(_dependencies.equal_range(*colItr));

    for(mmiter depItr(range.first); depItr != range.second; ++depItr){

      if(depItr->second == depItr->first)
	throw cms::Exception("Fatal") << "Collection " << depItr->second << " depends on itself";
      if(find(_usedCollections.begin(), _usedCollections.end(), depItr->second) == _usedCollections.end())
	throw cms::Exception("Fatal") << "Collection " << depItr->first << " depends on Collection " << depItr->second;

      if((findPoint = find(preSchedule.begin(), preSchedule.end(), depItr->second)) == preSchedule.end())
	preSchedule.insert(insertPoint, depItr->second);
      else if(findPoint > insertPoint)
	throw cms::Exception("InvalidConfiguration") << "Circular dependencies in Collections";

    }

    if(!inserted) preSchedule.push_back(*colItr);

  }

  for(std::vector<ecaldqm::Collections>::const_iterator colItr(preSchedule.begin()); colItr != preSchedule.end(); ++colItr){
    std::pair<Processor, ecaldqm::Collections> sch;

    switch(*colItr){
    case ecaldqm::kSource:
      sch.first = &EcalDQMonitorTask::runOnCollection<FEDRawDataCollection>; break;
    case ecaldqm::kEcalRawData:
      sch.first = &EcalDQMonitorTask::runOnCollection<EcalRawDataCollection>; break;
    case ecaldqm::kGainErrors:
      sch.first = &EcalDQMonitorTask::runOnCollection<DetIdCollection>; break;
    case ecaldqm::kChIdErrors:
      sch.first = &EcalDQMonitorTask::runOnCollection<DetIdCollection>; break;
    case ecaldqm::kGainSwitchErrors:
      sch.first = &EcalDQMonitorTask::runOnCollection<DetIdCollection>; break;
    case ecaldqm::kTowerIdErrors:
      sch.first = &EcalDQMonitorTask::runOnCollection<EcalElectronicsIdCollection>; break;
    case ecaldqm::kBlockSizeErrors:
      sch.first = &EcalDQMonitorTask::runOnCollection<EcalElectronicsIdCollection>; break;
    case ecaldqm::kMEMTowerIdErrors:
      sch.first = &EcalDQMonitorTask::runOnCollection<EcalElectronicsIdCollection>; break;
    case ecaldqm::kMEMBlockSizeErrors:
      sch.first = &EcalDQMonitorTask::runOnCollection<EcalElectronicsIdCollection>; break;
    case ecaldqm::kMEMChIdErrors:
      sch.first = &EcalDQMonitorTask::runOnCollection<EcalElectronicsIdCollection>; break;
    case ecaldqm::kMEMGainErrors:
      sch.first = &EcalDQMonitorTask::runOnCollection<EcalElectronicsIdCollection>; break;
    case ecaldqm::kEBSrFlag:
      sch.first = &EcalDQMonitorTask::runOnCollection<EBSrFlagCollection>; break;
    case ecaldqm::kEESrFlag:
      sch.first = &EcalDQMonitorTask::runOnCollection<EESrFlagCollection>; break;
    case ecaldqm::kEBDigi:
      sch.first = &EcalDQMonitorTask::runOnCollection<EBDigiCollection>; break;
    case ecaldqm::kEEDigi:
      sch.first = &EcalDQMonitorTask::runOnCollection<EEDigiCollection>; break;
    case ecaldqm::kPnDiodeDigi:
      sch.first = &EcalDQMonitorTask::runOnCollection<EcalPnDiodeDigiCollection>; break;
    case ecaldqm::kTrigPrimDigi:
      sch.first = &EcalDQMonitorTask::runOnCollection<EcalTrigPrimDigiCollection>; break;
    case ecaldqm::kTrigPrimEmulDigi:
      sch.first = &EcalDQMonitorTask::runOnCollection<EcalTrigPrimDigiCollection>; break;
    case ecaldqm::kEBUncalibRecHit:
      sch.first = &EcalDQMonitorTask::runOnCollection<EcalUncalibratedRecHitCollection>; break;
    case ecaldqm::kEEUncalibRecHit:
      sch.first = &EcalDQMonitorTask::runOnCollection<EcalUncalibratedRecHitCollection>; break;
    case ecaldqm::kEBRecHit:
      sch.first = &EcalDQMonitorTask::runOnCollection<EcalRecHitCollection>; break;
    case ecaldqm::kEERecHit:
      sch.first = &EcalDQMonitorTask::runOnCollection<EcalRecHitCollection>; break;
    case ecaldqm::kEBBasicCluster:
      sch.first = &EcalDQMonitorTask::runOnCollection<reco::BasicClusterCollection>; break;
    case ecaldqm::kEEBasicCluster:
      sch.first = &EcalDQMonitorTask::runOnCollection<reco::BasicClusterCollection>; break;
    case ecaldqm::kEBSuperCluster:
      sch.first = &EcalDQMonitorTask::runOnCollection<reco::SuperClusterCollection>; break;
    case ecaldqm::kEESuperCluster:
      sch.first = &EcalDQMonitorTask::runOnCollection<reco::SuperClusterCollection>; break;
    default:
      throw cms::Exception("InvalidConfiguration") << "Undefined collection " << *colItr;
    }

    sch.second = *colItr;

    schedule_.push_back(sch);
  }

}
