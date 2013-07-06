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

using namespace ecaldqm;

template <class C>
void
EcalDQMonitorTask::runOnCollection(const edm::Event& _evt, Collections _colName)
{
  edm::Handle<C> hndl;
  if(_evt.getByLabel(collectionTags_[_colName], hndl)){

    TStopwatch* sw(0);
    if(evaluateTime_){
      sw = new TStopwatch;
      sw->Stop();
    }

    DQWorkerTask* task(0);

    for(std::vector<DQWorkerTask *>::iterator wItr(taskLists_[_colName].begin()); wItr != taskLists_[_colName].end(); ++wItr){
      task = *wItr;
      if(evaluateTime_) sw->Start();
      if(enabled_[task]) task->analyze(hndl.product(), _colName);
      if(evaluateTime_){
	sw->Stop();
	taskTimes_[task] += sw->RealTime();
      }
    }

    delete sw;
  }
  else if(!allowMissingCollections_)
    throw cms::Exception("ObjectNotFound") << "Collection with InputTag " << collectionTags_[_colName] << " does not exist";
}

template <>
void
EcalDQMonitorTask::runOnCollection<DetIdCollection>(const edm::Event& _evt, Collections _colName)
{
  edm::Handle<EBDetIdCollection> ebHndl;
  edm::Handle<EEDetIdCollection> eeHndl;
  if(_evt.getByLabel(collectionTags_[_colName], ebHndl) && _evt.getByLabel(collectionTags_[_colName], eeHndl)){
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

    DQWorkerTask* task(0);

    for(std::vector<DQWorkerTask *>::iterator wItr(taskLists_[_colName].begin()); wItr != taskLists_[_colName].end(); ++wItr){
      task = *wItr;
      if(evaluateTime_) sw->Start();
      if(enabled_[task]) task->analyze(const_cast<const DetIdCollection*>(&ids), _colName);
      if(evaluateTime_){
	sw->Stop();
	taskTimes_[task] += sw->RealTime();
      }
    }

    delete sw;
  }
  else if(!allowMissingCollections_)
    throw cms::Exception("ObjectNotFound") << "DetIdCollection with InputTag " << collectionTags_[_colName] << " does not exist";
}

void
EcalDQMonitorTask::formSchedule_(const std::vector<Collections>& _usedCollections, const std::multimap<Collections, Collections>& _dependencies)
{
  using namespace std;
  typedef multimap<Collections, Collections>::const_iterator mmiter;

  vector<Collections> preSchedule;
  vector<Collections>::iterator insertPoint, findPoint;

  for(vector<Collections>::const_iterator colItr(_usedCollections.begin()); colItr != _usedCollections.end(); ++colItr){

    bool inserted(true);
    if((insertPoint = find(preSchedule.begin(), preSchedule.end(), *colItr)) == preSchedule.end()) inserted = false;

    pair<mmiter, mmiter> range(_dependencies.equal_range(*colItr));

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

  for(vector<Collections>::const_iterator colItr(preSchedule.begin()); colItr != preSchedule.end(); ++colItr){
    std::pair<Processor, Collections> sch;

    switch(*colItr){
    case kSource:
      sch.first = &EcalDQMonitorTask::runOnCollection<FEDRawDataCollection>; break;
    case kEcalRawData:
      sch.first = &EcalDQMonitorTask::runOnCollection<EcalRawDataCollection>; break;
    case kGainErrors:
      sch.first = &EcalDQMonitorTask::runOnCollection<DetIdCollection>; break;
    case kChIdErrors:
      sch.first = &EcalDQMonitorTask::runOnCollection<DetIdCollection>; break;
    case kGainSwitchErrors:
      sch.first = &EcalDQMonitorTask::runOnCollection<DetIdCollection>; break;
    case kTowerIdErrors:
      sch.first = &EcalDQMonitorTask::runOnCollection<EcalElectronicsIdCollection>; break;
    case kBlockSizeErrors:
      sch.first = &EcalDQMonitorTask::runOnCollection<EcalElectronicsIdCollection>; break;
    case kMEMTowerIdErrors:
      sch.first = &EcalDQMonitorTask::runOnCollection<EcalElectronicsIdCollection>; break;
    case kMEMBlockSizeErrors:
      sch.first = &EcalDQMonitorTask::runOnCollection<EcalElectronicsIdCollection>; break;
    case kMEMChIdErrors:
      sch.first = &EcalDQMonitorTask::runOnCollection<EcalElectronicsIdCollection>; break;
    case kMEMGainErrors:
      sch.first = &EcalDQMonitorTask::runOnCollection<EcalElectronicsIdCollection>; break;
    case kEBSrFlag:
      sch.first = &EcalDQMonitorTask::runOnCollection<EBSrFlagCollection>; break;
    case kEESrFlag:
      sch.first = &EcalDQMonitorTask::runOnCollection<EESrFlagCollection>; break;
    case kEBDigi:
      sch.first = &EcalDQMonitorTask::runOnCollection<EBDigiCollection>; break;
    case kEEDigi:
      sch.first = &EcalDQMonitorTask::runOnCollection<EEDigiCollection>; break;
    case kPnDiodeDigi:
      sch.first = &EcalDQMonitorTask::runOnCollection<EcalPnDiodeDigiCollection>; break;
    case kTrigPrimDigi:
      sch.first = &EcalDQMonitorTask::runOnCollection<EcalTrigPrimDigiCollection>; break;
    case kTrigPrimEmulDigi:
      sch.first = &EcalDQMonitorTask::runOnCollection<EcalTrigPrimDigiCollection>; break;
    case kEBUncalibRecHit:
      sch.first = &EcalDQMonitorTask::runOnCollection<EcalUncalibratedRecHitCollection>; break;
    case kEEUncalibRecHit:
      sch.first = &EcalDQMonitorTask::runOnCollection<EcalUncalibratedRecHitCollection>; break;
    case kEBRecHit:
      sch.first = &EcalDQMonitorTask::runOnCollection<EcalRecHitCollection>; break;
    case kEERecHit:
      sch.first = &EcalDQMonitorTask::runOnCollection<EcalRecHitCollection>; break;
    case kEBBasicCluster:
      sch.first = &EcalDQMonitorTask::runOnCollection<reco::BasicClusterCollection>; break;
    case kEEBasicCluster:
      sch.first = &EcalDQMonitorTask::runOnCollection<reco::BasicClusterCollection>; break;
    case kEBSuperCluster:
      sch.first = &EcalDQMonitorTask::runOnCollection<reco::SuperClusterCollection>; break;
    case kEESuperCluster:
      sch.first = &EcalDQMonitorTask::runOnCollection<reco::SuperClusterCollection>; break;
    default:
      throw cms::Exception("InvalidConfiguration") << "Undefined collection " << *colItr;
    }

    sch.second = *colItr;

    schedule_.push_back(sch);
  }

}
