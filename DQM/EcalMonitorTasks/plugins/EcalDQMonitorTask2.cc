#include "DQM/EcalMonitorTasks/interface/EcalDQMonitorTask.h"

#include "DQM/EcalMonitorTasks/interface/DQWorkerTask.h"

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

template <typename CollectionClass>
void EcalDQMonitorTask::runOnCollection(edm::Event const& _evt,
                                        ecaldqm::Collections _col,
                                        std::set<ecaldqm::DQWorker*> const& _enabledTasks) {
  edm::Handle<CollectionClass> hndl;
  if (!_evt.getByToken(collectionTokens_[_col], hndl)) {
    if (!allowMissingCollections_)
      throw cms::Exception("ObjectNotFound")
          << moduleName_ << "::runOnCollection: " << ecaldqm::collectionName[_col] << " does not exist";
    edm::LogWarning("EcalDQM") << moduleName_ << "::runOnCollection: " << ecaldqm::collectionName[_col]
                               << " does not exist";
    return;
  }

  CollectionClass const* collection(hndl.product());

  executeOnWorkers_(
      [collection, _col, &_enabledTasks](ecaldqm::DQWorker* worker) {
        if (_enabledTasks.find(worker) != _enabledTasks.end())
          static_cast<ecaldqm::DQWorkerTask*>(worker)->analyze(collection, _col);
      },
      "analyze");

  if (verbosity_ > 1)
    edm::LogInfo("EcalDQM") << moduleName_ << "::runOn" << ecaldqm::collectionName[_col] << " returning";
}

void EcalDQMonitorTask::formSchedule(std::vector<ecaldqm::Collections> const& _preSchedule,
                                     edm::ParameterSet const& _tagPSet) {
  std::vector<ecaldqm::Collections> collectionsToSkip;
  for (const auto& skipColName : skipCollections_) {
    for (unsigned iCol = 0; iCol <= ecaldqm::nCollections; iCol++) {
      if (iCol == ecaldqm::nCollections)
        throw cms::Exception("InvalidConfiguration")
            << moduleName_ << "::formSchedule: Collection name " << skipColName << " in skipCollections does not exist";
      if (skipColName == ecaldqm::collectionName[iCol]) {
        collectionsToSkip.push_back(ecaldqm::Collections(iCol));
        break;
      }
    }
  }

  for (std::vector<ecaldqm::Collections>::const_iterator colItr(_preSchedule.begin()); colItr != _preSchedule.end();
       ++colItr) {
    std::pair<Processor, ecaldqm::Collections> sch;

    edm::InputTag tag(_tagPSet.getUntrackedParameter<edm::InputTag>(ecaldqm::collectionName[*colItr]));

    auto skipItr = std::find(collectionsToSkip.begin(), collectionsToSkip.end(), *colItr);
    if (skipItr != collectionsToSkip.end()) {
      if (verbosity_ > 0)
        edm::LogInfo("EcalDQM") << moduleName_ << ": Skipping collection " << ecaldqm::collectionName[*colItr]
                                << " and removing from schedule";
      collectionsToSkip.erase(skipItr);
      continue;
    }

    switch (*colItr) {
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
      case ecaldqm::kEBCpuDigi:
      case ecaldqm::kEBGpuDigi:
        collectionTokens_[*colItr] = edm::EDGetToken(consumes<EBDigiCollection>(tag));
        sch.first = &EcalDQMonitorTask::runOnCollection<EBDigiCollection>;
        break;
      case ecaldqm::kEEDigi:
      case ecaldqm::kEECpuDigi:
      case ecaldqm::kEEGpuDigi:
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
      case ecaldqm::kEBCpuUncalibRecHit:
      case ecaldqm::kEECpuUncalibRecHit:
      case ecaldqm::kEBGpuUncalibRecHit:
      case ecaldqm::kEEGpuUncalibRecHit:
        collectionTokens_[*colItr] = edm::EDGetToken(consumes<EcalUncalibratedRecHitCollection>(tag));
        sch.first = &EcalDQMonitorTask::runOnCollection<EcalUncalibratedRecHitCollection>;
        break;
      case ecaldqm::kEBRecHit:
      case ecaldqm::kEBReducedRecHit:
      case ecaldqm::kEERecHit:
      case ecaldqm::kEEReducedRecHit:
      case ecaldqm::kEBCpuRecHit:
      case ecaldqm::kEECpuRecHit:
      case ecaldqm::kEBGpuRecHit:
      case ecaldqm::kEEGpuRecHit:
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
  for (const auto& colNotSkipped : collectionsToSkip)
    edm::LogWarning("EcalDQM") << moduleName_
                               << "::formSchedule: Collection: " << ecaldqm::collectionName[colNotSkipped]
                               << " is not in the schedule but was listed to be skipped";
}
