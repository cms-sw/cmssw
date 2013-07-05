#ifndef EcalDQMonitorTaskCollections_H
#define EcalDQMonitorTaskCollections_H

namespace ecaldqm {

  enum Collections {
    kSource, // 0
    kEcalRawData, // 1
    kGainErrors, // 2
    kChIdErrors, // 3
    kGainSwitchErrors, // 4
    kTowerIdErrors, // 5
    kBlockSizeErrors, // 6
    kMEMTowerIdErrors, // 7
    kMEMBlockSizeErrors, // 8
    kMEMChIdErrors, // 9
    kMEMGainErrors, // 10
    kEBSrFlag, // 11
    kEESrFlag, // 12
    kEBDigi, // 13
    kEEDigi, // 14
    kPnDiodeDigi, // 15
    kTrigPrimDigi, // 16
    kTrigPrimEmulDigi, // 17
    kEBUncalibRecHit, // 18
    kEEUncalibRecHit, // 19
    kEBRecHit, // 20
    kEERecHit, // 21
    kEBBasicCluster, // 22
    kEEBasicCluster, // 23
    kEBSuperCluster, // 24
    kEESuperCluster, // 25
    kRun, // 26
    kLumiSection, // 27
    nProcessedObjects, // 28
    nCollections = kRun  // 26
  };

  std::string const collectionName[nCollections] = {
    "Source",
    "EcalRawData",
    "GainErrors",
    "ChIdErrors",
    "GainSwitchErrors",
    "TowerIdErrors",
    "BlockSizeErrors",
    "MEMTowerIdErrors",
    "MEMBlockSizeErrors",
    "MEMChIdErrors",
    "MEMGainErrors",
    "EBSrFlag",
    "EESrFlag",
    "EBDigi",
    "EEDigi",
    "PnDiodeDigi",
    "TrigPrimDigi",
    "TrigPrimEmulDigi",
    "EBUncalibRecHit",
    "EEUncalibRecHit",
    "EBRecHit",
    "EERecHit",
    "EBBasicCluster",
    "EEBasicCluster",
    "EBSuperCluster",
    "EESuperCluster"
  };

}

#endif
