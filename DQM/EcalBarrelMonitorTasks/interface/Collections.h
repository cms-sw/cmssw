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
    kEBLaserLedUncalibRecHit, // 20
    kEELaserLedUncalibRecHit, // 21
    kEBTestPulseUncalibRecHit, // 22
    kEETestPulseUncalibRecHit, // 23
    kEBRecHit, // 24
    kEERecHit, // 25
    kEBBasicCluster, // 26
    kEEBasicCluster, // 27
    kEBSuperCluster, // 28
    kEESuperCluster, // 29
    kRun, // 30
    kLumiSection, // 31
    nProcessedObjects, // 32
    nCollections = kRun  // 30
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
    "EBLaserLedUncalibRecHit",
    "EELaserLedUncalibRecHit",
    "EBTestPulseUncalibRecHit",
    "EETestPulseUncalibRecHit",
    "EBRecHit",
    "EERecHit",
    "EBBasicCluster",
    "EEBasicCluster",
    "EBSuperCluster",
    "EESuperCluster"
  };

}

#endif
