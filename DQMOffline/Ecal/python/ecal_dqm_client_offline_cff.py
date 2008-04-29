import FWCore.ParameterSet.Config as cms

from DQM.EcalBarrelMonitorClient.EcalBarrelMonitorClient_cfi import *
dqmQTestEB = cms.EDFilter("QualityTester",
    reportThreshold = cms.untracked.string('red'),
    prescaleFactor = cms.untracked.int32(1),
    qtList = cms.untracked.FileInPath('DQM/EcalBarrelMonitorModule/test/data/EcalBarrelQualityTests.xml'),
    getQualityTestsFromFile = cms.untracked.bool(True)
)

ecal_dqm_client-offline = cms.Sequence(ecalBarrelMonitorClient*dqmQTestEB)
ecalBarrelMonitorClient.maskFile = ''
ecalBarrelMonitorClient.location = 'P5'
ecalBarrelMonitorClient.verbose = False
ecalBarrelMonitorClient.enabledClients = ['Integrity', 'StatusFlags', 'Occupancy', 'PedestalOnline', 'Timing', 
    'Cosmic', 'Cluster', 'TriggerTower', 'Summary']


