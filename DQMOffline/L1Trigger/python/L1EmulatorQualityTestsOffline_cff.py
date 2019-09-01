import FWCore.ParameterSet.Config as cms

# adapt the L1EmulatorQualityTests_cff configuration to offline DQM

#
# default configuration valid for online DQM
#
# configuration for online DQM
#    perform tests on endLumi
#    perform tests on endRun
#
# configuration for offline DQM
#    perform tests on endRun only
#
# for both online and offline
#    get the quality tests from an XML file
#    no tests in event loop
#    do not prescale
#    verboseQT true, but reportThreshold empty


# perform offline the quality tests in the clients in endRun only
from DQM.L1TMonitorClient.L1EmulatorQualityTests_cff import *
    

# L1 systems quality tests

# ECAL quality tests
l1EmulatorEcalQualityTests.qtestOnEndLumi = False

# HCAL quality tests
l1EmulatorHcalQualityTests.qtestOnEndLumi = False

# RCT quality tests
l1EmulatorRctQualityTests.qtestOnEndLumi = False

# GCT quality tests
l1EmulatorGctQualityTests.qtestOnEndLumi = False

# DTTF quality tests
l1EmulatorDttfQualityTests.qtestOnEndLumi = False

# DTTPG quality tests
#l1EmulatorDttpgQualityTests.qtestOnEndLumi = False

# CSCTF quality tests
l1EmulatorCsctfQualityTests.qtestOnEndLumi = False

# CSCTPG quality tests
l1EmulatorCsctpgQualityTests.qtestOnEndLumi = False

# RPC quality tests
l1EmulatorRpcQualityTests.qtestOnEndLumi = False

# GMT quality tests
l1EmulatorGmtQualityTests.qtestOnEndLumi = False

# GT quality tests
l1EmulatorGtQualityTests.qtestOnEndLumi = False

# L1 objects quality tests

# GtExternal quality tests
l1EmulatorObjGtExternalQualityTests.qtestOnEndLumi = False

# TechTrig quality tests
l1EmulatorObjTechTrigQualityTests.qtestOnEndLumi = False

# HfRingEtSums quality tests
l1EmulatorObjHfRingEtSumsQualityTests.qtestOnEndLumi = False

# HfBitCounts quality tests
l1EmulatorObjHfBitCountsQualityTests.qtestOnEndLumi = False

# HTM quality tests
l1EmulatorObjHTMQualityTests.qtestOnEndLumi = False

# HTT quality tests
l1EmulatorObjHTTQualityTests.qtestOnEndLumi = False

# ETM quality tests
l1EmulatorObjETMQualityTests.qtestOnEndLumi = False

# ETT quality tests
l1EmulatorObjETTQualityTests.qtestOnEndLumi = False

# TauJet quality tests
l1EmulatorObjTauJetQualityTests.qtestOnEndLumi = False

# ForJet quality tests
l1EmulatorObjForJetQualityTests.qtestOnEndLumi = False

# CenJet quality tests
l1EmulatorObjCenJetQualityTests.qtestOnEndLumi = False

# IsoEG quality tests
l1EmulatorObjIsoEGQualityTests.qtestOnEndLumi = False

# NoIsoEG quality tests
l1EmulatorObjNoIsoEGQualityTests.qtestOnEndLumi = False

# IsoTau quality tests
#l1EmulatorObjIsoTauQualityTests.qtestOnEndLumi = False

# NoIsoTau quality tests
#l1EmulatorObjNoIsoTauQualityTests.qtestOnEndLumi = False

# Mu quality tests
l1EmulatorObjMuQualityTests.qtestOnEndLumi = False

# stage 2
from DQM.L1TMonitorClient.L1TStage2QualityTests_cff import *
l1TStage2CaloLayer1QualityTests.qtestOnEndLumi = cms.untracked.bool(False)
l1TStage2BMTFQualityTests.qtestOnEndLumi = cms.untracked.bool(False)
l1TStage2OMTFQualityTests.qtestOnEndLumi = cms.untracked.bool(False)
l1TStage2EMTFQualityTests.qtestOnEndLumi = cms.untracked.bool(False)
l1TStage2uGMTQualityTests.qtestOnEndLumi = cms.untracked.bool(False)
l1TStage2uGTQualityTests.qtestOnEndLumi = cms.untracked.bool(False)
l1TStage2MuonQualityTests.qtestOnEndLumi = cms.untracked.bool(False)
l1TStage2MuonQualityTestsCollisions.qtestOnEndLumi = cms.untracked.bool(False)

from DQM.L1TMonitorClient.L1TStage2EmulatorQualityTests_cff import *
l1TStage2CaloLayer1DEQualityTests.qtestOnEndLumi = cms.untracked.bool(False)
l1TStage2BMTFDEQualityTests.qtestOnEndLumi = cms.untracked.bool(False)
l1TStage2OMTFDEQualityTests.qtestOnEndLumi = cms.untracked.bool(False)
l1TStage2EMTFDEQualityTests.qtestOnEndLumi = cms.untracked.bool(False)
l1TStage2uGMTDEQualityTests.qtestOnEndLumi = cms.untracked.bool(False)
l1TStage2uGTDEQualityTests.qtestOnEndLumi = cms.untracked.bool(False)

