# Quality tests for L1 Trigger DQM (L1T)

import FWCore.ParameterSet.Config as cms

# L1 systems quality tests

# ECAL quality tests
from DQM.L1TMonitorClient.L1TriggerEcalQualityTests_cfi import *
seqL1TriggerEcalQualityTests = cms.Sequence(l1TriggerEcalQualityTests)

# HCAL quality tests
from DQM.L1TMonitorClient.L1TriggerHcalQualityTests_cfi import *
seqL1TriggerHcalQualityTests = cms.Sequence(l1TriggerHcalQualityTests)

# RCT quality tests
from DQM.L1TMonitorClient.L1TriggerRctQualityTests_cfi import *
seqL1TriggerRctQualityTests = cms.Sequence(l1TriggerRctQualityTests)

# GCT quality tests
from DQM.L1TMonitorClient.L1TriggerGctQualityTests_cfi import *
seqL1TriggerGctQualityTests = cms.Sequence(l1TriggerGctQualityTests)

# DTTF quality tests
from DQM.L1TMonitorClient.L1TriggerDttfQualityTests_cfi import *
seqL1TriggerDttfQualityTests = cms.Sequence(l1TriggerDttfQualityTests)

# DTTPG quality tests
from DQM.L1TMonitorClient.L1TriggerDttpgQualityTests_cfi import *
seqL1TriggerDttpgQualityTests = cms.Sequence(l1TriggerDttpgQualityTests)

# CSCTF quality tests
from DQM.L1TMonitorClient.L1TriggerCsctfQualityTests_cfi import *
seqL1TriggerCsctfQualityTests = cms.Sequence(l1TriggerCsctfQualityTests)

# CSCTPG quality tests
from DQM.L1TMonitorClient.L1TriggerCsctpgQualityTests_cfi import *
seqL1TriggerCsctpgQualityTests = cms.Sequence(l1TriggerCsctpgQualityTests)

# RPC quality tests
from DQM.L1TMonitorClient.L1TriggerRpcQualityTests_cfi import *
seqL1TriggerRpcQualityTests = cms.Sequence(l1TriggerRpcQualityTests)

# GMT quality tests
from DQM.L1TMonitorClient.L1TriggerGmtQualityTests_cfi import *
seqL1TriggerGmtQualityTests = cms.Sequence(l1TriggerGmtQualityTests)

# GT quality tests
from DQM.L1TMonitorClient.L1TriggerGtQualityTests_cfi import *
seqL1TriggerGtQualityTests = cms.Sequence(l1TriggerGtQualityTests)

# L1 objects quality tests

# GtExternal quality tests
from DQM.L1TMonitorClient.L1TriggerObjGtExternalQualityTests_cfi import *
seqL1TriggerObjGtExternalQualityTests = cms.Sequence(l1TriggerObjGtExternalQualityTests)

# TechTrig quality tests
from DQM.L1TMonitorClient.L1TriggerObjTechTrigQualityTests_cfi import *
seqL1TriggerObjTechTrigQualityTests = cms.Sequence(l1TriggerObjTechTrigQualityTests)

# HfRingEtSums quality tests
from DQM.L1TMonitorClient.L1TriggerObjHfRingEtSumsQualityTests_cfi import *
seqL1TriggerObjHfRingEtSumsQualityTests = cms.Sequence(l1TriggerObjHfRingEtSumsQualityTests)

# HfBitCounts quality tests
from DQM.L1TMonitorClient.L1TriggerObjHfBitCountsQualityTests_cfi import *
seqL1TriggerObjHfBitCountsQualityTests = cms.Sequence(l1TriggerObjHfBitCountsQualityTests)

# HTM quality tests
from DQM.L1TMonitorClient.L1TriggerObjHTMQualityTests_cfi import *
seqL1TriggerObjHTMQualityTests = cms.Sequence(l1TriggerObjHTMQualityTests)

# HTT quality tests
from DQM.L1TMonitorClient.L1TriggerObjHTTQualityTests_cfi import *
seqL1TriggerObjHTTQualityTests = cms.Sequence(l1TriggerObjHTTQualityTests)

# ETM quality tests
from DQM.L1TMonitorClient.L1TriggerObjETMQualityTests_cfi import *
seqL1TriggerObjETMQualityTests = cms.Sequence(l1TriggerObjETMQualityTests)

# ETT quality tests
from DQM.L1TMonitorClient.L1TriggerObjETTQualityTests_cfi import *
seqL1TriggerObjETTQualityTests = cms.Sequence(l1TriggerObjETTQualityTests)

# TauJet quality tests
from DQM.L1TMonitorClient.L1TriggerObjTauJetQualityTests_cfi import *
seqL1TriggerObjTauJetQualityTests = cms.Sequence(l1TriggerObjTauJetQualityTests)

# ForJet quality tests
from DQM.L1TMonitorClient.L1TriggerObjForJetQualityTests_cfi import *
seqL1TriggerObjForJetQualityTests = cms.Sequence(l1TriggerObjForJetQualityTests)

# CenJet quality tests
from DQM.L1TMonitorClient.L1TriggerObjCenJetQualityTests_cfi import *
seqL1TriggerObjCenJetQualityTests = cms.Sequence(l1TriggerObjCenJetQualityTests)

# IsoEG quality tests
from DQM.L1TMonitorClient.L1TriggerObjIsoEGQualityTests_cfi import *
seqL1TriggerObjIsoEGQualityTests = cms.Sequence(l1TriggerObjIsoEGQualityTests)

# NoIsoEG quality tests
from DQM.L1TMonitorClient.L1TriggerObjNoIsoEGQualityTests_cfi import *
seqL1TriggerObjNoIsoEGQualityTests = cms.Sequence(l1TriggerObjNoIsoEGQualityTests)

# Mu quality tests
from DQM.L1TMonitorClient.L1TriggerObjMuQualityTests_cfi import *
seqL1TriggerObjMuQualityTests = cms.Sequence(l1TriggerObjMuQualityTests)

# L1 trigger rate quality test
from DQM.L1TMonitorClient.L1TriggerRateQualityTests_cfi import *

# L1 trigger synchronization quality test
from DQM.L1TMonitorClient.L1TriggerSyncQualityTests_cfi import *

# L1 trigger occupancy quality test
from DQM.L1TMonitorClient.L1TriggerOccupancyQualityTests_cfi import *

# sequence for L1 systems
l1TriggerSystemQualityTests = cms.Sequence(
                                seqL1TriggerEcalQualityTests + 
                                seqL1TriggerHcalQualityTests + 
                                seqL1TriggerRctQualityTests + 
                                seqL1TriggerGctQualityTests + 
                                seqL1TriggerDttfQualityTests + 
                                seqL1TriggerDttpgQualityTests + 
                                seqL1TriggerCsctfQualityTests + 
                                seqL1TriggerCsctpgQualityTests + 
                                seqL1TriggerRpcQualityTests + 
                                seqL1TriggerGmtQualityTests + 
                                seqL1TriggerGtQualityTests
                                )

# sequence for L1 objects
l1TriggerObjectQualityTests = cms.Sequence(
                                seqL1TriggerObjTechTrigQualityTests +
                                seqL1TriggerObjGtExternalQualityTests +
                                seqL1TriggerObjHfRingEtSumsQualityTests +
                                seqL1TriggerObjHfBitCountsQualityTests +
                                seqL1TriggerObjHTMQualityTests +
                                seqL1TriggerObjHTTQualityTests +
                                seqL1TriggerObjETMQualityTests +
                                seqL1TriggerObjETTQualityTests +
                                seqL1TriggerObjTauJetQualityTests +
                                seqL1TriggerObjForJetQualityTests +
                                seqL1TriggerObjCenJetQualityTests +
                                seqL1TriggerObjIsoEGQualityTests +
                                seqL1TriggerObjNoIsoEGQualityTests +
                                seqL1TriggerObjMuQualityTests
                                )


# general sequence
l1TriggerQualityTests = cms.Sequence(
                                      l1TriggerSystemQualityTests + 
                                      l1TriggerObjectQualityTests + 
                                      l1TriggerRateQualityTests +
                                      l1TriggerSyncQualityTests +
                                      l1TriggerOccupancyQualityTests
                                      )

