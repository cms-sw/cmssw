import FWCore.ParameterSet.Config as cms

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


# L1 systems quality tests

# ECAL quality tests
from DQM.L1TMonitorClient.L1EmulatorEcalQualityTests_cfi import *
seqL1EmulatorEcalQualityTests = cms.Sequence(l1EmulatorEcalQualityTests)

# HCAL quality tests
from DQM.L1TMonitorClient.L1EmulatorHcalQualityTests_cfi import *
seqL1EmulatorHcalQualityTests = cms.Sequence(l1EmulatorHcalQualityTests)

# RCT quality tests
from DQM.L1TMonitorClient.L1EmulatorRctQualityTests_cfi import *
seqL1EmulatorRctQualityTests = cms.Sequence(l1EmulatorRctQualityTests)

# GCT quality tests
from DQM.L1TMonitorClient.L1EmulatorGctQualityTests_cfi import *
seqL1EmulatorGctQualityTests = cms.Sequence(l1EmulatorGctQualityTests)

# DTTF quality tests
from DQM.L1TMonitorClient.L1EmulatorDttfQualityTests_cfi import *
seqL1EmulatorDttfQualityTests = cms.Sequence(l1EmulatorDttfQualityTests)

# DTTPG quality tests
#from DQM.L1TMonitorClient.L1EmulatorDttpgQualityTests_cfi import *
#seqL1EmulatorDttpgQualityTests = cms.Sequence(l1EmulatorDttpgQualityTests)

# CSCTF quality tests
from DQM.L1TMonitorClient.L1EmulatorCsctfQualityTests_cfi import *
seqL1EmulatorCsctfQualityTests = cms.Sequence(l1EmulatorCsctfQualityTests)

# CSCTPG quality tests
from DQM.L1TMonitorClient.L1EmulatorCsctpgQualityTests_cfi import *
seqL1EmulatorCsctpgQualityTests = cms.Sequence(l1EmulatorCsctpgQualityTests)

# RPC quality tests
from DQM.L1TMonitorClient.L1EmulatorRpcQualityTests_cfi import *
seqL1EmulatorRpcQualityTests = cms.Sequence(l1EmulatorRpcQualityTests)

# GMT quality tests
from DQM.L1TMonitorClient.L1EmulatorGmtQualityTests_cfi import *
seqL1EmulatorGmtQualityTests = cms.Sequence(l1EmulatorGmtQualityTests)

# GT quality tests
from DQM.L1TMonitorClient.L1EmulatorGtQualityTests_cfi import *
seqL1EmulatorGtQualityTests = cms.Sequence(l1EmulatorGtQualityTests)

# L1 objects quality tests

# GtExternal quality tests
from DQM.L1TMonitorClient.L1EmulatorObjGtExternalQualityTests_cfi import *
seqL1EmulatorObjGtExternalQualityTests = cms.Sequence(l1EmulatorObjGtExternalQualityTests)

# TechTrig quality tests
from DQM.L1TMonitorClient.L1EmulatorObjTechTrigQualityTests_cfi import *
seqL1EmulatorObjTechTrigQualityTests = cms.Sequence(l1EmulatorObjTechTrigQualityTests)

# HfRingEtSums quality tests
from DQM.L1TMonitorClient.L1EmulatorObjHfRingEtSumsQualityTests_cfi import *
seqL1EmulatorObjHfRingEtSumsQualityTests = cms.Sequence(l1EmulatorObjHfRingEtSumsQualityTests)

# HfBitCounts quality tests
from DQM.L1TMonitorClient.L1EmulatorObjHfBitCountsQualityTests_cfi import *
seqL1EmulatorObjHfBitCountsQualityTests = cms.Sequence(l1EmulatorObjHfBitCountsQualityTests)

# HTM quality tests
from DQM.L1TMonitorClient.L1EmulatorObjHTMQualityTests_cfi import *
seqL1EmulatorObjHTMQualityTests = cms.Sequence(l1EmulatorObjHTMQualityTests)

# HTT quality tests
from DQM.L1TMonitorClient.L1EmulatorObjHTTQualityTests_cfi import *
seqL1EmulatorObjHTTQualityTests = cms.Sequence(l1EmulatorObjHTTQualityTests)

# ETM quality tests
from DQM.L1TMonitorClient.L1EmulatorObjETMQualityTests_cfi import *
seqL1EmulatorObjETMQualityTests = cms.Sequence(l1EmulatorObjETMQualityTests)

# ETT quality tests
from DQM.L1TMonitorClient.L1EmulatorObjETTQualityTests_cfi import *
seqL1EmulatorObjETTQualityTests = cms.Sequence(l1EmulatorObjETTQualityTests)

# TauJet quality tests
from DQM.L1TMonitorClient.L1EmulatorObjTauJetQualityTests_cfi import *
seqL1EmulatorObjTauJetQualityTests = cms.Sequence(l1EmulatorObjTauJetQualityTests)

# ForJet quality tests
from DQM.L1TMonitorClient.L1EmulatorObjForJetQualityTests_cfi import *
seqL1EmulatorObjForJetQualityTests = cms.Sequence(l1EmulatorObjForJetQualityTests)

# CenJet quality tests
from DQM.L1TMonitorClient.L1EmulatorObjCenJetQualityTests_cfi import *
seqL1EmulatorObjCenJetQualityTests = cms.Sequence(l1EmulatorObjCenJetQualityTests)

# IsoEG quality tests
from DQM.L1TMonitorClient.L1EmulatorObjIsoEGQualityTests_cfi import *
seqL1EmulatorObjIsoEGQualityTests = cms.Sequence(l1EmulatorObjIsoEGQualityTests)

# NoIsoEG quality tests
from DQM.L1TMonitorClient.L1EmulatorObjNoIsoEGQualityTests_cfi import *
seqL1EmulatorObjNoIsoEGQualityTests = cms.Sequence(l1EmulatorObjNoIsoEGQualityTests)

# Mu quality tests
from DQM.L1TMonitorClient.L1EmulatorObjMuQualityTests_cfi import *
seqL1EmulatorObjMuQualityTests = cms.Sequence(l1EmulatorObjMuQualityTests)

# sequence for L1 systems
l1EmulatorSystemQualityTests = cms.Sequence(
                                seqL1EmulatorEcalQualityTests + 
                                seqL1EmulatorHcalQualityTests + 
                                seqL1EmulatorRctQualityTests + 
                                seqL1EmulatorGctQualityTests + 
                                seqL1EmulatorDttfQualityTests + 
                                #seqL1EmulatorDttpgQualityTests + 
                                seqL1EmulatorCsctfQualityTests + 
                                seqL1EmulatorCsctpgQualityTests + 
                                seqL1EmulatorRpcQualityTests + 
                                seqL1EmulatorGmtQualityTests + 
                                seqL1EmulatorGtQualityTests
                                )

# sequence for L1 objects
l1EmulatorObjectQualityTests = cms.Sequence(
                                seqL1EmulatorObjTechTrigQualityTests +
                                seqL1EmulatorObjGtExternalQualityTests +
                                seqL1EmulatorObjHfRingEtSumsQualityTests +
                                seqL1EmulatorObjHfBitCountsQualityTests +
                                seqL1EmulatorObjHTMQualityTests +
                                seqL1EmulatorObjHTTQualityTests +
                                seqL1EmulatorObjETMQualityTests +
                                seqL1EmulatorObjETTQualityTests +
                                seqL1EmulatorObjTauJetQualityTests +
                                seqL1EmulatorObjForJetQualityTests +
                                seqL1EmulatorObjCenJetQualityTests +
                                seqL1EmulatorObjIsoEGQualityTests +
                                seqL1EmulatorObjNoIsoEGQualityTests +
                                seqL1EmulatorObjMuQualityTests
                                )


# general sequence
l1EmulatorQualityTests = cms.Sequence(
                                      l1EmulatorSystemQualityTests + 
                                      l1EmulatorObjectQualityTests
                                      )

