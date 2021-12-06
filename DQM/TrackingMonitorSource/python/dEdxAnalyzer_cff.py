import FWCore.ParameterSet.Config as cms

# dEdx monitor ####
#from DQM.TrackingMonitor.dEdxAnalyzer_cff import *
from DQM.TrackingMonitor.dEdxAnalyzer_cfi import *
# Clone for all PDs but ZeroBias ####
dEdxMonCommon = dEdxAnalyzer.clone()

dEdxHitMonCommon = dEdxHitAnalyzer.clone()

from DQM.TrackingMonitorSource.pset4GenericTriggerEventFlag_cfi import *
# Clone for ZeroBias ####
dEdxMonMB = dEdxAnalyzer.clone(
    dEdxParameters = dEdxAnalyzer.dEdxParameters.clone(
        genericTriggerEventPSet = genericTriggerEventFlag4fullTrackerAndHLTdb
    )
)

dEdxHitMonMB = dEdxHitAnalyzer.clone(
    dEdxParameters = dEdxHitAnalyzer.dEdxParameters.clone(
        genericTriggerEventPSet = genericTriggerEventFlag4fullTrackerAndHLTdb
    )
)

# Clone for SingleMu ####
dEdxMonMU = dEdxAnalyzer.clone(
    dEdxParameters = cms.PSet(
        andOr = cms.bool(False),
        hltInputTag = cms.InputTag("TriggerResults::HLT"),
        hltPaths = cms.vstring("HLT_SingleMu40_Eta2p1_*"),
        errorReplyHlt = cms.bool(False),
        andOrHlt = cms.bool(True)
    )
)

dEdxHitMonMU = dEdxHitAnalyzer.clone(
    dEdxParameters = cms.PSet(
        andOr = cms.bool(False),
        hltInputTag = cms.InputTag("TriggerResults::HLT"),
        hltPaths = cms.vstring("HLT_SingleMu40_Eta2p1_*"),
        errorReplyHlt = cms.bool(False),
        andOrHlt = cms.bool(True)
    )
)
