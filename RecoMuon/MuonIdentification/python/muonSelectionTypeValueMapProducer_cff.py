import FWCore.ParameterSet.Config as cms

from RecoMuon.MuonIdentification.muonSelectionTypeValueMapProducer_cfi import *

muidTrackerMuonArbitrated = muonSelectionTypeValueMapProducer.clone()
muidTrackerMuonArbitrated.selectionType = cms.string("TrackerMuonArbitrated")
#
muidAllArbitrated = muonSelectionTypeValueMapProducer.clone()
muidAllArbitrated.selectionType = cms.string("AllArbitrated")
#
muidGlobalMuonPromptTight = muonSelectionTypeValueMapProducer.clone()
muidGlobalMuonPromptTight.selectionType = cms.string("GlobalMuonPromptTight")
#
muidTMLastStationLoose = muonSelectionTypeValueMapProducer.clone()
muidTMLastStationLoose.selectionType = cms.string("TMLastStationLoose")
#
muidTMLastStationTight = muonSelectionTypeValueMapProducer.clone()
muidTMLastStationTight.selectionType = cms.string("TMLastStationTight")
#
muidTM2DCompatibilityLoose = muonSelectionTypeValueMapProducer.clone()
muidTM2DCompatibilityLoose.selectionType = cms.string("TM2DCompatibilityLoose")
#
muidTM2DCompatibilityTight = muonSelectionTypeValueMapProducer.clone()
muidTM2DCompatibilityTight.selectionType = cms.string("TM2DCompatibilityTight")
#
muidTMOneStationLoose = muonSelectionTypeValueMapProducer.clone()
muidTMOneStationLoose.selectionType = cms.string("TMOneStationLoose")
#
muidTMOneStationTight = muonSelectionTypeValueMapProducer.clone()
muidTMOneStationTight.selectionType = cms.string("TMOneStationTight")
#
muidTMLastStationOptimizedLowPtLoose = muonSelectionTypeValueMapProducer.clone()
muidTMLastStationOptimizedLowPtLoose.selectionType = cms.string("TMLastStationOptimizedLowPtLoose")
#
muidTMLastStationOptimizedLowPtTight = muonSelectionTypeValueMapProducer.clone()
muidTMLastStationOptimizedLowPtTight.selectionType = cms.string("TMLastStationOptimizedLowPtTight")
#
muidGMTkChiCompatibility = muonSelectionTypeValueMapProducer.clone()
muidGMTkChiCompatibility.selectionType = cms.string("GMTkChiCompatibility")
#
muidGMStaChiCompatibility = muonSelectionTypeValueMapProducer.clone()
muidGMStaChiCompatibility.selectionType = cms.string("GMStaChiCompatibility")
#
muidGMTkKinkTight = muonSelectionTypeValueMapProducer.clone()
muidGMTkKinkTight.selectionType = cms.string("GMTkKinkTight")
#
muidTMLastStationAngLoose = muonSelectionTypeValueMapProducer.clone()
muidTMLastStationAngLoose.selectionType = cms.string("TMLastStationAngLoose")
#
muidTMLastStationAngTight = muonSelectionTypeValueMapProducer.clone()
muidTMLastStationAngTight.selectionType = cms.string("TMLastStationAngTight")
#
muidTMOneStationAngLoose = muonSelectionTypeValueMapProducer.clone()
muidTMOneStationAngLoose.selectionType = cms.string("TMOneStationAngLoose")
#
muidTMOneStationAngTight = muonSelectionTypeValueMapProducer.clone()
muidTMOneStationAngTight.selectionType = cms.string("TMOneStationAngTight")
#
muidRPCMuLoose = muonSelectionTypeValueMapProducer.clone()
muidRPCMuLoose.selectionType = cms.string("RPCMuLoose")
#
muonSelectionTypeSequence = cms.Sequence(
    muidTrackerMuonArbitrated
    +muidAllArbitrated
    +muidGlobalMuonPromptTight
    +muidTMLastStationLoose
    +muidTMLastStationTight
    +muidTM2DCompatibilityLoose
    +muidTM2DCompatibilityTight
    +muidTMOneStationLoose
    +muidTMOneStationTight
    +muidTMLastStationOptimizedLowPtLoose
    +muidTMLastStationOptimizedLowPtTight
    +muidGMTkChiCompatibility
    +muidGMStaChiCompatibility
    +muidGMTkKinkTight
    +muidTMLastStationAngLoose
    +muidTMLastStationAngTight
    +muidTMOneStationAngLoose
    +muidTMOneStationAngTight
    +muidRPCMuLoose)
