import FWCore.ParameterSet.Config as cms

from RecoMuon.MuonIdentification.muonSelectionTypeValueMapProducer_cfi import *

muidTrackerMuonArbitrated = muonSelectionTypeValueMapProducer.clone(
    selectionType = 'TrackerMuonArbitrated'
)

muidAllArbitrated = muonSelectionTypeValueMapProducer.clone(
    selectionType = 'AllArbitrated'
)

muidGlobalMuonPromptTight = muonSelectionTypeValueMapProducer.clone(
    selectionType = 'GlobalMuonPromptTight'
)

muidTMLastStationLoose = muonSelectionTypeValueMapProducer.clone(
    selectionType = 'TMLastStationLoose'
)

muidTMLastStationTight = muonSelectionTypeValueMapProducer.clone(
    selectionType = 'TMLastStationTight'
)

muidTM2DCompatibilityLoose = muonSelectionTypeValueMapProducer.clone(
    selectionType = 'TM2DCompatibilityLoose'
)

muidTM2DCompatibilityTight = muonSelectionTypeValueMapProducer.clone(
    selectionType = 'TM2DCompatibilityTight'
)

muidTMOneStationLoose = muonSelectionTypeValueMapProducer.clone(
    selectionType = 'TMOneStationLoose'
)

muidTMOneStationTight = muonSelectionTypeValueMapProducer.clone(
    selectionType = 'TMOneStationTight'
)

muidTMLastStationOptimizedLowPtLoose = muonSelectionTypeValueMapProducer.clone(
    selectionType = 'TMLastStationOptimizedLowPtLoose'
)

muidTMLastStationOptimizedLowPtTight = muonSelectionTypeValueMapProducer.clone(
    selectionType = 'TMLastStationOptimizedLowPtTight'
)

muidGMTkChiCompatibility = muonSelectionTypeValueMapProducer.clone(
    selectionType = 'GMTkChiCompatibility'
)

muidGMStaChiCompatibility = muonSelectionTypeValueMapProducer.clone(
    selectionType = 'GMStaChiCompatibility'
)

muidGMTkKinkTight = muonSelectionTypeValueMapProducer.clone(
    selectionType = 'GMTkKinkTight'
)

muidTMLastStationAngLoose = muonSelectionTypeValueMapProducer.clone(
    selectionType = 'TMLastStationAngLoose'
)

muidTMLastStationAngTight = muonSelectionTypeValueMapProducer.clone(
    selectionType = 'TMLastStationAngTight'
)

muidTMOneStationAngLoose = muonSelectionTypeValueMapProducer.clone(
    selectionType = 'TMOneStationAngLoose'
)

muidTMOneStationAngTight = muonSelectionTypeValueMapProducer.clone(
    selectionType = 'TMOneStationAngTight'
)

muidRPCMuLoose = muonSelectionTypeValueMapProducer.clone(
    selectionType = 'RPCMuLoose'
)

muonSelectionTypeTask = cms.Task(
    muidTrackerMuonArbitrated
    ,muidAllArbitrated
    ,muidGlobalMuonPromptTight
    ,muidTMLastStationLoose
    ,muidTMLastStationTight
    ,muidTM2DCompatibilityLoose
    ,muidTM2DCompatibilityTight
    ,muidTMOneStationLoose
    ,muidTMOneStationTight
    ,muidTMLastStationOptimizedLowPtLoose
    ,muidTMLastStationOptimizedLowPtTight
    ,muidGMTkChiCompatibility
    ,muidGMStaChiCompatibility
    ,muidGMTkKinkTight
    ,muidTMLastStationAngLoose
    ,muidTMLastStationAngTight
    ,muidTMOneStationAngLoose
    ,muidTMOneStationAngTight
    ,muidRPCMuLoose)
muonSelectionTypeSequence = cms.Sequence(muonSelectionTypeTask)
