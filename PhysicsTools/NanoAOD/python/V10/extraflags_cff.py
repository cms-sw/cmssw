import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.common_cff import *
from PhysicsTools.NanoAOD.nano_eras_cff import *

# Bad/clone muon filters - tagging mode to keep the event
badGlobalMuonTagger = cms.EDFilter("BadGlobalMuonTagger",
    muonPtCut = cms.double(20),
    muons = cms.InputTag("slimmedMuons"),
    selectClones = cms.bool(False),
    taggingMode = cms.bool(True),
    vtx = cms.InputTag("offlineSlimmedPrimaryVertices")
)

cloneGlobalMuonTagger = cms.EDFilter("BadGlobalMuonTagger",
    muonPtCut = cms.double(20),
    muons = cms.InputTag("slimmedMuons"),
    selectClones = cms.bool(True),
    taggingMode = cms.bool(True),
    vtx = cms.InputTag("offlineSlimmedPrimaryVertices")
)

BadPFMuonTagger = cms.EDFilter("BadParticleFilter",
    PFCandidates = cms.InputTag("particleFlow"),
    algo = cms.int32(14),
    filterType = cms.string('BadPFMuon'),
    innerTrackRelErr = cms.double(1),
    maxDR = cms.double(0.001),
    mightGet = cms.optional.untracked.vstring,
    minDzBestTrack = cms.double(-1),
    minMuonPt = cms.double(100),
    minMuonTrackRelErr = cms.double(2),
    minPtDiffRel = cms.double(0),
    muons = cms.InputTag("muons"),
    segmentCompatibility = cms.double(0.3),
    taggingMode = cms.bool(False),
    vtx = cms.InputTag("offlinePrimaryVertices")
)

# Bad charge hadron
BadChargedCandidateTagger = cms.EDFilter("BadParticleFilter",
    PFCandidates = cms.InputTag("packedPFCandidates"),
    algo = cms.int32(14),
    filterType = cms.string('BadChargedCandidate'),
    innerTrackRelErr = cms.double(1),
    maxDR = cms.double(1e-05),
    mightGet = cms.optional.untracked.vstring,
    minDzBestTrack = cms.double(-1),
    minMuonPt = cms.double(100),
    minMuonTrackRelErr = cms.double(2),
    minPtDiffRel = cms.double(1e-05),
    muons = cms.InputTag("slimmedMuons"),
    segmentCompatibility = cms.double(0.3),
    taggingMode = cms.bool(True),
    vtx = cms.InputTag("offlineSlimmedPrimaryVertices")
)

extraFlagsTable = cms.EDProducer("GlobalVariablesTableProducer",
    variables = cms.PSet(
        Flag_BadGlobalMuon = ExtVar(cms.InputTag("badGlobalMuonTagger:notBadEvent"), bool, doc = "Bad muon flag"),
        Flag_CloneGlobalMuon = ExtVar(cms.InputTag("cloneGlobalMuonTagger:notBadEvent"), bool, doc = "Clone muon flag"),
        Flag_BadPFMuonFilter = ExtVar(cms.InputTag("BadPFMuonTagger"), bool, doc = "Bad PF muon flag"),
        Flag_BadChargedCandidateFilter = ExtVar(cms.InputTag("BadChargedCandidateTagger"), bool, doc = "Bad charged hadron flag"),
    )
)

ecalBadCalibFilterNanoTagger = cms.EDFilter("EcalBadCalibFilter",
    EcalRecHitSource = cms.InputTag("reducedEcalRecHitsEE"),
    baddetEcal = cms.vuint32(
        872439604, 872422825, 872420274, 872423218, 872423215,
        872416066, 872435036, 872439336, 872420273, 872436907,
        872420147, 872439731, 872436657, 872420397, 872439732,
        872439339, 872439603, 872422436, 872439861, 872437051,
        872437052, 872420649, 872421950, 872437185, 872422564,
        872421566, 872421695, 872421955, 872421567, 872437184,
        872421951, 872421694, 872437056, 872437057, 872437313
    ),
    debug = cms.bool(False),
    ecalMinEt = cms.double(50.0),
    taggingMode = cms.bool(True)
)


# empty task as default
extraFlagsProducersTask = cms.Task()
extraFlagsTableTask = cms.Task()

## those need to be added only for some specific eras
extraFlagsProducersTask_80x = cms.Task(badGlobalMuonTagger,cloneGlobalMuonTagger,BadPFMuonTagger,BadChargedCandidateTagger)
extraFlagsProducersTask_102x = cms.Task(ecalBadCalibFilterNanoTagger)

