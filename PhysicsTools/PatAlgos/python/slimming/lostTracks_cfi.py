import FWCore.ParameterSet.Config as cms

lostTracks = cms.EDProducer("PATLostTracks",
    inputCandidates = cms.InputTag("particleFlow"),
    packedPFCandidates	= cms.InputTag("packedPFCandidates"),
    inputTracks = cms.InputTag("generalTracks"),
    secondaryVertices = cms.InputTag("inclusiveSecondaryVertices"),
    kshorts=cms.InputTag("generalV0Candidates","Kshort"),
    lambdas=cms.InputTag("generalV0Candidates","Lambda"),
    primaryVertices = cms.InputTag("offlineSlimmedPrimaryVertices"),
    originalVertices = cms.InputTag("offlinePrimaryVertices"),
    muons = cms.InputTag("muons"),
    minPt = cms.double(0.95),	
    minHits = cms.uint32(8),	
    minPixelHits = cms.uint32(1),  
    covarianceVersion = cms.int32(0), #so far: 0 is Phase0, 1 is Phase1   
    covarianceSchema = cms.int32(0), #old miniaod like
    qualsToAutoAccept = cms.vstring("highPurity"),
    minPtToStoreProps = cms.double(0.95),
    passThroughCut = cms.string("0"),
    allowMuonId = cms.bool(False)
)
from Configuration.Eras.Modifier_phase1Pixel_cff import phase1Pixel
phase1Pixel.toModify(lostTracks, covarianceVersion =1 )

from Configuration.Eras.Modifier_run2_miniAOD_devel_cff import run2_miniAOD_devel
run2_miniAOD_devel.toModify(lostTracks, passThroughCut="pt>2", allowMuonId=True)

