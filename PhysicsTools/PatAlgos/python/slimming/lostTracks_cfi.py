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
    minPt = cms.double(0.95),	
    minHits = cms.uint32(8),	
    minPixelHits = cms.uint32(1),  
    covarianceVersion = cms.int32(0), #so far: 0 is Phase0, 1 is Phase1   
    covarianceSchema = cms.int32(0), #old miniaod like
    qualsToAutoAccept = cms.vstring("highPurity"),
    minPtToStoreProps = cms.double(0.95)	
)
from Configuration.Eras.Modifier_phase1Pixel_cff import phase1Pixel
phase1Pixel.toModify(lostTracks, covarianceVersion =1 )

