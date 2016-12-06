import FWCore.ParameterSet.Config as cms

packedPFCandidates = cms.EDProducer("PATPackedCandidateProducer",
    inputCollection = cms.InputTag("particleFlow"),
    inputVertices = cms.InputTag("offlineSlimmedPrimaryVertices"),
    originalVertices = cms.InputTag("offlinePrimaryVertices"),
    originalTracks = cms.InputTag("generalTracks"),
    vertexAssociator = cms.InputTag("primaryVertexAssociation","original"),
    PuppiSrc = cms.InputTag("puppi"),
    PuppiNoLepSrc = cms.InputTag("puppiNoLep"),    
    chargedHadronIsolation = cms.InputTag("chargedHadronPFTrackIsolation"),
    minPtForChargedHadronProperties = cms.double(3.0),
    secondaryVerticesForWhiteList = cms.VInputTag(
      cms.InputTag("inclusiveCandidateSecondaryVertices"),
      cms.InputTag("inclusiveCandidateSecondaryVerticesCvsL"),
      cms.InputTag("generalV0Candidates","Kshort"),
      cms.InputTag("generalV0Candidates","Lambda"),
      ),      
    minPtForTrackProperties = cms.double(0.95),
    covarianceVersion = cms.int32(0), #so far: 0 is Phase0, 1 is Phase1   
#    covariancePackingSchemas = cms.vint32(1,257,513,769,0),  # a cheaper schema in kb/ev 
    covariancePackingSchemas = cms.vint32(8,264,520,776,0),   # more accurate schema +0.6kb/ev   
    storeTiming = cms.bool(False)
)

from Configuration.Eras.Modifier_phase1Pixel_cff import phase1Pixel
phase1Pixel.toModify(packedPFCandidates, covarianceVersion =1 )

from Configuration.Eras.Modifier_run2_miniAOD_80XLegacy_cff import run2_miniAOD_80XLegacy
run2_miniAOD_80XLegacy.toModify(packedPFCandidates, chargedHadronIsolation = "" )

from Configuration.Eras.Modifier_phase2_timing_cff import phase2_timing
phase2_timing.toModify(packedPFCandidates, storeTiming = cms.bool(True))

