import FWCore.ParameterSet.Config as cms

packedPFCandidates = cms.EDProducer("PATPackedCandidateProducer",
    inputCollection = cms.InputTag("particleFlow"),
    inputVertices = cms.InputTag("offlineSlimmedPrimaryVertices"),
    originalVertices = cms.InputTag("offlinePrimaryVertices"),
    originalTracks = cms.InputTag("generalTracks"),
    vertexAssociator = cms.InputTag("primaryVertexAssociation","original"),
    PuppiSrc = cms.InputTag("puppi"),
    PuppiNoLepSrc = cms.InputTag("puppiNoLep"),    
    secondaryVerticesForWhiteList = cms.VInputTag(
      cms.InputTag("inclusiveCandidateSecondaryVertices"),
      cms.InputTag("inclusiveCandidateSecondaryVerticesCvsL"),
      ),      
    minPtForTrackProperties = cms.double(0.95),
    covarianceVersion = cms.int32(1), #so far: 0 is Phase0, 1 is Phase1   
#    covariancePackingSchemas = cms.vint32(1,257,513,769,0)  # a cheaper schema in kb/ev 
    covariancePackingSchemas = cms.vint32(8,264,520,776,0)   # more accurate schema +0.6kb/ev   
)
