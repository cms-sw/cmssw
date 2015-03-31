import FWCore.ParameterSet.Config as cms
primaryVertexAssociation = cms.EDProducer("PFCandidatePrimaryVertexSorter",
    sorting = cms.PSet(),
    assignment = cms.PSet(
    #cuts to assign primary tracks not used in PV fit based on dZ compatibility
    maxDzSigForPrimaryAssignment = cms.double(5.0), # in OR with next
    maxDzForPrimaryAssignment = cms.double(0.03), # in OR with prev

    # cuts used to recover b-tracks if they are closed to jet axis
    maxJetDeltaR = cms.double(0.5),
    minJetPt = cms.double(25),
    maxDistanceToJetAxis = cms.double(0.07), # std cut in b-tag is 700um
    maxDzForJetAxisAssigment = cms.double(0.1), # 1mm, because b-track IP is boost invariant
    maxDxyForJetAxisAssigment = cms.double(0.1), # 1mm, because b-track IP is boost invariant

    #cuts used to identify primary tracks compatible with beamspot
    maxDxySigForNotReconstructedPrimary = cms.double(2), #in AND with next
    maxDxyForNotReconstructedPrimary = cms.double(0.01), #in AND with prev
    ),
  particles = cms.InputTag("particleFlow"),
  vertices= cms.InputTag("offlinePrimaryVertices"),
  jets= cms.InputTag("ak4PFJets"),
  qualityForPrimary = cms.int32(2),
  usePVMET = cms.bool(True),
  produceAssociationToOriginalVertices = cms.bool(True),
  produceSortedVertices = cms.bool(False),
  producePileUpCollection  = cms.bool(False),
  produceNoPileUpCollection = cms.bool(False),

)

