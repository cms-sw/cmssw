import FWCore.ParameterSet.Config as cms
sortedPrimaryVertices = cms.EDProducer("RecoChargedRefCandidatePrimaryVertexSorter",
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
  particles = cms.InputTag("trackRefsForJets"),
  vertices= cms.InputTag("offlinePrimaryVertices"),
#  Jets= cms.InputTag("ak4PFJets"),
  jets= cms.InputTag("ak4CaloJetsForTrk"),
  qualityForPrimary = cms.int32(3),
  usePVMET = cms.bool(True),
  produceAssociationToOriginalVertices=  cms.bool(False),
  produceSortedVertices = cms.bool(True),
  producePileUpCollection  = cms.bool(False),
  produceNoPileUpCollection = cms.bool(False),

)

