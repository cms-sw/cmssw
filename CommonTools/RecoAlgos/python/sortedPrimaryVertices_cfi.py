import FWCore.ParameterSet.Config as cms
sortedPrimaryVertices = cms.EDProducer("RecoChargedRefCandidatePrimaryVertexSorter",
    sorting = cms.PSet(),
    assignment = cms.PSet(
    #cuts to assign primary tracks not used in PV fit based on dZ compatibility
    maxDzSigForPrimaryAssignment = cms.double(5.0), # in AND with next
    maxDzForPrimaryAssignment = cms.double(0.1), # in AND with prev
    maxDzErrorForPrimaryAssignment = cms.double(0.05), # in AND with prev, tracks with uncertainty above 500um cannot tell us which pv they come from
    maxDtSigForPrimaryAssignment = cms.double(4.0),
    # cuts used to recover b-tracks if they are closed to jet axis
    maxJetDeltaR = cms.double(0.5),
    minJetPt = cms.double(25),
    maxDistanceToJetAxis = cms.double(0.07), # std cut in b-tag is 700um
    maxDzForJetAxisAssigment = cms.double(0.1), # 1mm, because b-track IP is boost invariant
    maxDxyForJetAxisAssigment = cms.double(0.1), # 1mm, because b-track IP is boost invariant

    #cuts used to identify primary tracks compatible with beamspot
    maxDxySigForNotReconstructedPrimary = cms.double(2), #in AND with next
    maxDxyForNotReconstructedPrimary = cms.double(0.01), #in AND with prev
    useTiming = cms.bool(False),
    preferHighRanked = cms.bool(False),
    ),
  particles = cms.InputTag("trackRefsForJets"),
  trackTimeTag = cms.InputTag(""),
  trackTimeResoTag = cms.InputTag(""),
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

