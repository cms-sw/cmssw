import FWCore.ParameterSet.Config as cms


pfPileUp = cms.EDProducer(
    "PFPileUp",
    PFCandidates = cms.InputTag("particleFlowTmpPtrs"),
    Vertices = cms.InputTag("offlinePrimaryVertices"),
    # pile-up identification now enabled by default. To be studied for jets
    Enable = cms.bool(True),
    verbose = cms.untracked.bool(False),
    checkClosestZVertex = cms.bool(True),
    usePrimaryVertexAssignment = cms.bool(False),
    assignmentQualityForPrimary = cms.int32(2),
    Jets = cms.InputTag("ak4PFJetsTmp"),
    assignment = cms.PSet(
      #cuts to assign primary tracks not used in PV fit based on dZ compatibility
      maxDzSigForPrimaryAssignment = cms.double(5.0), # in OR with next
      maxDzForPrimaryAssignment = cms.double(0.03), # in OR with prev
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
      ),
    )
 
 
from Configuration.Eras.Modifier_phase2_common_cff import phase2_common
phase2_common.toModify(
    pfPileUp,
    usePrimaryVertexAssignment = True
)

from Configuration.Eras.Modifier_phase2_timing_cff import phase2_timing
phase2_timing.toModify(
    pfPileUp,
    assignment=dict(useTiming=True)
)
