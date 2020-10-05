import FWCore.ParameterSet.Config as cms

pfImpactParameterTagInfos = cms.EDProducer("CandIPProducer",
    primaryVertex = cms.InputTag("offlinePrimaryVertices"),
    computeProbabilities = cms.bool(True),
    computeGhostTrack = cms.bool(True),
    ghostTrackPriorDeltaR = cms.double(0.03),
    minimumNumberOfPixelHits = cms.int32(1),
    minimumNumberOfHits = cms.int32(0),
    maximumTransverseImpactParameter = cms.double(0.2),
    minimumTransverseMomentum = cms.double(1.0),
    maximumChiSquared = cms.double(5.0),
    maximumLongitudinalImpactParameter = cms.double(17.0),
    jetDirectionUsingTracks = cms.bool(False),
    jetDirectionUsingGhostTrack = cms.bool(False),
    useTrackQuality = cms.bool(False),
    # this is candidate specific
    jets = cms.InputTag("ak4PFJetsCHS"),
    candidates = cms.InputTag("particleFlow"),
    maxDeltaR = cms.double(0.4)
)

from Configuration.Eras.Modifier_pp_on_AA_2018_cff import pp_on_AA_2018
from Configuration.Eras.Modifier_pp_on_PbPb_run3_cff import pp_on_PbPb_run3
(pp_on_AA_2018 | pp_on_PbPb_run3).toModify(pfImpactParameterTagInfos, jets = "akCs4PFJets")
