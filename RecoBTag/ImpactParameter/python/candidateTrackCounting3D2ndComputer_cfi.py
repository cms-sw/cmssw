import FWCore.ParameterSet.Config as cms

from RecoBTag.ImpactParameter.variableJTA_cff import *

# trackCounting3D2nd btag computer
candidateTrackCounting3D2ndComputer = cms.ESProducer("CandidateTrackCountingESProducer",
    variableJTAPars,
    minimumImpactParameter = cms.double(-1),
    useSignedImpactParameterSig = cms.bool(True),
    impactParameterType = cms.int32(0), ## 0 = 3D, 1 = 2D
    maximumDistanceToJetAxis = cms.double(0.07),
    deltaR = cms.double(-1.0), ## use cut from JTA
    maximumDecayLength = cms.double(5.0),
    nthTrack = cms.int32(2),
    trackQualityClass = cms.string("any"),
    useVariableJTA = cms.bool(False)
)
