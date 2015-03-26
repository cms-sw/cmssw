import FWCore.ParameterSet.Config as cms

from RecoBTag.ImpactParameter.variableJTA_cff import *

# jetBProbability btag computer
candidateJetBProbabilityComputer = cms.ESProducer("CandidateJetBProbabilityESProducer",
    variableJTAPars,
    impactParameterType = cms.int32(0), ## 0 = 3D, 1 = 2D
    deltaR = cms.double(-1.0), ## use cut from JTA
    maximumDistanceToJetAxis = cms.double(0.07),
    trackIpSign = cms.int32(1), ## 0 = use both, 1 = positive only, -1 = negative only
    minimumProbability = cms.double(0.005),
    numberOfBTracks = cms.uint32(4),
    maximumDecayLength = cms.double(5.0),
    trackQualityClass = cms.string("any"),
    useVariableJTA = cms.bool(False)
)
