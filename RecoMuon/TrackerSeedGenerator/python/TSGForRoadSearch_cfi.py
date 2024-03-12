import FWCore.ParameterSet.Config as cms

SeedGeneratorParameters = cms.PSet(
    propagatorCompatibleName = cms.string('SteppingHelixPropagatorAny'),
    #category: MuonRSSeedGeneratorAlgorithm
    #0 old code inside-out
    #1 new code inside-out
    #2 new code outside-out
    option = cms.uint32(0),
    ComponentName = cms.string('TSGForRoadSearch'),
    errorMatrixPset = cms.PSet(

    ),
    propagatorName = cms.string('SteppingHelixPropagatorAlong'),
    manySeeds = cms.bool(False),
    copyMuonRecHit = cms.bool(False),
    maxChi2 = cms.double(40.0)
)


# foo bar baz
# vWa1dscDZkVEu
# rRRyO6cLk0k0V
