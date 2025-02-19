import FWCore.ParameterSet.Config as cms

SeedGeneratorParameters = cms.PSet(
    propagatorCompatibleName = cms.string('SteppingHelixPropagatorAny'),
    #category: MuonRSSeedGeneratorAlgorithm
    #0 old code inside-out
    #1 new code inside-out
    #2 new code outside-out
    #3 old code outside-in
    option = cms.uint32(0),
    ComponentName = cms.string('TSGForRoadSearch'),
    errorMatrixPset = cms.PSet(

    ),
    propagatorName = cms.string('SteppingHelixPropagatorAlong'),
    manySeeds = cms.bool(False),
    copyMuonRecHit = cms.bool(False),
    maxChi2 = cms.double(40.0)
)


