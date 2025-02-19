import FWCore.ParameterSet.Config as cms

from RecoMuon.TrackingTools.MuonErrorMatrixValues_cff import *

SeedGeneratorParameters = cms.PSet(
    MuonErrorMatrixValues,
    propagatorCompatibleName = cms.string('SteppingHelixPropagatorOpposite'),
    #category: MuonRSSeedGeneratorAlgorithm
    #0 old code inside-out
    #1 new code inside-out
    #2 new code outside-out
    #3 old code outside-in
    option = cms.uint32(3),
    ComponentName = cms.string('TSGForRoadSearch'),
    propagatorName = cms.string('SteppingHelixPropagatorAlong'),
    manySeeds = cms.bool(False),
    copyMuonRecHit = cms.bool(False),
    maxChi2 = cms.double(40.0)
)


