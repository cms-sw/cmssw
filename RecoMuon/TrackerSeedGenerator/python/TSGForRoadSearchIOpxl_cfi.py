import FWCore.ParameterSet.Config as cms

from RecoMuon.TrackingTools.MuonErrorMatrixValues_cff import *

SeedGeneratorParameters = cms.PSet(
    propagatorCompatibleName = cms.string('SteppingHelixPropagatorAny'),
    #category: MuonRSSeedGeneratorAlgorithm
    #0 old code inside-out
    #1 new code inside-out
    #2 new code outside-out
    #3 old code outside-in
    #4 old code inside-out from pixel
    option = cms.uint32(4),
    ComponentName = cms.string('TSGForRoadSearch'),
    errorMatrixPset = cms.PSet(
    MuonErrorMatrixValues,
    action = cms.string('use'),
    atIP = cms.bool(True)
    ),
    propagatorName = cms.string('SteppingHelixPropagatorAlong'),
    manySeeds = cms.bool(False),
    copyMuonRecHit = cms.bool(False),
    maxChi2 = cms.double(40.0)
)


