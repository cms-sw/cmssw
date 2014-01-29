import FWCore.ParameterSet.Config as cms

from GEMCode.GEMValidation.simTrackMatching_cfi import SimTrackMatching

stm = SimTrackMatching.clone()

GEMCSCAnalyzer = cms.EDAnalyzer("GEMCSCAnalyzer",
    verbose = cms.untracked.int32(0),
    simInputLabel = cms.untracked.string("g4SimHits"),
    minPt = cms.untracked.double(4.5),
    minEta = cms.untracked.double(1.5),
    maxEta = cms.untracked.double(2.5),
    ntupleTrackChamberDelta = cms.untracked.bool(True),
    ntupleTrackEff = cms.untracked.bool(True),
    stationsToUse = cms.vint32(1,),
    simTrackMatching = stm
)
