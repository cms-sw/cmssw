import FWCore.ParameterSet.Config as cms

from GEMCode.GEMValidation.simTrackMatching_cfi import SimTrackMatching

stm = SimTrackMatching.clone()

GEMCSCAnalyzer = cms.EDAnalyzer("GEMCSCAnalyzer",
    verbose = cms.untracked.int32(0),
    ntupleTrackChamberDelta = cms.untracked.bool(True),
    ntupleTrackEff = cms.untracked.bool(True),
    stationsToUse = cms.vint32(1,2,3,4),
    simTrackMatching = stm
)
