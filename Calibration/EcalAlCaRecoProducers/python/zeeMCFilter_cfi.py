import FWCore.ParameterSet.Config as cms

#zeeMCFilter.cfi ############
zeeMCFilter = cms.EDFilter("MCZll",
    zMassMax = cms.untracked.double(999999.0),
    zMassMin = cms.untracked.double(0.0),
    filter = cms.untracked.bool(True),
    leptonPtMax = cms.untracked.double(999999.0),
    leptonEtaMax = cms.untracked.double(999999.0),
    leptonEtaMin = cms.untracked.double(0.0),
    leptonPtMin = cms.untracked.double(0.0)
)


