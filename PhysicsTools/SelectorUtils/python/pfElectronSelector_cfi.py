import FWCore.ParameterSet.Config as cms

#Electron Selector
pfElectronSelector = cms.PSet(
    version = cms.string('SPRING11'),
    MVA = cms.double(0.4),
    MaxMissingHits = cms.int32(1),
    D0 = cms.double(0.02),
    PFIso = cms.double(0.15)
    )
