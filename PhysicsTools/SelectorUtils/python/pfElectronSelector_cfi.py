import FWCore.ParameterSet.Config as cms

#Electron Selector
pfElectronSelector = cms.PSet(
    version = cms.string('TOPPAG'),
    Fiducial = cms.bool(True),
    MaxMissingHits = cms.int32(1),
    D0 = cms.double(0.02),
    ConversionRejection = cms.bool(True),
    PFIso = cms.double(0.1),
    MVA = cms.double(0.5),
    cutsToIgnore = cms.vstring()
    )
