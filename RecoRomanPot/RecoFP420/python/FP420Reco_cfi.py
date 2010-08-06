import FWCore.ParameterSet.Config as cms

FP420Reco = cms.EDFilter("ReconstructerFP420",
    #--------------------------------
    #--------------------------------
    ROUList = cms.vstring('FP420Track'),
    NumberFP420Detectors = cms.int32(3),
    Beam2 = cms.string('SimTransport/HectorProducer/data/LHCB2IR5_v6.500.tfs'),
    #-------------------------------------
    Beam1 = cms.string('SimTransport/HectorProducer/data/LHCB1IR5_v6.500.tfs'),
    RP420f = cms.double(420000.0),
    #--------------------------------
    #--------------------------------
    BeamLineLength = cms.double(430.0),
    RP420b = cms.double(420000.0),
    #--------------------------------
    #--------------------------------
    VerbosityLevel = cms.untracked.int32(0),
    zrefb = cms.double(8000.0),
    zreff = cms.double(8000.0)
)


