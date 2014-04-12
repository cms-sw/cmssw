import FWCore.ParameterSet.Config as cms

FP420Reco = cms.EDProducer("ReconstructerFP420",
    ROUList = cms.vstring('FP420Track'),
    VerbosityLevel = cms.untracked.int32(0),
    NumberFP420Detectors = cms.int32(3),
    RP420f = cms.double(420000.0),
    RP420b = cms.double(420000.0),
    zreff = cms.double(8000.0),
    zrefb = cms.double(8000.0),
    Beam2 = cms.string('SimTransport/HectorProducer/data/LHCB2IR5_v6.500.tfs'),
    Beam1 = cms.string('SimTransport/HectorProducer/data/LHCB1IR5_v6.500.tfs'),
    BeamLineLength = cms.double(430.0),
    VtxFlagGenRec  = cms.int32(0),
    genReadoutName = cms.string('source')
)
# VtxFlagGenRec:                =0 vtx=0;                =1 vtx=GEN;                    =2 vtx=REC


