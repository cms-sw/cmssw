import FWCore.ParameterSet.Config as cms

FP420Reco = cms.EDProducer("ReconstructerFP420",#
    ROUList = cms.vstring('FP420Track'),        #
    genReadoutName = cms.string('source'),      # HepMC source to be processed
    VerbosityLevel = cms.untracked.int32(0),    #
    Beam1 = cms.string('SimTransport/HectorProducer/data/LHCB1IR5_7TeV.tfs'),  #
    Beam2 = cms.string('SimTransport/HectorProducer/data/LHCB2IR5_7TeV.tfs'),  #
    NumberFP420Detectors = cms.int32(3),              #  =3 means 2 Trackers: +FP420 and -FP420; =0 -> no FP420 at all 
    RP420f = cms.double(420000.0),                    # distance of transport in clockwise dir. for FP420
    RP420b = cms.double(420000.0),                    # distance of transport in anti-clockwise dir. for FP420
    BeamLineLengthFP420 = cms.double(430.0),          # length of beam line for FP420: important for aperture checks
    zreffFP420 = cms.double(8000.0),                  # arm for FP420 clockwise detectors
    zrefbFP420 = cms.double(8000.0),                  # arm for FP420 anti-clockwise detectors
    VtxFlagGenRecFP420  = cms.int32(0),               # =0 vtx=0; =1 vtx=GEN/REC/INI
    VtxFP420X = cms.double(0.04),                     # GEN/REC/INI Xvertex in mm for FP420
    VtxFP420Y = cms.double(0.01),                     # GEN/REC/INI Yvertex in mm for FP420
    VtxFP420Z = cms.double(2.0),                      # GEN/REC/INI Zvertex in mm for FP420
    NumberHPS240Detectors = cms.int32(3),               #  =3 means 2 Trackers: +HPS240 and -HPS240; =0 -> no HPS240 at all 
    RP240f = cms.double(240000.0),                      # distance of transport in clockwise dir. for HPS240
    RP240b = cms.double(240000.0),                      # distance of transport in anti-clockwise dir. for HPS240
    BeamLineLengthHPS240 = cms.double(250.0),           # length of beam line for HPS240: important for aperture checks
    zreffHPS240 = cms.double(8000.0),                   # arm for HPS240 clockwise detectors
    zrefbHPS240 = cms.double(8000.0),                   # arm for HPS240 anti-clockwise detectors
    VtxFlagGenRecHPS240  = cms.int32(0),                # =0 vtx=0; =1 vtx=GEN/REC/INI
    VtxHPS240X = cms.double(0.04),                      # GEN/REC/INI Xvertex in mm for HPS240
    VtxHPS240Y = cms.double(0.01),                      # GEN/REC/INI Yvertex in mm for HPS240
    VtxHPS240Z = cms.double(2.0)                        # GEN/REC/INI Zvertex in mm for HPS240
)
