import FWCore.ParameterSet.Config as cms

dedxDiscrimProd = cms.EDProducer("DeDxDiscriminatorProducer",
    tracks                     = cms.InputTag("generalTracks"),
    trajectoryTrackAssociation = cms.InputTag("generalTracks"),

    Reccord            = cms.untracked.string("SiStripDeDxMip_3D_Rcd"),
    Formula            = cms.untracked.uint32(0),

    UseStrip           = cms.bool(True),
    UsePixel           = cms.bool(False),
    MeVperADCStrip     = cms.double(3.61e-06*250),
    MeVperADCPixel     = cms.double(3.61e-06),

    minTrackMomentum   = cms.untracked.double(3.0),

    UseCalibration  = cms.bool(False),
    calibrationPath = cms.string(""),
)
