import FWCore.ParameterSet.Config as cms

dedxDiscrimProd = cms.EDProducer("DeDxDiscriminatorProducer",
    tracks                     = cms.InputTag("generalTracks"),
    trajectoryTrackAssociation = cms.InputTag("generalTracks"),

    Reccord            = cms.untracked.string("SiStripDeDxMip_3D_Rcd"),
    Formula            = cms.untracked.uint32(0),
    ProbabilityMode    = cms.untracked.string("Accumulation"),

    UseStrip           = cms.bool(True),
    UsePixel           = cms.bool(False),
    MeVperADCStrip     = cms.double(3.61e-06*265),
    MeVperADCPixel     = cms.double(3.61e-06),

    UseCalibration  = cms.bool(False),
    calibrationPath = cms.string(""),
    ShapeTest       = cms.bool(True),
)

