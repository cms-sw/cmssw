import FWCore.ParameterSet.Config as cms

dedxDiscrimProd = cms.EDProducer("DeDxEstimatorProducer",
    tracks                     = cms.InputTag("generalTracks"),
    trajectoryTrackAssociation = cms.InputTag("generalTracks"),

    estimator      = cms.string('productDiscrim'),

    Reccord            = cms.untracked.string("SiStripDeDxMip_3D_Rcd"),
    Formula            = cms.untracked.uint32(0),
    ProbabilityMode    = cms.untracked.string("Accumulation"),

    UseStrip       = cms.bool(True),
    UsePixel       = cms.bool(False),
    MeVperADCStrip = cms.double(3.61e-06*265),
    MeVperADCPixel = cms.double(3.61e-06),

    UseCalibration  = cms.bool(False),
    calibrationPath = cms.string(""),
    ShapeTest       = cms.bool(True),
)

dedxDiscrimBTag         = dedxDiscrimProd.clone()
dedxDiscrimBTag.estimator = cms.string('btagDiscrim')

dedxDiscrimSmi         = dedxDiscrimProd.clone()
dedxDiscrimSmi.estimator = cms.string('smirnovDiscrim')

dedxDiscrimASmi         = dedxDiscrimProd.clone()
dedxDiscrimASmi.estimator = cms.string('asmirnovDiscrim')

doAlldEdXDiscriminators = cms.Sequence(dedxDiscrimProd * dedxDiscrimBTag * dedxDiscrimSmi * dedxDiscrimASmi)
