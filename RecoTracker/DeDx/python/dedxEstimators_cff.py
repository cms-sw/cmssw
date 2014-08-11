import FWCore.ParameterSet.Config as cms

dedxHarmonic2 = cms.EDProducer("DeDxEstimatorProducer",
    tracks                     = cms.InputTag("generalTracks"),
    trajectoryTrackAssociation = cms.InputTag("generalTracks"),
 
    estimator      = cms.string('generic'),
    fraction       = cms.double(0.4),        #Used only if estimator='truncated'
    exponent       = cms.double(-2.0),       #Used only if estimator='generic'
 
    UseStrip       = cms.bool(True),
    UsePixel       = cms.bool(False),
    UseTrajectory  = cms.untracked.bool(False),
    MeVperADCStrip = cms.double(3.61e-06*265),
    MeVperADCPixel = cms.double(3.61e-06),

    UseCalibration  = cms.bool(False),
    calibrationPath = cms.string(""),
    ShapeTest       = cms.bool(True),

    Reccord            = cms.untracked.string("SiStripDeDxMip_3D_Rcd"), #used only for discriminators : estimators='productDiscrim' or 'btagDiscrim' or 'smirnovDiscrim' or 'asmirnovDiscrim'
    ProbabilityMode    = cms.untracked.string("Accumulation"),          #used only for discriminators : estimators='productDiscrim' or 'btagDiscrim' or 'smirnovDiscrim' or 'asmirnovDiscrim'
)

dedxTruncated40 = dedxHarmonic2.clone()
dedxTruncated40.estimator =  cms.string('truncated')

dedxMedian  = dedxHarmonic2.clone()
dedxMedian.estimator =  cms.string('median')

dedxUnbinned = dedxHarmonic2.clone()
dedxUnbinned.estimator =  cms.string('unbinnedFit')

dedxDiscrimProd =  dedxHarmonic2.clone()
dedxDiscrimProd.estimator = cms.string('productDiscrim')

dedxDiscrimBTag         = dedxHarmonic2.clone()
dedxDiscrimBTag.estimator = cms.string('btagDiscrim')

dedxDiscrimSmi         = dedxHarmonic2.clone()
dedxDiscrimSmi.estimator = cms.string('smirnovDiscrim')

dedxDiscrimASmi         = dedxHarmonic2.clone()
dedxDiscrimASmi.estimator = cms.string('asmirnovDiscrim')

doAlldEdXEstimators = cms.Sequence(dedxTruncated40 + dedxHarmonic2 + dedxDiscrimASmi)
