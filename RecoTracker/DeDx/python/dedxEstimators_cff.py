import FWCore.ParameterSet.Config as cms

dedxHitInfo = cms.EDProducer("DeDxHitInfoProducer",
    tracks                     = cms.InputTag("generalTracks"),

    minTrackHits       = cms.uint32(0),
    minTrackPt         = cms.double(10),
    maxTrackEta        = cms.double(5.0),

    useStrip           = cms.bool(True),
    usePixel           = cms.bool(True),
    MeVperADCStrip     = cms.double(3.61e-06*265),
    MeVperADCPixel     = cms.double(3.61e-06),

    useCalibration     = cms.bool(False),
    calibrationPath    = cms.string("file:Gains.root"),
    shapeTest          = cms.bool(True),
)

dedxHarmonic2 = cms.EDProducer("DeDxEstimatorProducer",
    tracks                     = cms.InputTag("generalTracks"),
 
    estimator      = cms.string('generic'),
    fraction       = cms.double(0.4),        #Used only if estimator='truncated'
    exponent       = cms.double(-2.0),       #Used only if estimator='generic'
 
    UseStrip       = cms.bool(True),
    UsePixel       = cms.bool(False),
    ShapeTest      = cms.bool(True),
    MeVperADCStrip = cms.double(3.61e-06*265),
    MeVperADCPixel = cms.double(3.61e-06),

    Reccord            = cms.string("SiStripDeDxMip_3D_Rcd"), #used only for discriminators : estimators='productDiscrim' or 'btagDiscrim' or 'smirnovDiscrim' or 'asmirnovDiscrim'
    ProbabilityMode    = cms.string("Accumulation"),          #used only for discriminators : estimators='productDiscrim' or 'btagDiscrim' or 'smirnovDiscrim' or 'asmirnovDiscrim'

    UseCalibration  = cms.bool(False),
    calibrationPath = cms.string(""),
)

dedxPixelHarmonic2 = dedxHarmonic2.clone(UseStrip = False, UsePixel = True)

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

doAlldEdXEstimators = cms.Sequence(dedxTruncated40 + dedxHarmonic2 + dedxPixelHarmonic2 + dedxHitInfo)
