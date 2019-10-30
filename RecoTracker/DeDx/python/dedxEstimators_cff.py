import FWCore.ParameterSet.Config as cms

dedxHitInfo = cms.EDProducer("DeDxHitInfoProducer",
    tracks                     = cms.InputTag("generalTracks"),

    minTrackHits       = cms.uint32(0),
    minTrackPt         = cms.double(10),
    minTrackPtPrescale = cms.double(0.5), # minimal pT for prescaled low pT tracks
    maxTrackEta        = cms.double(5.0),

    useStrip           = cms.bool(True),
    usePixel           = cms.bool(True),
    MeVperADCStrip     = cms.double(3.61e-06*265),
    MeVperADCPixel     = cms.double(3.61e-06),

    useCalibration     = cms.bool(False),
    calibrationPath    = cms.string("file:Gains.root"),
    shapeTest          = cms.bool(True),

    lowPtTracksPrescalePass = cms.uint32(100),   # prescale factor for low pt tracks above the dEdx cut
    lowPtTracksPrescaleFail = cms.uint32(2000), # prescale factor for low pt tracks below the dEdx cut
    lowPtTracksEstimatorParameters = cms.PSet( # generalized truncated average
        fraction = cms.double(-0.15), # negative = throw away the 15% with lowest charge
        exponent = cms.double(-2.0),
    ),
    lowPtTracksDeDxThreshold = cms.double(3.5), # threshold on tracks
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

from Configuration.Eras.Modifier_fastSim_cff import fastSim

# explicit python dependency
import FastSimulation.SimplifiedGeometryPropagator.FastTrackDeDxProducer_cfi

# do this before defining dedxPixelHarmonic2 so it automatically comes out right
fastSim.toReplaceWith(dedxHarmonic2,
    FastSimulation.SimplifiedGeometryPropagator.FastTrackDeDxProducer_cfi.FastTrackDeDxProducer.clone(
        ShapeTest = False,
        simHit2RecHitMap = cms.InputTag("fastMatchedTrackerRecHits","simHit2RecHitMap"),
        simHits = cms.InputTag("fastSimProducer","TrackerHits"),
    )
)

dedxPixelHarmonic2 = dedxHarmonic2.clone(UseStrip = False, UsePixel = True)

dedxPixelAndStripHarmonic2T085 = dedxHarmonic2.clone(
        UseStrip = True, UsePixel = True,
        estimator = 'genericTruncated',
        fraction  = -0.15, # Drop the lowest 15% of hits
        exponent  = -2.0, # Harmonic02
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

doAlldEdXEstimatorsTask = cms.Task(dedxTruncated40 , dedxHarmonic2 , dedxPixelHarmonic2 , dedxPixelAndStripHarmonic2T085 , dedxHitInfo)
doAlldEdXEstimators = cms.Sequence(doAlldEdXEstimatorsTask)

fastSim.toReplaceWith(doAlldEdXEstimatorsTask, cms.Task(dedxHarmonic2, dedxPixelHarmonic2))
