import FWCore.ParameterSet.Config as cms

dedxHitInfo = cms.EDProducer("DeDxHitInfoProducer",
    tracks             = cms.InputTag("generalTracks"),

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
    clusterShapeCache  = cms.InputTag("siPixelClusterShapeCache"),

    lowPtTracksPrescalePass = cms.uint32(100),   # prescale factor for low pt tracks above the dEdx cut
    lowPtTracksPrescaleFail = cms.uint32(2000), # prescale factor for low pt tracks below the dEdx cut
    lowPtTracksEstimatorParameters = cms.PSet( # generalized truncated average
        fraction = cms.double(-0.15), # negative = throw away the 15% with lowest charge
        exponent = cms.double(-2.0),
        truncate = cms.bool(True),
    ),
    lowPtTracksDeDxThreshold = cms.double(3.5), # threshold on tracks
    usePixelForPrescales = cms.bool(True)
)

import RecoTracker.DeDx.DeDxEstimatorProducer_cfi as _mod

dedxHarmonic2 = _mod.DeDxEstimatorProducer.clone(
    estimator      = 'generic',
    fraction       = 0.4,        #Used only if estimator='truncated'
    exponent       = -2.0,       #Used only if estimator='generic'

    Record            = "SiStripDeDxMip_3D_Rcd", #used only for discriminators : estimators='productDiscrim' or 'btagDiscrim' or 'smirnovDiscrim' or 'asmirnovDiscrim'
    ProbabilityMode    = "Accumulation",          #used only for discriminators : estimators='productDiscrim' or 'btagDiscrim' or 'smirnovDiscrim' or 'asmirnovDiscrim'
)

from Configuration.Eras.Modifier_fastSim_cff import fastSim

# explicit python dependency
import FastSimulation.SimplifiedGeometryPropagator.FastTrackDeDxProducer_cfi

# do this before defining dedxPixelHarmonic2 so it automatically comes out right
fastSim.toReplaceWith(dedxHarmonic2,
    FastSimulation.SimplifiedGeometryPropagator.FastTrackDeDxProducer_cfi.FastTrackDeDxProducer.clone(
        ShapeTest = False,
        simHit2RecHitMap = "fastMatchedTrackerRecHits:simHit2RecHitMap",
        simHits = "fastSimProducer:TrackerHits",
    )
)

dedxPixelHarmonic2 = dedxHarmonic2.clone(UseStrip = False, UsePixel = True)

dedxPixelAndStripHarmonic2T085 = dedxHarmonic2.clone(
        UseStrip = True, UsePixel = True,
        estimator = 'genericTruncated',
        fraction  = -0.15, # Drop the lowest 15% of hits
        exponent  = -2.0, # Harmonic02
)

dedxTruncated40 = dedxHarmonic2.clone(estimator = 'truncated')

dedxMedian = dedxHarmonic2.clone(estimator = 'median')

dedxUnbinned = dedxHarmonic2.clone(estimator = 'unbinnedFit')

dedxDiscrimProd =  dedxHarmonic2.clone(estimator = 'productDiscrim')

dedxDiscrimBTag = dedxHarmonic2.clone(estimator = 'btagDiscrim')

dedxDiscrimSmi  = dedxHarmonic2.clone(estimator = 'smirnovDiscrim')

dedxDiscrimASmi = dedxHarmonic2.clone(estimator = 'asmirnovDiscrim')

doAlldEdXEstimatorsTask = cms.Task(dedxTruncated40 , dedxHarmonic2 , dedxPixelHarmonic2 , dedxPixelAndStripHarmonic2T085 , dedxHitInfo)
doAlldEdXEstimators = cms.Sequence(doAlldEdXEstimatorsTask)

fastSim.toReplaceWith(doAlldEdXEstimatorsTask, cms.Task(dedxHarmonic2, dedxPixelHarmonic2))

# use only the strips for Run-3
from Configuration.Eras.Modifier_run3_common_cff import run3_common
run3_common.toModify(dedxHitInfo,
    lowPtTracksEstimatorParameters = dict(fraction = 0., exponent = -2.0,truncate = False),
    usePixelForPrescales = False
)

# dEdx for Run-3 UPC
from Configuration.Eras.Modifier_run3_upc_cff import run3_upc
run3_upc.toModify(dedxHitInfo, minTrackPt = 0)

from RecoTracker.DeDx.dedxHitCalibrator_cfi import dedxHitCalibrator as _dedxHitCalibrator
from SimGeneral.MixingModule.SiStripSimParameters_cfi import SiStripSimBlock as _SiStripSimBlock
from RecoLocalTracker.SiPixelClusterizer.SiPixelClusterizer_cfi import siPixelClusters as _siPixelClusters
dedxHitCalibrator = _dedxHitCalibrator.clone(
    MeVPerElectron = 1000*_SiStripSimBlock.GevPerElectron.value(),
    VCaltoElectronGain = _siPixelClusters.VCaltoElectronGain,
    VCaltoElectronGain_L1 = _siPixelClusters.VCaltoElectronGain_L1,
    VCaltoElectronOffset = _siPixelClusters.VCaltoElectronOffset,
    VCaltoElectronOffset_L1 = _siPixelClusters.VCaltoElectronOffset_L1
)

dedxAllLikelihood = _mod.DeDxEstimatorProducer.clone(
    UseStrip = True, UsePixel = True,
    estimator = 'likelihoodFit',
    UseDeDxHits = True,
    pixelDeDxHits = 'dedxHitCalibrator:PixelHits',
    stripDeDxHits = 'dedxHitCalibrator:StripHits'
)
dedxPixelLikelihood = dedxAllLikelihood.clone(UseStrip = False, UsePixel = True)
dedxStripLikelihood = dedxAllLikelihood.clone(UseStrip = True,  UsePixel = False)

from Configuration.Eras.Modifier_run3_egamma_2023_cff import run3_egamma_2023
(run3_upc & ~run3_egamma_2023).toReplaceWith(doAlldEdXEstimatorsTask, cms.Task(doAlldEdXEstimatorsTask.copy(), dedxHitCalibrator, dedxStripLikelihood, dedxPixelLikelihood, dedxAllLikelihood))
