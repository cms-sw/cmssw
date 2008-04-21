import FWCore.ParameterSet.Config as cms

import copy
from RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilder_cfi import *
# PixelLayerPairsESProducer modified
myTTRHBuilderWithoutAngle4PixelPairs = copy.deepcopy(ttrhbwr)
pixelLayerPairsModifiedESProducer = cms.ESProducer("PixelLayerPairsESProducer",
    ComponentName = cms.string('PixelLayerPairsModified'),
    layerList = cms.vstring('BPix1+BPix2'),
    BPix = cms.PSet(
        useErrorsFromParam = cms.untracked.bool(True),
        hitErrorRPhi = cms.double(0.0027),
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4PixelPairs'),
        HitProducer = cms.string('pixelFreePrimHits'),
        hitErrorRZ = cms.double(0.006)
    ),
    FPix = cms.PSet(
        useErrorsFromParam = cms.untracked.bool(True),
        hitErrorRPhi = cms.double(0.0051),
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4PixelPairs'),
        HitProducer = cms.string('pixelFreePrimHits'),
        hitErrorRZ = cms.double(0.0036)
    )
)

# PixelLayerTripletsESProducer modified
pixelLayerTripletsModifiedESProducer = cms.ESProducer("PixelLayerTripletsESProducer",
    ComponentName = cms.string('PixelLayerTripletsModified'),
    layerList = cms.vstring('BPix1+BPix2+BPix3', 
        'BPix1+BPix2+FPix1_pos', 
        'BPix1+BPix2+FPix1_neg', 
        'BPix1+FPix1_pos+FPix2_pos', 
        'BPix1+FPix1_neg+FPix2_neg'),
    BPix = cms.PSet(
        useErrorsFromParam = cms.untracked.bool(True),
        hitErrorRPhi = cms.double(0.0027),
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4PixelTriplets'),
        HitProducer = cms.string('pixelFreeSecoHits'),
        hitErrorRZ = cms.double(0.006)
    ),
    FPix = cms.PSet(
        useErrorsFromParam = cms.untracked.bool(True),
        hitErrorRPhi = cms.double(0.0051),
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4PixelTriplets'),
        HitProducer = cms.string('pixelFreeSecoHits'),
        hitErrorRZ = cms.double(0.0036)
    )
)

pixelLayerTripletsPbPbESProducer = cms.ESProducer("PixelLayerTripletsESProducer",
    ComponentName = cms.string('PixelLayerTripletsPbPb'),
    layerList = cms.vstring('BPix1+BPix2+BPix3'),
    BPix = cms.PSet(
        useErrorsFromParam = cms.untracked.bool(True),
        hitErrorRPhi = cms.double(0.0027),
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4PixelTriplets'),
        HitProducer = cms.string('siPixelRecHits'),
        hitErrorRZ = cms.double(0.006)
    ),
    FPix = cms.PSet(
        useErrorsFromParam = cms.untracked.bool(True),
        hitErrorRPhi = cms.double(0.0051),
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4PixelTriplets'),
        HitProducer = cms.string('siPixelRecHits'),
        hitErrorRZ = cms.double(0.0036)
    )
)

myTTRHBuilderWithoutAngle4PixelPairs.StripCPE = 'Fake'
myTTRHBuilderWithoutAngle4PixelPairs.ComponentName = 'TTRHBuilderWithoutAngle4PixelPairs'

