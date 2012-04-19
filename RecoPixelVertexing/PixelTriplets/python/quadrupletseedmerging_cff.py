
import FWCore.ParameterSet.Config as cms

# creating quadruplet SeedingLayerSets for the merger;
pixelseedmergerlayers = cms.ESProducer( "SeedingLayersESProducer",
  appendToDataLabel = cms.string( "" ),
  ComponentName = cms.string( "PixelSeedMergerQuadruplets" ),

  # this simply describes all possible layer/disk combinations
  # on which quadruplets are expected to appear
  layerList = cms.vstring(
    ## straightforward combinations:
    'BPix1+BPix2+BPix3+BPix4',
    'BPix1+BPix2+BPix3+FPix1_pos',
    'BPix1+BPix2+BPix3+FPix1_neg',
    'BPix1+BPix2+FPix1_pos+FPix2_pos',
    'BPix1+BPix2+FPix1_neg+FPix2_neg',
    'BPix1+FPix1_pos+FPix2_pos+FPix3_pos',
    'BPix1+FPix1_neg+FPix2_neg+FPix3_neg'
#    ## "gap" combinations:
#    'BPix2+FPix1_pos+FPix2_pos+FPix3_pos',
#    'BPix2+FPix1_neg+FPix2_neg+FPix3_neg',
#    'BPix1+BPix2+FPix2_pos+FPix3_pos',
#    'BPix1+BPix2+FPix2_neg+FPix3_neg',
#    'BPix1+BPix2+FPix1_pos+FPix3_pos',
#    'BPix1+BPix2+FPix1_neg+FPix3_neg'
  ),

  BPix = cms.PSet( 
    useErrorsFromParam = cms.bool( True ),
    hitErrorRPhi = cms.double( 0.0027 ),
    TTRHBuilder = cms.string( "TTRHBuilderPixelOnly" ),
    HitProducer = cms.string( "hltSiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.0060 )
  ),
  FPix = cms.PSet( 
    useErrorsFromParam = cms.bool( True ),
    hitErrorRPhi = cms.double( 0.0051 ),
    TTRHBuilder = cms.string( "TTRHBuilderPixelOnly" ),
    HitProducer = cms.string( "hltSiPixelRecHits" ),
    hitErrorRZ = cms.double( 0.0036 )
  ),
  TEC = cms.PSet(  )
)

