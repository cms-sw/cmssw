
import FWCore.ParameterSet.Config as cms

# creating quadruplet SeedingLayerSets for the merger;
PixelSeedMergerQuadruplets = cms.PSet(
  appendToDataLabel = cms.string( "" ),

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
    TTRHBuilder = cms.string( "TTRHBuilderPixelOnly" ),
    HitProducer = cms.string( "hltSiPixelRecHits" ),
  ),
  FPix = cms.PSet( 
    TTRHBuilder = cms.string( "TTRHBuilderPixelOnly" ),
    HitProducer = cms.string( "hltSiPixelRecHits" ),
  ),
  TEC = cms.PSet(  )
)

