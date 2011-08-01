import FWCore.ParameterSet.Config as cms

from RecoTracker.ConversionSeedGenerators.ConversionStep_iter2Clone_cff import *


conversionStepSixth   = cms.Sequence(fifthFilter*sixthClusters*sixthPixelRecHits*sixthStripRecHits*
                                     sixthTripletsA*sixthTripletsB*sixthTriplets*
                                     sixthTrackCandidates*sixthWithMaterialTracks*
                                     sixthStepLoose*sixthStepTight*sixthStep
                                     )

#conversionStep = cms.Sequence(conversionStepSixth)

