import FWCore.ParameterSet.Config as cms

from RecoTracker.ConversionSeedGenerators.ConversionStep_cff import *


conversionStepSixth   = cms.Sequence(fifthFilter*sixthClusters*sixthPixelRecHits*sixthStripRecHits*
                                     sixthTriplets*
                                     sixthSeedsPositive*sixthSeedsNegative*
                                     sixthSeeds*
                                     sixthTrackCandidates*sixthWithMaterialTracks*
                                     sixthStepLoose*sixthStepTight*sixthStep
                                     )

convStepSeventhSeeds = cms.Sequence(sixthFilter*seventhClusters*seventhPixelRecHits*seventhStripRecHits*
                                    seventhPLSeeds*
                                    seventhSeedsPositive*seventhSeedsNegative*
                                    seventhSeeds
                                    )

convStepSeventhTks = cms.Sequence (seventhTrackCandidates*seventhStepTracks*
                                   seventhStepLoose*seventhStepTight*seventhStep
                                   )

conversionStepSeventh = cms.Sequence( convStepSeventhSeeds * convStepSeventhTks )

conversionStep = cms.Sequence(conversionStepSixth * conversionStepSeventh)

