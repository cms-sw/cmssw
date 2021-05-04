import FWCore.ParameterSet.Config as cms

EcalLaserCorrectionService = cms.ESProducer("EcalLaserCorrectionService", 
                                        ### delta t is the safety margin before stopping
                                        ### the laser correction extrapolation (this is in seconds)
                                        maxExtrapolationTimeInSec = cms.uint32( 0 ) )



