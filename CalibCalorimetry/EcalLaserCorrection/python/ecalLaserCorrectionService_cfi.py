import FWCore.ParameterSet.Config as cms

EcalLaserCorrectionService = cms.ESProducer("EcalLaserCorrectionService", 
                                        ### delta t is the safety margin before stopping
                                        ### the laser correction extrapolation (this is in seconds)
                                        deltat_safety = cms.untracked.int32( 0  ) )



