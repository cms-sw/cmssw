import FWCore.ParameterSet.Config as cms

conversionTrackProducer = cms.EDProducer("ConversionTrackProducer",
                                         #input collection of tracks or gsf tracks
                                         TrackProducer = cms.string(''),
                                         #control whether to get ref to Trajectory (only available in reco job)
                                         useTrajectory = cms.bool(True),
                                         #control which flags are set in output collection
                                         setTrackerOnly = cms.bool(False),
                                         setArbitratedEcalSeeded = cms.bool(False),
                                         setArbitratedMerged = cms.bool(True),
                                         setArbitratedMergedEcalGeneral = cms.bool(False),
                                         beamSpotInputTag   = cms.InputTag("offlineBeamSpot"),
                                         filterOnConvTrackHyp = cms.bool(True),
                                         minConvRadius = cms.double(2.0) #cm
                                         )
