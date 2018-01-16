import FWCore.ParameterSet.Config as cms
materialDumperAnalyzer = DQMStep1Module('TrackingRecoMaterialAnalyser',
                                        folder = cms.string('Tracking/RecoMaterial/'),
                                        tracks = cms.InputTag("generalTracks"),
                                        beamspot = cms.InputTag("offlineBeamSpot"),
                                        usePV = cms.bool(False),
                                        vertices = cms.InputTag("offlinePrimaryVertices"),
                                        DoPredictionsOnly = cms.bool(False),
                                        Fitter = cms.string('KFFitterForRefitInsideOut'),
                                        TrackerRecHitBuilder = cms.string('WithAngleAndTemplate'),
                                        Smoother = cms.string('KFSmootherForRefitInsideOut'),
                                        MuonRecHitBuilder = cms.string('MuonRecHitBuilder'),
                                        RefitDirection = cms.string('alongMomentum'),
                                        RefitRPCHits = cms.bool(True),
                                        Propagator = cms.string('SmartPropagatorAnyRKOpposite'),
                                        #Propagators
                                        PropagatorAlong = cms.string("RungeKuttaTrackerPropagator"),
                                        PropagatorOpposite = cms.string("RungeKuttaTrackerPropagatorOpposite")
)

from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
phase2_tracker.toModify(materialDumperAnalyzer, TrackerRecHitBuilder='WithTrackAngle')

materialDumper = cms.Sequence(materialDumperAnalyzer)
materialDumper_step = cms.Path(materialDumper)

