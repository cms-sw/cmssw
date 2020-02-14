import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
materialDumperAnalyzer = DQMEDAnalyzer('TrackingRecoMaterialAnalyser',
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
                                        MTDRecHitBuilder = cms.string('MTDRecHitBuilder'),
                                        RefitDirection = cms.string('alongMomentum'),
                                        RefitRPCHits = cms.bool(True),
                                        Propagator = cms.string('SmartPropagatorAnyRKOpposite'),
                                        #Propagators
                                        PropagatorAlong = cms.string("RungeKuttaTrackerPropagator"),
                                        PropagatorOpposite = cms.string("RungeKuttaTrackerPropagatorOpposite")
)

from Configuration.ProcessModifiers.run4_PixelCPEGeneric_cff import run4_PixelCPEGeneric
run4_PixelCPEGeneric.toModify(materialDumperAnalyzer, TrackerRecHitBuilder='WithTrackAngle')

materialDumper = cms.Sequence(materialDumperAnalyzer)
materialDumper_step = cms.Path(materialDumper)

