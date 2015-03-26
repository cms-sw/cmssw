import FWCore.ParameterSet.Config as cms

# Take all pixel tracks but the potential electrons
hltL1SeededEgammaRegionalCTFFinalFitWithMaterial = cms.EDProducer("TrackProducer",
                                                                  src = cms.InputTag("globalPixelTrackCandidatesForPhotons"),
                                                                  TTRHBuilder = cms.string('WithoutRefit'),
                                                                  Fitter = cms.string('KFFittingSmootherWithOutlierRejection'),
                                                                  Propagator = cms.string('PropagatorWithMaterial'),
                                                                  TrajectoryInEvent = cms.bool(True),
                                                                  AlgorithmName = cms.string('undefAlgorithm'),
                                                                  MeasurementTrackerEvent = cms.InputTag("MeasurementTrackerEvent"),
                                                                  beamSpot = cms.InputTag("offlineBeamSpot"),
                                                                  clusterRemovalInfo = cms.InputTag(""),
                                                                  SimpleMagneticField = cms.string(''),
                                                                  NavigationSchool = cms.string('SimpleNavigationSchool'),
                                                                  useSimpleMF = cms.bool(False),
                                                                  GeometricInnerState = cms.bool(False),
                                                                  MeasurementTracker = cms.string(''),
                                                                  useHitsSplitting = cms.bool(False))




# The sequence
HLTL1SeededEgammaRegionalRecoTrackerSequence = cms.Sequence(cms.SequencePlaceholder("globalPixelTracking")
                                                            +hltL1SeededEgammaRegionalCTFFinalFitWithMaterial
)

