
def loadKAACommons( cms, process ) :
    #
    # magnetic field
    #
    process.load( "Configuration.StandardSequences.MagneticField_cff" )
    #
    # ideal geometry and interface
    #
    process.load( "Geometry.CMSCommonData.cmsIdealGeometryXML_cfi" )
    process.load( "Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi" )
    process.load( "Geometry.CommonDetUnit.bareGlobalTrackingGeometry_cfi" )
    #
    # beam spot
    #
    process.load( "RecoVertex.BeamSpotProducer.BeamSpot_cff" )
    #
    # track selector
    #
    process.load("Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi")
    #
    # track refitter (adapted to alignment needs)
    #
    process.load( "TrackingTools.KalmanUpdators.KFUpdatorESProducer_cfi" )
    process.load( "TrackingTools.KalmanUpdators.Chi2MeasurementEstimatorESProducer_cfi" )
    process.load( "TrackingTools.TrackFitters.KFTrajectoryFitterESProducer_cfi" )
    process.load( "TrackingTools.TrackFitters.KFTrajectorySmootherESProducer_cfi" )
    process.load( "TrackingTools.TrackFitters.KFFittingSmootherESProducer_cfi" )
    process.load( "TrackingTools.MaterialEffects.MaterialPropagator_cfi" )
    process.load( "TrackingTools.GeomPropagators.AnalyticalPropagator_cfi" )
    process.load( "RecoTracker.TrackProducer.RefitterWithMaterial_cff" )
    process.load( "RecoTracker.TransientTrackingRecHit.TTRHBuilders_cff" )
    process.TrackRefitter.src = "AlignmentTrackSelector"
    process.TrackRefitter.TTRHBuilder = "WithAngleAndTemplate"
    process.TrackRefitter.TrajectoryInEvent = True
    #
    # alignment producer
    #
    process.load( "Alignment.CommonAlignmentProducer.AlignmentProducer_cff" )
    process.AlignmentProducer.algoConfig = cms.PSet( process.KalmanAlignmentAlgorithm )
    #
    # KAA specific includes
    #
    process.load( "Alignment.KalmanAlignmentAlgorithm.AlignmentUpdators_cff" )
    process.load( "Alignment.KalmanAlignmentAlgorithm.MetricsUpdators_cff" )
    process.load( "Alignment.ReferenceTrajectories.TrajectoryFactories_cff" )
    #
    # standard processing path for track-based alignment
    #
    process.p = cms.Path( process.AlignmentTrackSelector * process.offlineBeamSpot * process.TrackRefitter )
