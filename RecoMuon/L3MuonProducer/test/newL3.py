import FWCore.ParameterSet.Config as cms

def addL3ToHLT(process):
	def filters_by_type(process, type):
		return (filter for filter in process._Process__filters.values() if filter._TypedParameterizable__type == type)
	
	for l3Filter in filters_by_type(process, 'HLTMuonL3PreFilter'):
		if hasattr(l3Filter, 'CandTag'):
			if (l3Filter.CandTag==cms.InputTag("hltL3MuonCandidates")):
				l3Filter.CandTag=cms.InputTag("hltBRSL3MuonCandidates")
				l3Filter.InputLinks=cms.InputTag( "hltBRSL3MuonsLinksCombination")

	if hasattr(process, 'hltPixelTracksForSeedsL3Muon'):
		process.hltPixelTracksForSeedsL3Muon.RegionFactoryPSet.RegionPSet.input=cms.InputTag("hltBRSL3MuonCandidates")
	if hasattr(process, 'hltIter1L3MuonPixelSeeds'):
		process.hltIter1L3MuonPixelSeeds.RegionFactoryPSet.RegionPSet.input=cms.InputTag("hltBRSL3MuonCandidates")
	if hasattr(process, 'hltIter2L3MuonPixelSeeds'):
		process.hltIter2L3MuonPixelSeeds.RegionFactoryPSet.RegionPSet.input=cms.InputTag("hltBRSL3MuonCandidates")

	def producers_by_type(process, type):
    		return (module for module in process._Process__producers.values() if module._TypedParameterizable__type == type)

	for PFModule in producers_by_type(process, 'MuonHLTRechitInRegionsProducer'):
		if hasattr(PFModule, 'l1TagIsolated'):
			if(PFModule.l1TagIsolated==cms.InputTag("hltL3MuonCandidates")):
				PFModule.l1TagIsolated=cms.InputTag("hltBRSL3MuonCandidates")
	#Isolation paths:
	for PFModule in producers_by_type(process, 'L3MuonCombinedRelativeIsolationProducer'):
		if hasattr(PFModule, 'inputMuonCollection'):
			if(PFModule.inputMuonCollection==cms.InputTag("hltL3MuonCandidates")):
				PFModule.inputMuonCollection=cms.InputTag("hltBRSL3MuonCandidates")

	for l3Filter in filters_by_type(process, 'HLTMuonIsoFilter'):
		if hasattr(l3Filter, 'CandTag'):
			if (l3Filter.CandTag==cms.InputTag("hltL3MuonCandidates")):
				l3Filter.CandTag=cms.InputTag("hltBRSL3MuonCandidates")

	for l3Filter in filters_by_type(process, 'HLTMuonDimuonL3Filter'):
		if hasattr(l3Filter, 'CandTag'):
			if (l3Filter.CandTag==cms.InputTag("hltL3MuonCandidates")):
				l3Filter.CandTag=cms.InputTag("hltBRSL3MuonCandidates")

	if hasattr(process, 'hltMuonEcalPFClusterIsoForMuons'):
		process.hltMuonEcalPFClusterIsoForMuons.recoCandidateProducer = cms.InputTag("hltBRSL3MuonCandidates")

	if hasattr(process, 'hltDiMuonLinks'):
		process.hltDiMuonLinks.LinkCollection = cms.InputTag("hltBRSL3MuonsLinksCombination")
	#############################################################
	#Making Pixel Vertices:
	process.hltPixelTracks = cms.EDProducer( "PixelTrackProducer",
	    FilterPSet = cms.PSet(
	      chi2 = cms.double( 1000.0 ),
	      nSigmaTipMaxTolerance = cms.double( 0.0 ),
	      ComponentName = cms.string( "PixelTrackFilterByKinematics" ),
	      nSigmaInvPtTolerance = cms.double( 0.0 ),
	      ptMin = cms.double( 0.1 ),
	      tipMax = cms.double( 1.0 )
	    ),
	    useFilterWithES = cms.bool( False ),
	    passLabel = cms.string( "Pixel triplet primary tracks with vertex constraint" ),
	    FitterPSet = cms.PSet(
	      ComponentName = cms.string( "PixelFitterByHelixProjections" ),
	      TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
	      fixImpactParameter = cms.double( 0.0 )
	    ),
	    RegionFactoryPSet = cms.PSet(
	      ComponentName = cms.string( "GlobalRegionProducerFromBeamSpot" ),
	      RegionPSet = cms.PSet(
	        precise = cms.bool( True ),
	        originRadius = cms.double( 0.2 ),
	        ptMin = cms.double( 0.9 ),
	        originHalfLength = cms.double( 24.0 ),
	        beamSpot = cms.InputTag( "hltOnlineBeamSpot" )
	      )
	    ),
	    CleanerPSet = cms.PSet(  ComponentName = cms.string( "PixelTrackCleanerBySharedHits" ) ),
	    OrderedHitsFactoryPSet = cms.PSet(
	      ComponentName = cms.string( "StandardHitTripletGenerator" ),
	      GeneratorPSet = cms.PSet(
	        useBending = cms.bool( True ),
	        useFixedPreFiltering = cms.bool( False ),
	        maxElement = cms.uint32( 100000 ),
	        phiPreFiltering = cms.double( 0.3 ),
	        extraHitRPhitolerance = cms.double( 0.06 ),
	        useMultScattering = cms.bool( True ),
	        SeedComparitorPSet = cms.PSet(
	          ComponentName = cms.string( "LowPtClusterShapeSeedComparitor" ),
	          clusterShapeCacheSrc = cms.InputTag( "hltSiPixelClustersCache" )
	        ),
	        extraHitRZtolerance = cms.double( 0.06 ),
	        ComponentName = cms.string( "PixelTripletHLTGenerator" )
	      ),
	      SeedingLayers = cms.InputTag( "hltPixelLayerTriplets" )
	    )
	)
	
	process.hltPixelVertices = cms.EDProducer( "PixelVertexProducer",
	    WtAverage = cms.bool( True ),
	    Method2 = cms.bool( True ),
	    beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
	    PVcomparer = cms.PSet(  refToPSet_ = cms.string( "HLTPSetPvClusterComparerForIT" ) ),
	    Verbosity = cms.int32( 0 ),
	    UseError = cms.bool( True ),
	    TrackCollection = cms.InputTag( "hltPixelTracks" ),
	    PtMin = cms.double( 1.0 ),
	    NTrkMin = cms.int32( 2 ),
	    ZOffset = cms.double( 5.0 ),
	    Finder = cms.string( "DivisiveVertexFinder" ),
	    ZSeparation = cms.double( 0.05 )
	)
	#/Making Pixel Vertices, could probably use the following PTP tho?
	
	process.HLTRecopixelvertexingSequence = cms.Sequence(
	 process.hltPixelLayerTriplets
	 + process.hltPixelTracks
	 + process.hltPixelVertices
	)
	
	######### Define Master MTRB ROI, using values from Tau TRP:
	MasterMuonTrackingRegionBuilder = cms.PSet(
	        Rescale_eta = cms.double( 3.0 ),
	        Rescale_phi = cms.double( 3.0 ),
	        Rescale_Dz = cms.double( 4.0 ),                 #Normally 4
	        EscapePt = cms.double( 3.0 ),                   #Normally 1.5 but it should be at least 8 for us
	        EtaR_UpperLimit_Par1 = cms.double( 0.25 ),      #Normally 0.25
	        EtaR_UpperLimit_Par2 = cms.double( 0.15 ),      #Normally 0.15
	        PhiR_UpperLimit_Par1 = cms.double( 0.6 ),       #Normally 0.6
	        PhiR_UpperLimit_Par2 = cms.double( 0.2 ),       #Normally 0.2
	        UseVertex = cms.bool( False ),                  #Normally False
	        Pt_fixed = cms.bool( False ),                   #Normally True
	        Z_fixed = cms.bool( False ),    #True for IOH
	        Phi_fixed = cms.bool( True ),   #False for IOH
	        Eta_fixed = cms.bool( True ),   #False for IOH
	        Pt_min = cms.double( 3.0 ),     #Is 0.9 for Tau; normally 8 here
	        Phi_min = cms.double( 0.1 ),
	        Eta_min = cms.double( 0.1 ),
	        DeltaZ = cms.double( 24.2 ),    #default for tau: 24.2, for old IOH: 15.9
	        DeltaR = cms.double( 0.025 ),   #This changes for different iterations. for old IOH: ?
	        DeltaEta = cms.double( 0.04 ),  #default 0.15
	        DeltaPhi = cms.double( 0.15 ),   #default 0.2
	        maxRegions = cms.int32( 2 ),
	        precise = cms.bool( True ),
	        OnDemand = cms.int32( -1 ),
	        MeasurementTrackerName = cms.InputTag( "hltESPMeasurementTracker" ),
	        beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
	        vertexCollection = cms.InputTag( "pixelVertices" ), #Warning: I am not generating colleciton. Vertex is off anyway
	        input = cms.InputTag( 'hltL2Muons','UpdatedAtVtx' )
	)
	
	IterMasterMuonTrackingRegionBuilder = MasterMuonTrackingRegionBuilder
	IterMasterMuonTrackingRegionBuilder.input = cms.InputTag( 'hltL2SelectorForL3OI')	#Switch off for IO Only
	
	
	########## OI Algorthim:
	#Trajectory Filter
	process.HLTPSetCkfTrajectoryFilterNewOI = cms.PSet(
	  ComponentType = cms.string( "CkfBaseTrajectoryFilter" ),
	  minimumNumberOfHits = cms.int32( 5 ),
	  chargeSignificance = cms.double( -1.0 ),
	  minPt = cms.double( 0.9 ),
	  nSigmaMinPt = cms.double( 5.0 ),
	  minHitsMinPt = cms.int32( 3 ),
	  maxLostHits = cms.int32( 999 ),
	  maxConsecLostHits = cms.int32( 1 ),
	  maxNumberOfHits = cms.int32( 100 ),
	  maxLostHitsFraction = cms.double(1./10),
	  constantValueForLostHitsFractionFilter = cms.double(10),
	  minNumberOfHits = cms.int32(13),
	  minNumberOfHitsPerLoop = cms.int32(4),
	  extraNumberOfHitsBeforeTheFirstLoop = cms.int32(4)
	)
	
	process.HLTPSetCkfTrajectoryBuilderNewOI = cms.PSet(
	  propagatorAlong = cms.string( "PropagatorWithMaterial" ),
	  trajectoryFilter = cms.PSet(  refToPSet_ = cms.string( "HLTPSetCkfTrajectoryFilterNewOI" ) ),
	  maxCand = cms.int32( 5 ),
	  ComponentType = cms.string( "CkfTrajectoryBuilder" ),
	  propagatorOpposite = cms.string( "PropagatorWithMaterialOpposite" ),
	  MeasurementTrackerName = cms.string( "hltESPMeasurementTracker" ),
	  estimator = cms.string( "hltESPChi2MeasurementEstimator30" ),
	  TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
	  updator = cms.string( "hltESPKFUpdator" ),
	  alwaysUseInvalidHits = cms.bool( True ),
	  intermediateCleaning = cms.bool( True ),
	  lostHitPenalty = cms.double( 1.0 )
	)
	
	process.hltESPChi2MeasurementEstimator100 = cms.ESProducer( "Chi2MeasurementEstimatorESProducer",
	  MaxChi2 = cms.double( 100.0 ),
	  nSigma = cms.double( 4.0 ),
	  ComponentName = cms.string( "hltESPChi2MeasurementEstimator100" )
	)
	
	
	#OI Seeding:
	process.hltBRSOISeedsFromL2Muons = cms.EDProducer("TSGForOI",
	        MeasurementTrackerEvent = cms.InputTag("hltSiStripClusters"),
	        UseHitlessSeeds = cms.bool(True),
	        adjustErrorsDyanmicallyForHitless = cms.bool(False),
	        adjustErrorsDyanmicallyForHits = cms.bool(False),
	        debug = cms.untracked.bool(True),
	        estimator = cms.string('hltESPChi2MeasurementEstimator100'),
	        fixedErrorRescaleFactorForHitless = cms.double(5.0),
	        fixedErrorRescaleFactorForHits = cms.double(2.0),
	        hitsToTry = cms.int32(1),
	        layersToTry = cms.int32(1),
	        maxEtaForTOB = cms.double(1.2),
	        maxSeeds = cms.uint32(1),
	        minEtaForTEC = cms.double(0.8),
	        src = cms.InputTag("hltL2Muons","UpdatedAtVtx")
	)
	
	###---------- Trajectory Cleaner, deciding how overlapping track candidates are arbitrated  ----------------
	import TrackingTools.TrajectoryCleaning.TrajectoryCleanerBySharedHits_cfi
	process.muonSeededTrajectoryCleanerBySharedHits = TrackingTools.TrajectoryCleaning.TrajectoryCleanerBySharedHits_cfi.trajectoryCleanerBySharedHits.clone(
	    ComponentName = cms.string('muonSeededTrajectoryCleanerBySharedHits'),
	    fractionShared = cms.double(0.1),
	    ValidHitBonus = cms.double(1000.0),
	    MissingHitPenalty = cms.double(1.0),
	    ComponentType = cms.string('TrajectoryCleanerBySharedHits'),
	    allowSharedFirstHit = cms.bool(True)
	)
	
	#OI Trajectory Building:
	process.hltBRSOITrackCandidates = cms.EDProducer("CkfTrackCandidateMaker",
	    RedundantSeedCleaner = cms.string('CachingSeedCleanerBySharedInput'),
	    TrajectoryCleaner = cms.string('muonSeededTrajectoryCleanerBySharedHits'), #TrajectoryCleanerBySharedHits
	    cleanTrajectoryAfterInOut = cms.bool(True),
	    useHitsSplitting = cms.bool(True),
	    doSeedingRegionRebuilding = cms.bool(True),
	    maxNSeeds = cms.uint32(500000),
	    maxSeedsBeforeCleaning = cms.uint32(5000),
	    src = cms.InputTag('hltBRSOISeedsFromL2Muons'),
	    SimpleMagneticField = cms.string(''),
	    NavigationSchool = cms.string('SimpleNavigationSchool'),
	    TrajectoryBuilder = cms.string('CkfTrajectoryBuilder'),
	    TrajectoryBuilderPSet = cms.PSet(refToPSet_ = cms.string('HLTPSetCkfTrajectoryBuilderNewOI')),      #Was HLTPSetCkfTrajectoryBuilder
	    TransientInitialStateEstimatorParameters = cms.PSet(
	        propagatorAlongTISE = cms.string('PropagatorWithMaterialParabolicMf'),  # parabolic magnetic field
	        propagatorOppositeTISE = cms.string('PropagatorWithMaterialParabolicMfOpposite'), # parabolic magnetic field
	        numberMeasurementsForFit = cms.int32(4)
	    ),
	    MeasurementTrackerEvent = cms.InputTag("hltSiStripClusters"),
	    reverseTrajectories = cms.bool( True )
	)
	
	###-------------  Fitter-Smoother -------------------
#	process.load("TrackingTools.MaterialEffects.RungeKuttaTrackerPropagator_cfi")
#	import TrackingTools.TrackFitters.RungeKuttaFitters_cff
	process.load("TrackingTools.TrackFitters.RungeKuttaFitters_cff")

	process.RKTrajectoryFitter.Propagator = "hltESPRungeKuttaTrackerPropagator"
	process.RKTrajectorySmoother.Propagator = "hltESPRungeKuttaTrackerPropagator"
	process.RKTrajectoryFitter.Updator = "hltESPKFUpdator"
	process.RKTrajectorySmoother.Updator = "hltESPKFUpdator"
	process.RKTrajectoryFitter.Estimator = "hltESPChi2MeasurementEstimator30"
	process.RKTrajectorySmoother.Estimator = "hltESPChi2MeasurementEstimator30"
	process.RKTrajectoryFitter.RecoGeometry = "hltESPGlobalDetLayerGeometry"
	process.RKTrajectorySmoother.RecoGeometry = "hltESPGlobalDetLayerGeometry"

	process.muonSeededFittingSmootherWithOutliersRejectionAndRK = TrackingTools.TrackFitters.RungeKuttaFitters_cff.KFFittingSmootherWithOutliersRejectionAndRK.clone(
	    ComponentName = cms.string("muonSeededFittingSmootherWithOutliersRejectionAndRK"),
	    BreakTrajWith2ConsecutiveMissing = cms.bool(False),
	    EstimateCut = cms.double(50.), ## was 20.
	)

	#OI Track Producer:
	process.hltBRSMuonSeededTracksOutIn = cms.EDProducer("TrackProducer",
	    useSimpleMF = cms.bool(False),
	    SimpleMagneticField = cms.string(""),
	    src = cms.InputTag("hltBRSOITrackCandidates"),       #Modified
	    clusterRemovalInfo = cms.InputTag(""),
	    beamSpot = cms.InputTag("hltOnlineBeamSpot"),       #Modified
	    Fitter = cms.string('muonSeededFittingSmootherWithOutliersRejectionAndRK'),       #Modified
	    useHitsSplitting = cms.bool(False),
	    alias = cms.untracked.string('ctfWithMaterialTracks'),
	    TrajectoryInEvent = cms.bool(True),
	    TTRHBuilder = cms.string('hltESPTTRHBWithTrackAngle'),      #Was: WithAngleAndTemplate
	    AlgorithmName = cms.string('iter10'),       #Modified
	    Propagator = cms.string('hltESPRungeKuttaTrackerPropagator'), #Others use PropagatorWithMaterial
	    GeometricInnerState = cms.bool(False),
	    NavigationSchool = cms.string('SimpleNavigationSchool'),  #Others are blank        
	    MeasurementTracker = cms.string("hltESPMeasurementTracker"),
	    MeasurementTrackerEvent = cms.InputTag('hltSiStripClusters'),       #Modified     
	)
	
	#OI L3 Muon Producer:
	process.hltL3MuonsBRSOI = cms.EDProducer( "L3MuonProducer",
	    ServiceParameters = cms.PSet(
	      Propagators = cms.untracked.vstring( 'hltESPSmartPropagatorAny',
	        'SteppingHelixPropagatorAny',
	        'hltESPSmartPropagator',
	        'hltESPSteppingHelixPropagatorOpposite' ),
	      RPCLayers = cms.bool( True ),
	      UseMuonNavigation = cms.untracked.bool( True )
	    ),
	    L3TrajBuilderParameters = cms.PSet(
	      ScaleTECyFactor = cms.double( -1.0 ),
	      GlbRefitterParameters = cms.PSet(
	        TrackerSkipSection = cms.int32( -1 ),
	        DoPredictionsOnly = cms.bool( False ),
	        PropDirForCosmics = cms.bool( False ),
	        HitThreshold = cms.int32( 1 ),
	        RefitFlag = cms.bool( True ),          #Usually true
	        MuonHitsOption = cms.int32( 1 ),
	        Chi2CutRPC = cms.double( 1.0 ),
	        Fitter = cms.string( "hltESPL3MuKFTrajectoryFitter" ),
	        DTRecSegmentLabel = cms.InputTag( "hltDt4DSegments" ),
	        TrackerRecHitBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
	        MuonRecHitBuilder = cms.string( "hltESPMuonTransientTrackingRecHitBuilder" ),
	        RefitDirection = cms.string( "insideOut" ),
	        CSCRecSegmentLabel = cms.InputTag( "hltCscSegments" ),
	        Chi2CutCSC = cms.double( 150.0 ),
	        Chi2CutDT = cms.double( 10.0 ),
	        RefitRPCHits = cms.bool( True ),
	        SkipStation = cms.int32( -1 ),
	        Propagator = cms.string( "hltESPSmartPropagatorAny" ),
	        TrackerSkipSystem = cms.int32( -1 ),
	        DYTthrs = cms.vint32( 30, 15 )
	      ),
	      ScaleTECxFactor = cms.double( -1.0 ),
	      TrackerRecHitBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
	      MuonRecHitBuilder = cms.string( "hltESPMuonTransientTrackingRecHitBuilder" ),
#	      MuonTrackingRegionBuilder = MasterMuonTrackingRegionBuilder,      #Using the master Muon ROI params - Although it is not used
             MuonTrackingRegionBuilder = cms.PSet(
	        Rescale_eta = cms.double( 3.0 ),
	        Rescale_phi = cms.double( 3.0 ),
	        Rescale_Dz = cms.double( 4.0 ),                 #Normally 4
	        EscapePt = cms.double( 3.0 ),                   #Normally 1.5 but it should be at least 8 for us
	        EtaR_UpperLimit_Par1 = cms.double( 0.25 ),      #Normally 0.25
	        EtaR_UpperLimit_Par2 = cms.double( 0.15 ),      #Normally 0.15
	        PhiR_UpperLimit_Par1 = cms.double( 0.6 ),       #Normally 0.6
	        PhiR_UpperLimit_Par2 = cms.double( 0.2 ),       #Normally 0.2
	        UseVertex = cms.bool( False ),                  #Normally False
	        Pt_fixed = cms.bool( False ),                   #Normally True
	        Z_fixed = cms.bool( False ),    #True for IOH
	        Phi_fixed = cms.bool( True ),   #False for IOH
	        Eta_fixed = cms.bool( True ),   #False for IOH
	        Pt_min = cms.double( 3.0 ),     #Is 0.9 for Tau; normally 8 here
	        Phi_min = cms.double( 0.1 ),
	        Eta_min = cms.double( 0.1 ),
	        DeltaZ = cms.double( 24.2 ),    #default for tau: 24.2, for old IOH: 15.9
	        DeltaR = cms.double( 0.025 ),   #This changes for different iterations. for old IOH: ?
	        DeltaEta = cms.double( 0.04 ),  #default 0.15
	        DeltaPhi = cms.double( 0.15 ),   #default 0.2
	        maxRegions = cms.int32( 2 ),
	        precise = cms.bool( True ),
	        OnDemand = cms.int32( -1 ),
	        MeasurementTrackerName = cms.InputTag( "hltESPMeasurementTracker" ),
	        beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
	        vertexCollection = cms.InputTag( "pixelVertices" ), #Warning: I am not generating colleciton. Vertex is off anyway
	        input = cms.InputTag( 'hltL2Muons','UpdatedAtVtx' )
	      ),
	      RefitRPCHits = cms.bool( True ),
	      PCut = cms.double( 2.5 ),
	      TrackTransformer = cms.PSet(
	        DoPredictionsOnly = cms.bool( False ),
	        Fitter = cms.string( "hltESPL3MuKFTrajectoryFitter" ),
	        TrackerRecHitBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
	        Smoother = cms.string( "hltESPKFTrajectorySmootherForMuonTrackLoader" ),
	        MuonRecHitBuilder = cms.string( "hltESPMuonTransientTrackingRecHitBuilder" ),
	        RefitDirection = cms.string( "insideOut" ),
	        RefitRPCHits = cms.bool( True ),
	        Propagator = cms.string( "hltESPSmartPropagatorAny" )
	      ),
	      GlobalMuonTrackMatcher = cms.PSet(
	        Pt_threshold1 = cms.double( 0.0 ),
	        DeltaDCut_3 = cms.double( 15.0 ),
	        MinP = cms.double( 2.5 ),
	        MinPt = cms.double( 1.0 ),
	        Chi2Cut_1 = cms.double( 50.0 ),
	        Pt_threshold2 = cms.double( 9.99999999E8 ),
	        LocChi2Cut = cms.double( 0.001 ),
	        Eta_threshold = cms.double( 1.2 ),
	        Quality_3 = cms.double( 7.0 ),
	        Quality_2 = cms.double( 15.0 ),
	        Chi2Cut_2 = cms.double( 50.0 ),
	        Chi2Cut_3 = cms.double( 200.0 ),
	        DeltaDCut_1 = cms.double( 40.0 ),
	        DeltaRCut_2 = cms.double( 0.2 ),
	        DeltaRCut_3 = cms.double( 1.0 ),
	        DeltaDCut_2 = cms.double( 10.0 ),
	        DeltaRCut_1 = cms.double( 0.1 ),
	        Propagator = cms.string( "hltESPSmartPropagator" ),
	        Quality_1 = cms.double( 20.0 )
	      ),
	      PtCut = cms.double( 1.0 ),
	      TrackerPropagator = cms.string( "SteppingHelixPropagatorAny" ),
	      tkTrajLabel = cms.InputTag( "hltBRSMuonSeededTracksOutIn" ),    #Feed tracks from iterations into L3MTB
	      tkTrajBeamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
	      tkTrajMaxChi2 = cms.double( 9999.0 ),
	      tkTrajMaxDXYBeamSpot = cms.double( 9999.0 ),      #Using same values as old algos
	      tkTrajVertex = cms.InputTag( "hltPixelVertices" ),        #From pixelVertice      #From pixelVerticesss
	      tkTrajUseVertex = cms.bool( False ),
	      matchToSeeds = cms.bool( True )
	    ),
	    TrackLoaderParameters = cms.PSet(
	      PutTkTrackIntoEvent = cms.untracked.bool( False ),
	      beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
	      SmoothTkTrack = cms.untracked.bool( False ),
	      MuonSeededTracksInstance = cms.untracked.string( "L2Seeded" ),
	      Smoother = cms.string( "hltESPKFTrajectorySmootherForMuonTrackLoader" ),
	      MuonUpdatorAtVertexParameters = cms.PSet(
	        MaxChi2 = cms.double( 1000000.0 ),
	        Propagator = cms.string( "hltESPSteppingHelixPropagatorOpposite" ),
	        BeamSpotPositionErrors = cms.vdouble( 0.1, 0.1, 5.3 )
	      ),
	      VertexConstraint = cms.bool( False ),
	      DoSmoothing = cms.bool( False )    #Usually true
	    ),
	    MuonCollectionLabel = cms.InputTag( 'hltL2Muons','UpdatedAtVtx' )
	)
	
	process.hltBRSOIL3MuonsLinksCombination = cms.EDProducer( "L3TrackLinksCombiner",
	    labels = cms.VInputTag( 'hltL3MuonsBRSOI' )
	)
	process.hltBRSOIL3Muons = cms.EDProducer( "L3TrackCombiner",
	    labels = cms.VInputTag( 'hltL3MuonsBRSOI' )
	)
	process.hltBRSOIL3MuonCandidates = cms.EDProducer( "L3MuonCandidateProducer",
	    InputLinksObjects = cms.InputTag( "hltBRSOIL3MuonsLinksCombination" ),
	    InputObjects = cms.InputTag( "hltBRSOIL3Muons" ),
	    MuonPtOption = cms.string( "Tracker" )
	)
	
	process.hltL2SelectorForL3OI = cms.EDProducer("HLTMuonL2SelectorForL3IO",
	    l2Src = cms.InputTag('hltL2Muons','UpdatedAtVtx'),
	    l3OISrc = cms.InputTag('hltL3MuonsBRSOI')
	)
	
	
	########## IO Algorthim:
	#Making Pixel Vertices:
	process.hltPixelTracks = cms.EDProducer( "PixelTrackProducer",
	    FilterPSet = cms.PSet(
	      chi2 = cms.double( 1000.0 ),
	      nSigmaTipMaxTolerance = cms.double( 0.0 ),
	      ComponentName = cms.string( "PixelTrackFilterByKinematics" ),
	      nSigmaInvPtTolerance = cms.double( 0.0 ),
	      ptMin = cms.double( 0.1 ),
	      tipMax = cms.double( 1.0 )
	    ),
	    useFilterWithES = cms.bool( False ),
	    passLabel = cms.string( "Pixel triplet primary tracks with vertex constraint" ),
	    FitterPSet = cms.PSet(
	      ComponentName = cms.string( "PixelFitterByHelixProjections" ),
	      TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
	      fixImpactParameter = cms.double( 0.0 )
	    ),
	    RegionFactoryPSet = cms.PSet(
	      ComponentName = cms.string( "GlobalRegionProducerFromBeamSpot" ),
	      RegionPSet = cms.PSet(
	        precise = cms.bool( True ),
	        originRadius = cms.double( 0.2 ),
	        ptMin = cms.double( 0.9 ),
	        originHalfLength = cms.double( 24.0 ),
	        beamSpot = cms.InputTag( "hltOnlineBeamSpot" )
	      )
	    ),
	    CleanerPSet = cms.PSet(  ComponentName = cms.string( "PixelTrackCleanerBySharedHits" ) ),
	    OrderedHitsFactoryPSet = cms.PSet(
	      ComponentName = cms.string( "StandardHitTripletGenerator" ),
	      GeneratorPSet = cms.PSet(
	        useBending = cms.bool( True ),
	        useFixedPreFiltering = cms.bool( False ),
	        maxElement = cms.uint32( 100000 ),
	        phiPreFiltering = cms.double( 0.3 ),
	        extraHitRPhitolerance = cms.double( 0.06 ),
	        useMultScattering = cms.bool( True ),
	        SeedComparitorPSet = cms.PSet(
	          ComponentName = cms.string( "LowPtClusterShapeSeedComparitor" ),
	          clusterShapeCacheSrc = cms.InputTag( "hltSiPixelClustersCache" )
	        ),
	        extraHitRZtolerance = cms.double( 0.06 ),
	        ComponentName = cms.string( "PixelTripletHLTGenerator" )
	      ),
	      SeedingLayers = cms.InputTag( "hltPixelLayerTriplets" )
	    )
	)
	
	process.hltPixelVertices = cms.EDProducer( "PixelVertexProducer",
	    WtAverage = cms.bool( True ),
	    Method2 = cms.bool( True ),
	    beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
	    PVcomparer = cms.PSet(  refToPSet_ = cms.string( "HLTPSetPvClusterComparerForIT" ) ),
	    Verbosity = cms.int32( 0 ),
	    UseError = cms.bool( True ),
	    TrackCollection = cms.InputTag( "hltPixelTracks" ),
	    PtMin = cms.double( 1.0 ),
	    NTrkMin = cms.int32( 2 ),
	    ZOffset = cms.double( 5.0 ),
	    Finder = cms.string( "DivisiveVertexFinder" ),
	    ZSeparation = cms.double( 0.05 )
	)
	#/Making Pixel Vertices, could probably use the following PTP tho?
	
	#Start Iterative tracking:
	process.hltBRSIter0HighPtTkMuPixelTracks = cms.EDProducer( "PixelTrackProducer",
	    FilterPSet = cms.PSet(
	      chi2 = cms.double( 1000.0 ),
	      nSigmaTipMaxTolerance = cms.double( 0.0 ),
	      ComponentName = cms.string( "PixelTrackFilterByKinematics" ),
	      nSigmaInvPtTolerance = cms.double( 0.0 ),
	      ptMin = cms.double( 0.1 ),
	      tipMax = cms.double( 1.0 )
	    ),
	    useFilterWithES = cms.bool( False ),
	    passLabel = cms.string( "Pixel triplet primary tracks with vertex constraint" ),
	    FitterPSet = cms.PSet(
	      ComponentName = cms.string( "PixelFitterByHelixProjections" ),
	      TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
	      fixImpactParameter = cms.double( 0.0 )
	    ),
	    RegionFactoryPSet = IterMasterMuonTrackingRegionBuilder,
	    CleanerPSet = cms.PSet(  ComponentName = cms.string( "PixelTrackCleanerBySharedHits" ) ),
	    OrderedHitsFactoryPSet = cms.PSet(
	      ComponentName = cms.string( "StandardHitTripletGenerator" ),
	      GeneratorPSet = cms.PSet(
	        useBending = cms.bool( True ),
	        useFixedPreFiltering = cms.bool( False ),
	        maxElement = cms.uint32( 100000 ),
	        phiPreFiltering = cms.double( 0.3 ),
	        extraHitRPhitolerance = cms.double( 0.06 ),
	        useMultScattering = cms.bool( True ),
	        SeedComparitorPSet = cms.PSet(
	          ComponentName = cms.string( "LowPtClusterShapeSeedComparitor" ),
	          clusterShapeCacheSrc = cms.InputTag( "hltSiPixelClustersCache" )
	        ),
	        extraHitRZtolerance = cms.double( 0.06 ),
	        ComponentName = cms.string( "PixelTripletHLTGenerator" )
	      ),
	      SeedingLayers = cms.InputTag( "hltPixelLayerTriplets" )
	    )
	)
	process.hltBRSIter0HighPtTkMuPixelTracks.RegionFactoryPSet.ComponentName = cms.string( "MuonTrackingRegionBuilder" )
	process.hltBRSIter0HighPtTkMuPixelTracks.RegionFactoryPSet.DeltaR = cms.double( 0.2 )
	
	process.hltBRSIter0HighPtTkMuPixelSeedsFromPixelTracks = cms.EDProducer( "SeedGeneratorFromProtoTracksEDProducer",
	    useEventsWithNoVertex = cms.bool( True ),
	    originHalfLength = cms.double( 1.0E9 ),
	    useProtoTrackKinematics = cms.bool( False ),
	    usePV = cms.bool( False ),
	    SeedCreatorPSet = cms.PSet(  refToPSet_ = cms.string( "HLTSeedFromProtoTracks" ) ),
	    InputVertexCollection = cms.InputTag( "" ),
	    TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
	    InputCollection = cms.InputTag( "hltBRSIter0HighPtTkMuPixelTracks" ),
	    originRadius = cms.double( 1.0E9 )
	)

	process.hltBRSIter0HighPtTkMuCkfTrackCandidates = cms.EDProducer( "CkfTrackCandidateMaker",
	    src = cms.InputTag( "hltBRSIter0HighPtTkMuPixelSeedsFromPixelTracks" ),
	    maxSeedsBeforeCleaning = cms.uint32( 1000 ),
	    SimpleMagneticField = cms.string( "ParabolicMf" ),
	    TransientInitialStateEstimatorParameters = cms.PSet(
	      propagatorAlongTISE = cms.string( "PropagatorWithMaterialParabolicMf" ),
	      numberMeasurementsForFit = cms.int32( 4 ),
	      propagatorOppositeTISE = cms.string( "PropagatorWithMaterialParabolicMfOpposite" )
	    ),
	    TrajectoryCleaner = cms.string( "hltESPTrajectoryCleanerBySharedHits" ),
	    MeasurementTrackerEvent = cms.InputTag( "hltSiStripClusters" ),
	    cleanTrajectoryAfterInOut = cms.bool( False ),
	    useHitsSplitting = cms.bool( False ),
	    RedundantSeedCleaner = cms.string( "CachingSeedCleanerBySharedInput" ),
	    doSeedingRegionRebuilding = cms.bool( False ),
	    maxNSeeds = cms.uint32( 100000 ),
	    TrajectoryBuilderPSet = cms.PSet(  refToPSet_ = cms.string( "HLTIter0HighPtTkMuPSetTrajectoryBuilderIT" ) ),
	    NavigationSchool = cms.string( "SimpleNavigationSchool" ),
	    TrajectoryBuilder = cms.string( "" )
	)
	process.hltBRSIter0HighPtTkMuCtfWithMaterialTracks = cms.EDProducer( "TrackProducer",
	    src = cms.InputTag( "hltBRSIter0HighPtTkMuCkfTrackCandidates" ),
	    SimpleMagneticField = cms.string( "ParabolicMf" ),
	    clusterRemovalInfo = cms.InputTag( "" ),
	    beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
	    MeasurementTrackerEvent = cms.InputTag( "hltSiStripClusters" ),
	    Fitter = cms.string( "hltESPFittingSmootherIT" ),
	    useHitsSplitting = cms.bool( False ),
	    MeasurementTracker = cms.string( "" ),
	    AlgorithmName = cms.string( "iter0" ),
	    alias = cms.untracked.string( "ctfWithMaterialTracks" ),
	    NavigationSchool = cms.string( "" ),
	    TrajectoryInEvent = cms.bool( True ),
	    TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
	    GeometricInnerState = cms.bool( True ),
	    useSimpleMF = cms.bool( True ),
	    Propagator = cms.string( "hltESPRungeKuttaTrackerPropagator" )
	)
	process.hltBRSIter0HighPtTkMuTrackSelectionHighPurity = cms.EDProducer( "AnalyticalTrackSelector",
	    max_d0 = cms.double( 100.0 ),
	    minNumber3DLayers = cms.uint32( 0 ),
	    max_lostHitFraction = cms.double( 1.0 ),
	    applyAbsCutsIfNoPV = cms.bool( False ),
	    qualityBit = cms.string( "highPurity" ),
	    minNumberLayers = cms.uint32( 3 ),
	    chi2n_par = cms.double( 0.7 ),
	    useVtxError = cms.bool( False ),
	    nSigmaZ = cms.double( 4.0 ),
	    dz_par2 = cms.vdouble( 0.4, 4.0 ),
	    applyAdaptedPVCuts = cms.bool( True ),
	    min_eta = cms.double( -9999.0 ),
	    dz_par1 = cms.vdouble( 0.35, 4.0 ),
	    copyTrajectories = cms.untracked.bool( True ),
	    vtxNumber = cms.int32( -1 ),
	    max_d0NoPV = cms.double( 100.0 ),
	    keepAllTracks = cms.bool( False ),
	    maxNumberLostLayers = cms.uint32( 1 ),
	    beamspot = cms.InputTag( "hltOnlineBeamSpot" ),
	    max_relpterr = cms.double( 9999.0 ),
	    copyExtras = cms.untracked.bool( True ),
	    max_z0NoPV = cms.double( 100.0 ),
	    vertexCut = cms.string( "tracksSize>=3" ),
	    max_z0 = cms.double( 100.0 ),
	    useVertices = cms.bool( False ),
	    min_nhits = cms.uint32( 0 ),
	    src = cms.InputTag( "hltBRSIter0HighPtTkMuCtfWithMaterialTracks" ),
	    max_minMissHitOutOrIn = cms.int32( 99 ),
	    chi2n_no1Dmod_par = cms.double( 9999.0 ),
	    vertices = cms.InputTag( "notUsed" ),
	    max_eta = cms.double( 9999.0 ),
	    d0_par2 = cms.vdouble( 0.4, 4.0 ),
	    d0_par1 = cms.vdouble( 0.3, 4.0 ),
	    res_par = cms.vdouble( 0.003, 0.001 ),
	    minHitsToBypassChecks = cms.uint32( 20 )
	)
	process.hltBRSIter2HighPtTkMuClustersRefRemoval = cms.EDProducer( "HLTTrackClusterRemoverNew",
	    doStrip = cms.bool( True ),
	    doStripChargeCheck = cms.bool( True ),
	    trajectories = cms.InputTag( "hltBRSIter0HighPtTkMuTrackSelectionHighPurity" ),
	    oldClusterRemovalInfo = cms.InputTag( "" ),
	    stripClusters = cms.InputTag( "hltSiStripRawToClustersFacility" ),
	    pixelClusters = cms.InputTag( "hltSiPixelClusters" ),
	    Common = cms.PSet(
	      maxChi2 = cms.double( 16.0 ),
	      minGoodStripCharge = cms.double( 60.0 )
	    ),
	    doPixel = cms.bool( True )
	)
	process.hltBRSIter2HighPtTkMuMaskedMeasurementTrackerEvent = cms.EDProducer( "MaskedMeasurementTrackerEventProducer",
	    clustersToSkip = cms.InputTag( "hltBRSIter2HighPtTkMuClustersRefRemoval" ),
	    OnDemand = cms.bool( False ),
	    src = cms.InputTag( "hltSiStripClusters" )
	)
	process.hltBRSIter2HighPtTkMuPixelLayerPairs = cms.EDProducer( "SeedingLayersEDProducer",
	    layerList = cms.vstring( 'BPix1+BPix2',
	      'BPix1+BPix3',
	      'BPix2+BPix3',
	      'BPix1+FPix1_pos',
	      'BPix1+FPix1_neg',
	      'BPix1+FPix2_pos',
	      'BPix1+FPix2_neg',
	      'BPix2+FPix1_pos',
	      'BPix2+FPix1_neg',
	      'BPix2+FPix2_pos',
	      'BPix2+FPix2_neg',
	      'FPix1_pos+FPix2_pos',
	      'FPix1_neg+FPix2_neg' ),
	    MTOB = cms.PSet(  ),
	    TEC = cms.PSet(  ),
	    MTID = cms.PSet(  ),
	    FPix = cms.PSet(
	      HitProducer = cms.string( "hltSiPixelRecHits" ),
	      hitErrorRZ = cms.double( 0.0036 ),
	      useErrorsFromParam = cms.bool( True ),
	      TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
	      skipClusters = cms.InputTag( "hltBRSIter2HighPtTkMuClustersRefRemoval" ),
	      hitErrorRPhi = cms.double( 0.0051 )
	    ),
	    MTEC = cms.PSet(  ),
	    MTIB = cms.PSet(  ),
	    TID = cms.PSet(  ),
	    TOB = cms.PSet(  ),
	    BPix = cms.PSet(
	      HitProducer = cms.string( "hltSiPixelRecHits" ),
	      hitErrorRZ = cms.double( 0.006 ),
	      useErrorsFromParam = cms.bool( True ),
	      TTRHBuilder = cms.string( "hltESPTTRHBuilderPixelOnly" ),
	      skipClusters = cms.InputTag( "hltBRSIter2HighPtTkMuClustersRefRemoval" ),
	      hitErrorRPhi = cms.double( 0.0027 )
	    ),
	    TIB = cms.PSet(  )
	)
	process.hltBRSIter2HighPtTkMuPixelSeeds = cms.EDProducer( "SeedGeneratorFromRegionHitsEDProducer",
	    RegionFactoryPSet = IterMasterMuonTrackingRegionBuilder,
	    SeedComparitorPSet = cms.PSet(
	      ComponentName = cms.string( "PixelClusterShapeSeedComparitor" ),
	      ClusterShapeHitFilterName = cms.string( "ClusterShapeHitFilter" ),
	      FilterPixelHits = cms.bool( True ), #Usually True 
	      FilterStripHits = cms.bool( False ),
	      FilterAtHelixStage = cms.bool( True ), #Usually True 
	      ClusterShapeCacheSrc = cms.InputTag( "hltSiPixelClustersCache" )
	    ),
	    ClusterCheckPSet = cms.PSet(
	      PixelClusterCollectionLabel = cms.InputTag( "hltSiPixelClusters" ),
	      MaxNumberOfCosmicClusters = cms.uint32( 50000 ),
	      doClusterCheck = cms.bool( False ),
	      ClusterCollectionLabel = cms.InputTag( "hltSiStripClusters" ),
	      MaxNumberOfPixelClusters = cms.uint32( 10000 )
	    ),
	    OrderedHitsFactoryPSet = cms.PSet(
	      maxElement = cms.uint32( 0 ),
	      ComponentName = cms.string( "StandardHitPairGenerator" ),
	      GeneratorPSet = cms.PSet(
	        maxElement = cms.uint32( 100000 ),
	        SeedComparitorPSet = cms.PSet(  ComponentName = cms.string( "none" ) )
	      ),
	      SeedingLayers = cms.InputTag( "hltBRSIter2HighPtTkMuPixelLayerPairs" )
	    ),
	    SeedCreatorPSet = cms.PSet(  refToPSet_ = cms.string( "HLTSeedFromConsecutiveHitsCreatorIT" ) ),
	    TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" )
	)
	process.hltBRSIter2HighPtTkMuPixelSeeds.RegionFactoryPSet.ComponentName = cms.string( "MuonTrackingRegionBuilder" )
	process.hltBRSIter2HighPtTkMuPixelSeeds.RegionFactoryPSet.DeltaR = cms.double( 0.025 )
	
	process.hltBRSIter2HighPtTkMuCkfTrackCandidates = cms.EDProducer( "CkfTrackCandidateMaker",
	    src = cms.InputTag( "hltBRSIter2HighPtTkMuPixelSeeds" ),
	    maxSeedsBeforeCleaning = cms.uint32( 1000 ),
	    SimpleMagneticField = cms.string( "ParabolicMf" ),
	    TransientInitialStateEstimatorParameters = cms.PSet(
	      propagatorAlongTISE = cms.string( "PropagatorWithMaterialParabolicMf" ),
	      numberMeasurementsForFit = cms.int32( 4 ),
	      propagatorOppositeTISE = cms.string( "PropagatorWithMaterialParabolicMfOpposite" )
	    ),
	    TrajectoryCleaner = cms.string( "hltESPTrajectoryCleanerBySharedHits" ),
	    MeasurementTrackerEvent = cms.InputTag( "hltBRSIter2HighPtTkMuMaskedMeasurementTrackerEvent" ),
	    cleanTrajectoryAfterInOut = cms.bool( False ),
	    useHitsSplitting = cms.bool( False ),
	    RedundantSeedCleaner = cms.string( "CachingSeedCleanerBySharedInput" ),
	    doSeedingRegionRebuilding = cms.bool( False ),
	    maxNSeeds = cms.uint32( 100000 ),
	    TrajectoryBuilderPSet = cms.PSet(  refToPSet_ = cms.string( "HLTIter2HighPtTkMuPSetTrajectoryBuilderIT" ) ),
	    NavigationSchool = cms.string( "SimpleNavigationSchool" ),
	    TrajectoryBuilder = cms.string( "" )
	)
	process.hltBRSIter2HighPtTkMuCtfWithMaterialTracks = cms.EDProducer( "TrackProducer",
	    src = cms.InputTag( "hltBRSIter2HighPtTkMuCkfTrackCandidates" ),
	    SimpleMagneticField = cms.string( "ParabolicMf" ),
	    clusterRemovalInfo = cms.InputTag( "" ),
	    beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
	    MeasurementTrackerEvent = cms.InputTag( "hltBRSIter2HighPtTkMuMaskedMeasurementTrackerEvent" ),
	    Fitter = cms.string( "hltESPFittingSmootherIT" ),
	    useHitsSplitting = cms.bool( False ),
	    MeasurementTracker = cms.string( "" ),
	    AlgorithmName = cms.string( "iter2" ),
	    alias = cms.untracked.string( "ctfWithMaterialTracks" ),
	    NavigationSchool = cms.string( "" ),
	    TrajectoryInEvent = cms.bool( True ),
	    TTRHBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
	    GeometricInnerState = cms.bool( True ),
	    useSimpleMF = cms.bool( True ),
	    Propagator = cms.string( "hltESPRungeKuttaTrackerPropagator" )
	)
	process.hltBRSIter2HighPtTkMuTrackSelectionHighPurity = cms.EDProducer( "AnalyticalTrackSelector",
	    max_d0 = cms.double( 100.0 ),
	    minNumber3DLayers = cms.uint32( 0 ),
	    max_lostHitFraction = cms.double( 1.0 ),
	    applyAbsCutsIfNoPV = cms.bool( False ),
	    qualityBit = cms.string( "highPurity" ),
	    minNumberLayers = cms.uint32( 3 ),
	    chi2n_par = cms.double( 0.7 ),
	    useVtxError = cms.bool( False ),
	    nSigmaZ = cms.double( 4.0 ),
	    dz_par2 = cms.vdouble( 0.4, 4.0 ),
	    applyAdaptedPVCuts = cms.bool( True ),
	    min_eta = cms.double( -9999.0 ),
	    dz_par1 = cms.vdouble( 0.35, 4.0 ),
	    copyTrajectories = cms.untracked.bool( True ),
	    vtxNumber = cms.int32( -1 ),
	    max_d0NoPV = cms.double( 100.0 ),
	    keepAllTracks = cms.bool( False ),
	    maxNumberLostLayers = cms.uint32( 1 ),
	    beamspot = cms.InputTag( "hltOnlineBeamSpot" ),
	    max_relpterr = cms.double( 9999.0 ),
	    copyExtras = cms.untracked.bool( True ),
	    max_z0NoPV = cms.double( 100.0 ),
	    vertexCut = cms.string( "tracksSize>=3" ),
	    max_z0 = cms.double( 100.0 ),
	    useVertices = cms.bool( False ),
	    min_nhits = cms.uint32( 0 ),
	    src = cms.InputTag( "hltBRSIter2HighPtTkMuCtfWithMaterialTracks" ),
	    max_minMissHitOutOrIn = cms.int32( 99 ),
	    chi2n_no1Dmod_par = cms.double( 9999.0 ),
	    vertices = cms.InputTag( "notUsed" ),
	    max_eta = cms.double( 9999.0 ),
	    d0_par2 = cms.vdouble( 0.4, 4.0 ),
	    d0_par1 = cms.vdouble( 0.3, 4.0 ),
	    res_par = cms.vdouble( 0.003, 0.001 ),
	    minHitsToBypassChecks = cms.uint32( 20 )
	)
	process.hltBRSIter2HighPtTkMuMerged = cms.EDProducer( "TrackListMerger",
	    ShareFrac = cms.double( 0.19 ),
	    writeOnlyTrkQuals = cms.bool( False ),
	    MinPT = cms.double( 0.05 ),
	    allowFirstHitShare = cms.bool( True ),
	    copyExtras = cms.untracked.bool( True ),
	    Epsilon = cms.double( -0.001 ),
	    selectedTrackQuals = cms.VInputTag( 'hltBRSIter0HighPtTkMuTrackSelectionHighPurity','hltBRSIter2HighPtTkMuTrackSelectionHighPurity' ),
	    indivShareFrac = cms.vdouble( 1.0, 1.0 ),
	    MaxNormalizedChisq = cms.double( 1000.0 ),
	    copyMVA = cms.bool( False ),
	    FoundHitBonus = cms.double( 5.0 ),
	    setsToMerge = cms.VPSet( 
	      cms.PSet(  pQual = cms.bool( False ),
	        tLists = cms.vint32( 0, 1 )
	      )
	    ),
	    MinFound = cms.int32( 3 ),
	    hasSelector = cms.vint32( 0, 0 ),
	    TrackProducers = cms.VInputTag( 'hltBRSIter0HighPtTkMuTrackSelectionHighPurity','hltBRSIter2HighPtTkMuTrackSelectionHighPurity' ),
	    LostHitPenalty = cms.double( 20.0 ),
	    newQuality = cms.string( "confirmed" )
	)

	#Iterative tracking finished
	
	# L3MuonProducer from iterative tracking:
	process.BRSIterL3Muons = cms.EDProducer( "L3MuonProducer",
	    ServiceParameters = cms.PSet(
	      Propagators = cms.untracked.vstring( 'hltESPSmartPropagatorAny',
	        'SteppingHelixPropagatorAny',
	        'hltESPSmartPropagator',
	        'hltESPSteppingHelixPropagatorOpposite' ),
	      RPCLayers = cms.bool( True ),
	      UseMuonNavigation = cms.untracked.bool( True )
	    ),
	    L3TrajBuilderParameters = cms.PSet(
	      ScaleTECyFactor = cms.double( -1.0 ),
	      GlbRefitterParameters = cms.PSet(
	        TrackerSkipSection = cms.int32( -1 ),
	        DoPredictionsOnly = cms.bool( False ),
	        PropDirForCosmics = cms.bool( False ),
	        HitThreshold = cms.int32( 1 ),
	        RefitFlag = cms.bool( True ),           #Usually true
	        MuonHitsOption = cms.int32( 1 ),
	        Chi2CutRPC = cms.double( 1.0 ),
	        Fitter = cms.string( "hltESPL3MuKFTrajectoryFitter" ),
	        DTRecSegmentLabel = cms.InputTag( "hltDt4DSegments" ),
	        TrackerRecHitBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
	        MuonRecHitBuilder = cms.string( "hltESPMuonTransientTrackingRecHitBuilder" ),
	        RefitDirection = cms.string( "insideOut" ),
	        CSCRecSegmentLabel = cms.InputTag( "hltCscSegments" ),
	        Chi2CutCSC = cms.double( 150.0 ),
	        Chi2CutDT = cms.double( 10.0 ),
	        RefitRPCHits = cms.bool( True ),
	        SkipStation = cms.int32( -1 ),
	        Propagator = cms.string( "hltESPSmartPropagatorAny" ),
	        TrackerSkipSystem = cms.int32( -1 ),
	        DYTthrs = cms.vint32( 30, 15 )
	      ),
	      ScaleTECxFactor = cms.double( -1.0 ),
	      TrackerRecHitBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
	      MuonRecHitBuilder = cms.string( "hltESPMuonTransientTrackingRecHitBuilder" ),
	      MuonTrackingRegionBuilder = cms.PSet(
	        Rescale_eta = cms.double( 3.0 ),
	        Rescale_phi = cms.double( 3.0 ),
	        Rescale_Dz = cms.double( 4.0 ),                 #Normally 4
	        EscapePt = cms.double( 3.0 ),                   #Normally 1.5 but it should be at least 8 for us
	        EtaR_UpperLimit_Par1 = cms.double( 0.25 ),      #Normally 0.25
	        EtaR_UpperLimit_Par2 = cms.double( 0.15 ),      #Normally 0.15
	        PhiR_UpperLimit_Par1 = cms.double( 0.6 ),       #Normally 0.6
	        PhiR_UpperLimit_Par2 = cms.double( 0.2 ),       #Normally 0.2
	        UseVertex = cms.bool( False ),                  #Normally False
	        Pt_fixed = cms.bool( False ),                   #Normally True
	        Z_fixed = cms.bool( False ),    #True for IOH
	        Phi_fixed = cms.bool( True ),   #False for IOH
	        Eta_fixed = cms.bool( True ),   #False for IOH
	        Pt_min = cms.double( 3.0 ),     #Is 0.9 for Tau; normally 8 here
	        Phi_min = cms.double( 0.1 ),
	        Eta_min = cms.double( 0.1 ),
	        DeltaZ = cms.double( 24.2 ),    #default for tau: 24.2, for old IOH: 15.9
	        DeltaR = cms.double( 0.025 ),   #This changes for different iterations. for old IOH: ?
	        DeltaEta = cms.double( 0.04 ),  #default 0.15
	        DeltaPhi = cms.double( 0.15 ),   #default 0.2
	        maxRegions = cms.int32( 2 ),
	        precise = cms.bool( True ),
	        OnDemand = cms.int32( -1 ),
	        MeasurementTrackerName = cms.InputTag( "hltESPMeasurementTracker" ),
	        beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
	        vertexCollection = cms.InputTag( "pixelVertices" ), #Warning: I am not generating colleciton. Vertex is off anyway
	        input = cms.InputTag( 'hltL2Muons','UpdatedAtVtx' )
	      ),
#	      MuonTrackingRegionBuilder = MasterMuonTrackingRegionBuilder,  #Using the master Muon ROI params - Although it is not used
	      RefitRPCHits = cms.bool( True ),
	      PCut = cms.double( 2.5 ),
	      TrackTransformer = cms.PSet(
	        DoPredictionsOnly = cms.bool( False ),
	        Fitter = cms.string( "hltESPL3MuKFTrajectoryFitter" ),
	        TrackerRecHitBuilder = cms.string( "hltESPTTRHBWithTrackAngle" ),
	        Smoother = cms.string( "hltESPKFTrajectorySmootherForMuonTrackLoader" ),
	        MuonRecHitBuilder = cms.string( "hltESPMuonTransientTrackingRecHitBuilder" ),
	        RefitDirection = cms.string( "insideOut" ),
	        RefitRPCHits = cms.bool( True ),
	        Propagator = cms.string( "hltESPSmartPropagatorAny" )
	      ),
	      GlobalMuonTrackMatcher = cms.PSet(
	        Pt_threshold1 = cms.double( 0.0 ),
	        DeltaDCut_3 = cms.double( 15.0 ),
	        MinP = cms.double( 2.5 ),
	        MinPt = cms.double( 1.0 ),
	        Chi2Cut_1 = cms.double( 50.0 ),
	        Pt_threshold2 = cms.double( 9.99999999E8 ),
	        LocChi2Cut = cms.double( 0.001 ),
	        Eta_threshold = cms.double( 1.2 ),
	        Quality_3 = cms.double( 7.0 ),
	        Quality_2 = cms.double( 15.0 ),
	        Chi2Cut_2 = cms.double( 50.0 ),
	        Chi2Cut_3 = cms.double( 200.0 ),
	        DeltaDCut_1 = cms.double( 40.0 ),
	        DeltaRCut_2 = cms.double( 0.2 ),
	        DeltaRCut_3 = cms.double( 1.0 ),
	        DeltaDCut_2 = cms.double( 10.0 ),
	        DeltaRCut_1 = cms.double( 0.1 ),
	        Propagator = cms.string( "hltESPSmartPropagator" ),
	        Quality_1 = cms.double( 20.0 )
	      ),
	      PtCut = cms.double( 1.0 ),
	      TrackerPropagator = cms.string( "SteppingHelixPropagatorAny" ),
	      tkTrajLabel = cms.InputTag( "hltBRSIter2HighPtTkMuMerged" ),      #Feed tracks from iterations into L3MTB
	      tkTrajBeamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
	      tkTrajMaxChi2 = cms.double( 9999.0 ),
	      tkTrajMaxDXYBeamSpot = cms.double( 9999.0 ),      #same cuts as old algos
	      tkTrajVertex = cms.InputTag( "pixelVertices" ),
	      tkTrajUseVertex = cms.bool( False ),
	      matchToSeeds = cms.bool( True )
	    ),
	    TrackLoaderParameters = cms.PSet(
	      PutTkTrackIntoEvent = cms.untracked.bool( False ),
	      beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
	      SmoothTkTrack = cms.untracked.bool( False ),
	      MuonSeededTracksInstance = cms.untracked.string( "L2Seeded" ),
	      Smoother = cms.string( "hltESPKFTrajectorySmootherForMuonTrackLoader" ),
	      MuonUpdatorAtVertexParameters = cms.PSet(
	        MaxChi2 = cms.double( 1000000.0 ),
	        Propagator = cms.string( "hltESPSteppingHelixPropagatorOpposite" ),
	        BeamSpotPositionErrors = cms.vdouble( 0.1, 0.1, 5.3 )
	      ),
	      VertexConstraint = cms.bool( False ),
	      DoSmoothing = cms.bool( False )   #Usually true
	    ),
	    MuonCollectionLabel = cms.InputTag( 'hltL2Muons','UpdatedAtVtx' )
	)
	
	process.hltBRSL3MuonsLinksCombination = cms.EDProducer( "L3TrackLinksCombiner",
	    labels = cms.VInputTag( 'hltL3MuonsBRSOI','BRSIterL3Muons' )
	)
	process.hltBRSL3Muons = cms.EDProducer( "L3TrackCombiner",
	    labels = cms.VInputTag( 'hltL3MuonsBRSOI','BRSIterL3Muons' )
	)
	process.hltBRSL3MuonCandidates = cms.EDProducer( "L3MuonCandidateProducer",
	    InputLinksObjects = cms.InputTag( "hltBRSL3MuonsLinksCombination" ),
	    InputObjects = cms.InputTag( "hltBRSL3Muons" ),
	    MuonPtOption = cms.string( "Tracker" )
	)

###	FOR IO Only:
#	process.hltBRSL3MuonCandidates.InputLinksObjects = cms.InputTag( "BRSIterL3Muons" )
#	process.hltBRSL3MuonCandidates.InputObjects = cms.InputTag( "BRSIterL3Muons" )
#        process.hltL3fL1sMu16orMu25L1f0L2f16QL3Filtered50Q.InputLinks = cms.InputTag( "BRSIterL3Muons" )

###    FOR OI Only:
#	process.hltBRSL3MuonCandidates.InputLinksObjects = cms.InputTag( "hltL3MuonsBRSOI" )
#	process.hltBRSL3MuonCandidates.InputObjects = cms.InputTag( "hltL3MuonsBRSOI" )
#        process.hltL3fL1sMu16orMu25L1f0L2f16QL3Filtered50Q.InputLinks = cms.InputTag( "hltL3MuonsBRSOI" )
	#############################################################
	
	####################### NEW Combo:
	process.HLTBRSIterativeTrackingHighPtTkMuIteration0 = cms.Sequence(
	 process.hltPixelLayerTriplets +
	 process.hltBRSIter0HighPtTkMuPixelTracks +
	 process.hltBRSIter0HighPtTkMuPixelSeedsFromPixelTracks +
	 process.hltBRSIter0HighPtTkMuCkfTrackCandidates +
	 process.hltBRSIter0HighPtTkMuCtfWithMaterialTracks +
	 process.hltBRSIter0HighPtTkMuTrackSelectionHighPurity
	)
	process.HLTBRSIterativeTrackingHighPtTkMuIteration2 = cms.Sequence(
	 process.hltBRSIter2HighPtTkMuClustersRefRemoval +
	 process.hltBRSIter2HighPtTkMuMaskedMeasurementTrackerEvent +
	 process.hltBRSIter2HighPtTkMuPixelLayerPairs +
	 process.hltBRSIter2HighPtTkMuPixelSeeds +
	 process.hltBRSIter2HighPtTkMuCkfTrackCandidates +
	 process.hltBRSIter2HighPtTkMuCtfWithMaterialTracks +
	 process.hltBRSIter2HighPtTkMuTrackSelectionHighPurity
	)
	process.HLTBRSIterativeTrackingHighPtTkMu = cms.Sequence(
	 process.HLTBRSIterativeTrackingHighPtTkMuIteration0 +
	 process.HLTBRSIterativeTrackingHighPtTkMuIteration2 +
	 process.hltBRSIter2HighPtTkMuMerged
	)
	
	process.HLTL3muonTkCandidateSequence = cms.Sequence(
	 process.HLTDoLocalPixelSequence + process.HLTDoLocalStripSequence +
	 process.HLTRecopixelvertexingSequence +
	 process.hltBRSOISeedsFromL2Muons +	#OIStart#off for IO
	 process.hltBRSOITrackCandidates +		#off for IO
	 process.hltBRSMuonSeededTracksOutIn +		#off for IO
	 process.hltL3MuonsBRSOI +			#off for IO
	 process.hltL2SelectorForL3OI + #OIEnd		#off for IO
	 process.HLTBRSIterativeTrackingHighPtTkMu +	#off for OI
	 process.BRSIterL3Muons 				#off for OI
	)
	
	process.HLTL3muonrecoNocandSequence = cms.Sequence(
	 process.HLTL3muonTkCandidateSequence
	 + process.hltBRSL3MuonsLinksCombination	#off for IO or OI only
	 + process.hltBRSL3Muons			#off for IO or OI only
	)
	process.HLTL3muonrecoSequence = cms.Sequence(
	 process.HLTL3muonrecoNocandSequence
	 + process.hltBRSL3MuonCandidates
	)

	return process
