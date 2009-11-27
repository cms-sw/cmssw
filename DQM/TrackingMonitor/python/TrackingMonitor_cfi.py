import FWCore.ParameterSet.Config as cms

TrackMon = cms.EDFilter("TrackingMonitor",

	# input tags
    TrackProducer = cms.InputTag("generalTracks"),
    SeedProducer  = cms.InputTag("newSeedFromTriplets"),
    TCProducer    = cms.InputTag("newTrackCandidateMaker"),
    AlgoName      = cms.string('GenTk'),
    beamSpot 	  = cms.InputTag("offlineBeamSpot"),                

	# output parameters
    OutputMEsInRootFile = cms.bool(False),
    OutputFileName      = cms.string('MonitorTrack.root'),
    FolderName 			= cms.string('Track/GlobalParameters'),
    BSFolderName 		= cms.string('Track/BeamSpotParameters'),

	
	# determines where to evaluate track parameters
	# options: 'default'      --> straight up track parametes
	#		   'ImpactPoint'  --> evalutate at impact point 
	#		   'InnerSurface' --> evalutate at innermost measurement state 
	#		   'OuterSurface' --> evalutate at outermost measurement state 
    MeasurementState = cms.string('ImpactPoint'),

    doTrackerSpecific 	  = cms.bool(True),
    doAllPlots 		  = cms.bool(False),                    
    doBeamSpotPlots       = cms.bool(True),
    doSeedParameterHistos = cms.bool(True),

	# paramters of the Track
	# ============================================================ 

	# chi2
    Chi2Bin = cms.int32(500),
    Chi2Max = cms.double(249.5),
    Chi2Min = cms.double(-0.5),

	# chi^2 probability
    Chi2ProbBin = cms.int32(100),
    Chi2ProbMax = cms.double(1.0),
    Chi2ProbMin = cms.double(0.0),

	# Number of Tracks per Event
    TkSizeBin = cms.int32(500),
    TkSizeMin = cms.double(-0.5),
    TkSizeMax = cms.double(499.5),

	# Number of seeds per Event
    TkSeedSizeBin = cms.int32(500),
    TkSeedSizeMin = cms.double(-0.5),
    TkSeedSizeMax = cms.double(999.5),

	# num rec hits
    RecHitBin = cms.int32(40),
    RecHitMin = cms.double(-0.5),
    RecHitMax = cms.double(39.5),

	# num rec hits lost
    RecLostBin = cms.int32(20),
    RecLostMin = cms.double(-0.5),
    RecLostMax = cms.double(19.5),

	# num layers 
    RecLayBin = cms.int32(20),
    RecLayMin = cms.double(-0.5),
    RecLayMax = cms.double(19.5),
                        
	# Track |p|	
    TrackPBin = cms.int32(100),
    TrackPMin = cms.double(0),
    TrackPMax = cms.double(100),

	# Track pT
    TrackPtBin = cms.int32(100),
    TrackPtMin = cms.double(0.1),
    TrackPtMax = cms.double(50),

	# Track px 
    TrackPxBin = cms.int32(100),
    TrackPxMin = cms.double(-20.0),
    TrackPxMax = cms.double(20.0),

	# Track py
    TrackPyBin = cms.int32(100),
    TrackPyMin = cms.double(-20.0),
    TrackPyMax = cms.double(20.0),

	# Track pz
    TrackPzBin = cms.int32(1000),
    TrackPzMin = cms.double(-30.0),
    TrackPzMax = cms.double(30.0),
                        
	# Track |p|	error
    pErrBin = cms.int32(100),
    pErrMin = cms.double(0.0),
    pErrMax = cms.double(0.1),

	# track theta
    ThetaBin = cms.int32(100),
    ThetaMin = cms.double(0.0),
    ThetaMax = cms.double(3.2),

	# track eta
    EtaBin = cms.int32(100),
    EtaMin = cms.double(-4.0),
    EtaMax = cms.double(4.0),

	# track phi
    PhiBin = cms.int32(36),
    PhiMin = cms.double(-3.2),
    PhiMax = cms.double(3.2),

	# Track pT error
    ptErrBin = cms.int32(100),
    ptErrMin = cms.double(0.0),
    ptErrMax = cms.double(0.1),

	# Track px error
    pxErrBin = cms.int32(100),
    pxErrMin = cms.double(0.0),
    pxErrMax = cms.double(0.1),

	# Track py error
    pyErrBin = cms.int32(100),                        
    pyErrMin = cms.double(0.0),
    pyErrMax = cms.double(0.1),

	# Track pz error
    pzErrBin = cms.int32(100),
    pzErrMin = cms.double(0.0),
    pzErrMax = cms.double(0.1),

	# track eta error
    etaErrBin = cms.int32(100),
    etaErrMin = cms.double(0.0),
    etaErrMax = cms.double(0.1),

	# track phi Error
    phiErrBin = cms.int32(100),
    phiErrMin = cms.double(0.0),
    phiErrMax = cms.double(0.1),

	# Track d0 (transverse impact parameter)
    D0Bin = cms.int32(100),
    D0Max = cms.double(0.5),
    D0Min = cms.double(-0.5),                        
                        
	# PCA x position
    VXBin = cms.int32(100),
    VXMin = cms.double(-0.5),
    VXMax = cms.double(0.5),

	# PCA y position
    VYBin = cms.int32(100),
    VYMin = cms.double(-0.5),
    VYMax = cms.double(0.5),

	# PCA z position
    VZBin = cms.int32(100),
    VZMin = cms.double(-30.0),
    VZMax = cms.double(30.0),

	# PCA x position for 2D plot
    X0Bin = cms.int32(100),
    X0Min = cms.double(-0.5),
    X0Max = cms.double(0.5),

	# PCA y position for 2D plot
    Y0Bin = cms.int32(100),
    Y0Min = cms.double(-0.5),
    Y0Max = cms.double(0.5),

	# PCA z position for 2D plot
    Z0Bin = cms.int32(100),
    Z0Min = cms.double(-30.0),
    Z0Max = cms.double(30.0),

    TTRHBuilder = cms.string('WithTrackAngle')
)


