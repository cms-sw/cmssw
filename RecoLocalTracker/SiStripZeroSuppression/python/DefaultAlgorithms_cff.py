import FWCore.ParameterSet.Config as cms

DefaultAlgorithms = cms.PSet(

    ## Pedestal subtraction ----------------
    PedestalSubtractionFedMode = cms.bool(False),
    SiStripFedZeroSuppressionMode = cms.uint32(4),

    ## Baseline finder ---------------------
    ##Supported CMN modes: Median, Percentile, IteratedMedian, TT6, FastLinear
    #CommonModeNoiseSubtractionMode = cms.string('Median'), 

	
	CommonModeNoiseSubtractionMode = cms.string("IteratedMedian"),
    

    #CutToAvoidSignal = cms.double(3.0), ## for TT6
    
    #Percentile = cms.double(25.0),      ## for Percentile

    CutToAvoidSignal = cms.double(2.0), ## for IteratedMedian
    Iterations = cms.int32(3),          ##
    
		

    ## APV restoration ---------------------
    APVInspectMode = cms.string("BaselineFollower"),
    APVRestoreMode = cms.string("BaselineFollower"),
	ForceNoRestore = cms.bool(False),
    SelfSelectRestoreAlgo = cms.bool(False),
    useRealMeanCM = cms.bool(False),
	Fraction = cms.double(0.2),
    Deviation = cms.uint32(25),
    restoreThreshold = cms.double(0.5),
	DeltaCMThreshold = cms.uint32(20),
    nSigmaNoiseDerTh = cms.uint32(4),         # threshold for rejecting hit strips, nSigma the noise
    consecThreshold = cms.uint32(5),         # minimum length of flat region
    hitStripThreshold = cms.uint32(40),      # height above median when strip is definitely a hit
    nSmooth = cms.uint32(9),                 # for smoothing and local minimum determination (odd number)
    minStripsToFit = cms.uint32(4),          # minimum strips to try spline algo (otherwise default to median)
    distortionThreshold = cms.uint32(40),     # it the difference between the max and the min of the baseline in an APV. it define how flat is the APV
    nSaturatedStrip = cms.uint32(2),
	## Zero suppression --------------------
    TruncateInSuppressor = cms.bool(True)
    
    )
