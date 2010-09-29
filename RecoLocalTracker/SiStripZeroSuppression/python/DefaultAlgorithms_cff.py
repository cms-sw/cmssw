import FWCore.ParameterSet.Config as cms

DefaultAlgorithms = cms.PSet(
    PedestalSubtractionFedMode = cms.bool(True),
    SiStripFedZeroSuppressionMode = cms.uint32(4),
    TruncateInSuppressor = cms.bool(True),

    ##Supported CMN modes: Median, Percentile, IteratedMedian, TT6, FastLinear
    CommonModeNoiseSubtractionMode = cms.string('Median') 
    
    #CutToAvoidSignal = cms.double(3.0), ## for TT6
    
    #Percentile = cms.double(25.0),      ## for Percentile

    #CutToAvoidSignal = cms.double(2.0), ## for IteratedMedian
    #Iterations = cms.int32(3),          ##
    
    )
