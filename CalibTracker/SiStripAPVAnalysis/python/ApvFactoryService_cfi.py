import FWCore.ParameterSet.Config as cms

ApvFactoryService = cms.Service("ApvFactoryService",

    #CalculatorAlgorithm = cms.string('MIX'),
    CMType    = cms.string('Median'),
    useDB     = cms.bool(False),

    CalculatorAlgorithm    = cms.string('TT6'),
    NumCMstripsInGroup     = cms.int32(128),
    MaskCalculationFlag    = cms.int32(1),
    MaskNoiseCut           = cms.double(6.0),                                
    MaskDeadCut            = cms.double(0.7),
    MaskTruncationCut      = cms.double(0.05),
    CutToAvoidSignal       = cms.double(3.0),
                                
    NumberOfEventsForInit      = cms.int32(10),
    NumberOfEventsForIteration = cms.int32(100)
)


