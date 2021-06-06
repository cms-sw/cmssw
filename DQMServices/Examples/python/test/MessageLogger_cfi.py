import FWCore.ParameterSet.Config as cms

MessageLogger = cms.Service("MessageLogger",
    MessageLogger = cms.untracked.PSet(
        MEtoEDMConverter_endRun = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        ConverterTester_endJob = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        ConverterQualityTester_endRun = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        MEtoEDMConverter_MEtoEDMConverter = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        lineLength = cms.untracked.int32(132),
        PostConverterAnalyzer_PostConverterAnalyzer = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        EDMtoMEConverter_endRun = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        MEtoEDMConverter_beginRun = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        PostConverterAnalyzer_endJob = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        ConverterTester_beginRun = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        Root_Error = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        Root_Warning = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        ConverterTester_ConverterTester = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        EDMtoMEConverter_EDMtoMEConverter = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        ConverterQualityTester_ConverterQualityTester = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        ScheduleExecutionFailure = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        ConverterQualityTester_beginRun = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        PostConverterAnalyzer_endRun = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        MEtoEDMConverter_endJob = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        ConverterQualityTester_endJob = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        EventSetupDependency = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        EDMtoMEConverter_endJob = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        PostConverterAnalyzer_beginRun = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        EDMtoMEConverter_beginRun = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        noLineBreaks = cms.untracked.bool(True),
        ConverterTester_endRun = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        )
    ),
    cout = cms.untracked.PSet(
        MEtoEDMConverter_endRun = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        ConverterTester_endJob = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        ConverterQualityTester_endRun = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        MEtoEDMConverter_MEtoEDMConverter = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        threshold = cms.untracked.string('INFO'),
        lineLength = cms.untracked.int32(132),
        PostConverterAnalyzer_PostConverterAnalyzer = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        EDMtoMEConverter_endRun = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        MEtoEDMConverter_beginRun = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        PostConverterAnalyzer_endJob = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        ConverterTester_beginRun = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        ConverterTester_ConverterTester = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        EDMtoMEConverter_EDMtoMEConverter = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        ConverterQualityTester_ConverterQualityTester = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        ConverterQualityTester_beginRun = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        PostConverterAnalyzer_endRun = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        MEtoEDMConverter_endJob = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        ConverterQualityTester_endJob = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        EDMtoMEConverter_endJob = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        PostConverterAnalyzer_beginRun = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        EDMtoMEConverter_beginRun = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        noLineBreaks = cms.untracked.bool(True),
        ConverterTester_endRun = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        )
    ),
    cerr = cms.untracked.PSet(
        MEtoEDMConverter_endRun = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        ConverterTester_endJob = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        ConverterQualityTester_endRun = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        MEtoEDMConverter_MEtoEDMConverter = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        threshold = cms.untracked.string('WARNING'),
        lineLength = cms.untracked.int32(132),
        PostConverterAnalyzer_PostConverterAnalyzer = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        EDMtoMEConverter_endRun = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        MEtoEDMConverter_beginRun = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        PostConverterAnalyzer_endJob = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        ConverterTester_beginRun = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        Root_Error = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        Root_Warning = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        ConverterTester_ConverterTester = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        EDMtoMEConverter_EDMtoMEConverter = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        ConverterQualityTester_ConverterQualityTester = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        ScheduleExecutionFailure = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        ConverterQualityTester_beginRun = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        PostConverterAnalyzer_endRun = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        MEtoEDMConverter_endJob = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        ConverterQualityTester_endJob = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        EventSetupDependency = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        EDMtoMEConverter_endJob = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        PostConverterAnalyzer_beginRun = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        EDMtoMEConverter_beginRun = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        noLineBreaks = cms.untracked.bool(True),
        ConverterTester_endRun = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        )
    ),
    categories = cms.untracked.vstring('FwkJob', 
        'MEtoEDMConverter_MEtoEDMConverter', 
        'MEtoEDMConverter_endJob', 
        'MEtoEDMConverter_beginRun', 
        'MEtoEDMConverter_endRun', 
        'EDMtoMEConverter_EDMtoMEConverter', 
        'EDMtoMEConverter_endJob', 
        'EDMtoMEConverter_beginRun', 
        'EDMtoMEConverter_endRun', 
        'ConverterTester_ConverterTester', 
        'ConverterTester_endJob', 
        'ConverterTester_beginRun', 
        'ConverterTester_endRun', 
        'PostConverterAnalyzer_PostConverterAnalyzer', 
        'PostConverterAnalyzer_endJob', 
        'PostConverterAnalyzer_beginRun', 
        'PostConverterAnalyzer_endRun', 
        'ConverterQualityTester_ConverterQualityTester', 
        'ConverterQualityTester_endJob', 
        'ConverterQualityTester_beginRun', 
        'ConverterQualityTester_endRun', 
        'ScheduleExecutionFailure', 
        'EventSetupDependency', 
        'Root_Warning', 
        'Root_Error'),
    destinations = cms.untracked.vstring('MessageLogger', 
        'cout', 
        'cerr')
)

