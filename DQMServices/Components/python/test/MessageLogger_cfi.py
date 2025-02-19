import FWCore.ParameterSet.Config as cms

MessageLogger = cms.Service("MessageLogger",
    cout = cms.untracked.PSet(
        lineLength = cms.untracked.int32(132),
        threshold = cms.untracked.string('INFO'),
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        FwkReport = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        noLineBreaks = cms.untracked.bool(True)
    ),
    cerr = cms.untracked.PSet(
        lineLength = cms.untracked.int32(132),
        threshold = cms.untracked.string('WARNING'),
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        FwkReport = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        noLineBreaks = cms.untracked.bool(True)
    ),
    FrameworkJobReport = cms.untracked.PSet(
        FwkJob = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        )
    ),
    destinations = cms.untracked.vstring('cout', 
        'cerr'),
    categories = cms.untracked.vstring('FwkJob', 
        'MEtoEDMConverter_MEtoEDMConverter', 
        'MEtoEDMConverter_endJob', 
        'MEtoEDMConverter_beginRun', 
        'MEtoEDMConverter_endRun', 
        'EDMtoMEConverter_EDMtoMEConverter', 
        'EDMtoMEConverter_endJob', 
        'EDMtoMEConverter_beginRun', 
        'EDMtoMEConverter_endRun', 
        'ScheduleExecutionFailure', 
        'EventSetupDependency', 
        'Root_Warning', 
        'Root_Error'),
    fwkJobReports = cms.untracked.vstring('FrameworkJobReport')
)


