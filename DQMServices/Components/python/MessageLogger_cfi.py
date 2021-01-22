import FWCore.ParameterSet.Config as cms

MessageLogger = cms.Service("MessageLogger",
    MessageLogger = cms.untracked.PSet(
        lineLength = cms.untracked.int32(132),
        MEtoEDMConverter_beginRun = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        MEtoEDMConverter_endRun = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        ScheduleExecutionFailure = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        EDMtoMEConverter_EDMtoMEConverter = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        EDMtoMEConverter_beginRun = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        noLineBreaks = cms.untracked.bool(True),
        Root_Error = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        MEtoEDMConverter_endJob = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        MEtoEDMConverter_MEtoEDMConverter = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        Root_Warning = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        EventSetupDependency = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        EDMtoMEConverter_endJob = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        EDMtoMEConverter_endRun = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        )
    ),
    cout = cms.untracked.PSet(
        lineLength = cms.untracked.int32(132),
        MEtoEDMConverter_beginRun = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        MEtoEDMConverter_endRun = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        EDMtoMEConverter_EDMtoMEConverter = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        EDMtoMEConverter_beginRun = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        noLineBreaks = cms.untracked.bool(True),
        MEtoEDMConverter_endJob = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        MEtoEDMConverter_MEtoEDMConverter = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        threshold = cms.untracked.string('INFO'),
        EDMtoMEConverter_endJob = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        EDMtoMEConverter_endRun = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        )
    ),
    cerr = cms.untracked.PSet(
        lineLength = cms.untracked.int32(132),
        EventSetupDependency = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        MEtoEDMConverter_beginRun = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        MEtoEDMConverter_endRun = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        ScheduleExecutionFailure = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        EDMtoMEConverter_EDMtoMEConverter = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        EDMtoMEConverter_beginRun = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        noLineBreaks = cms.untracked.bool(True),
        Root_Error = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        MEtoEDMConverter_endJob = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        MEtoEDMConverter_MEtoEDMConverter = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        Root_Warning = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        threshold = cms.untracked.string('WARNING'),
        EDMtoMEConverter_endJob = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        EDMtoMEConverter_endRun = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        )
    ),
    categories = cms.untracked.vstring('FwkJob', 'MEtoEDMConverter_MEtoEDMConverter', 'MEtoEDMConverter_endJob', 'MEtoEDMConverter_beginRun', 'MEtoEDMConverter_endRun', 'EDMtoMEConverter_EDMtoMEConverter', 'EDMtoMEConverter_endJob', 'EDMtoMEConverter_beginRun', 'EDMtoMEConverter_endRun', 'ScheduleExecutionFailure', 'EventSetupDependency', 'Root_Warning', 'Root_Error'),
    destinations = cms.untracked.vstring('MessageLogger', 'cout', 'cerr')
)


