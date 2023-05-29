import FWCore.ParameterSet.Config as cms

process = cms.Process('HLT')

process.options.wantSummary = True

process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.cerr.FwkReport.reportEvery = 100 # only report every 100th event start
process.MessageLogger.cerr.enableStatistics = False # enable "MessageLogger Summary" message
process.MessageLogger.cerr.threshold = 'INFO' # change to 'WARNING' not to show INFO-level messages
## enable reporting of INFO-level messages (default is limit=0, i.e. no messages reported)
#process.MessageLogger.cerr.INFO = cms.untracked.PSet(
#    reportEvery = cms.untracked.int32(1), # every event!
#    limit = cms.untracked.int32(-1)       # no limit!
#)

# define the Prescaler service, and set the scale factors
process.PrescaleService = cms.Service('PrescaleService',
    prescaleTable = cms.VPSet(
        cms.PSet(
            pathName  = cms.string('Path_1'),
            prescales = cms.vuint32( 2 )
        ),
        cms.PSet(
            pathName  = cms.string('Path_2'),
            prescales = cms.vuint32( 3 )
        ),
        cms.PSet(
            pathName  = cms.string('Path_3'),
            prescales = cms.vuint32( 5 )
        )
    ),
    lvl1Labels = cms.vstring('any'),
    lvl1DefaultLabel = cms.string('any')
)    

# define an empty source, and ask for 1000 events
process.source = cms.Source('EmptySource')
process.maxEvents.input = 1000

# define 3 prescalers, one per path
process.scale_1 = cms.EDFilter('HLTPrescaler')
process.scale_2 = cms.EDFilter('HLTPrescaler')
process.scale_3 = cms.EDFilter('HLTPrescaler')
process.fail    = cms.EDFilter('HLTBool', result = cms.bool(False))
process.success = cms.EDFilter('HLTBool', result = cms.bool(True))

process.Path_1  = cms.Path(process.scale_1)
process.Path_2  = cms.Path(process.scale_2)
process.Path_3  = cms.Path(process.scale_3)
process.AlwaysTrue    = cms.Path(process.success)
process.AlwaysFalse   = cms.Path(process.fail)
process.L1_Path = cms.Path(process.success)

# define and EndPath to analyze all other path results
process.hltTrigReport = cms.EDAnalyzer( 'HLTrigReport',
    HLTriggerResults = cms.InputTag( 'TriggerResults','','HLT' )
)
process.HLTAnalyzerEndpath = cms.EndPath( process.hltTrigReport )

# define the PoolOutputModule
process.poolOutput = cms.OutputModule('PoolOutputModule',
    fileName = cms.untracked.string('file:trigger.root')
)
process.output = cms.EndPath(process.poolOutput)
