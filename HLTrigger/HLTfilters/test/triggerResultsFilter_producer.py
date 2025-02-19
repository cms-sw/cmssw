import FWCore.ParameterSet.Config as cms

process = cms.Process('HLT')

# define the Prescaler service, and set the scale factors
process.PrescaleService = cms.Service('PrescaleService',
    prescaleTable = cms.VPSet(
        cms.PSet(
            pathName  = cms.string('HLT_Path_1'),
            prescales = cms.vuint32( 2 )
        ),
        cms.PSet(
            pathName  = cms.string('HLT_Path_2'),
            prescales = cms.vuint32( 3 )
        ),
        cms.PSet(
            pathName  = cms.string('HLT_Path_3'),
            prescales = cms.vuint32( 5 )
        )
    ),
    lvl1Labels = cms.vstring('any'),
    lvl1DefaultLabel = cms.untracked.string('any')
)    

# define an empty source, and ask for 100 events
process.source = cms.Source('EmptySource')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)

# define 3 prescalers, one per path
process.scale_1 = cms.EDFilter('HLTPrescaler')
process.scale_2 = cms.EDFilter('HLTPrescaler')
process.scale_3 = cms.EDFilter('HLTPrescaler')
process.fail    = cms.EDFilter('HLTBool', result = cms.bool(False))
process.success = cms.EDFilter('HLTBool', result = cms.bool(True))

process.HLT_Path_1 = cms.Path(process.scale_1)
process.HLT_Path_2 = cms.Path(process.scale_2)
process.HLT_Path_3 = cms.Path(process.scale_3)
process.HLT_True   = cms.Path(process.success)
process.HLT_False  = cms.Path(process.fail)

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
