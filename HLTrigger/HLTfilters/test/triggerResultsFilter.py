import FWCore.ParameterSet.Config as cms

process = cms.Process('TEST')

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)
process.load('FWCore.MessageService.MessageLogger_cfi')
#process.MessageLogger.cerr.INFO = cms.untracked.PSet(
#    reportEvery = cms.untracked.int32(1), # every!
#    limit = cms.untracked.int32(-1)       # no limit!
#    )
#process.MessageLogger.cerr.FwkReport.reportEvery = 10 # only report every 10th event start
#process.MessageLogger.cerr_stats.threshold = 'INFO' # also info in statistics

# load conditions from the global tag
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff')
from Configuration.AlCa.GlobalTag_condDBv2 import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')

# read back the trigger decisions
process.source = cms.Source('PoolSource',
    fileNames = cms.untracked.vstring('file:trigger.root')
)

import HLTrigger.HLTfilters.triggerResultsFilter_cfi as hlt

# accept if 'Path_1' succeeds
process.filter_1 = hlt.triggerResultsFilter.clone(
    triggerConditions =  ( 'Path_1', ),
    l1tResults = '',
    throw = False
    )

# accept if 'Path_2' succeeds
process.filter_2 = hlt.triggerResultsFilter.clone(
    triggerConditions = ( 'Path_2', ),
    l1tResults = '',
    throw = False
    )

# accept if 'Path_3' succeeds
process.filter_3 = hlt.triggerResultsFilter.clone(
    triggerConditions = ( 'Path_3', ),
    l1tResults = '',
    throw = False
    )

# accept if any path succeeds (explicit OR)
process.filter_any_or = hlt.triggerResultsFilter.clone(
    triggerConditions = ( 'Path_1', 'Path_2', 'Path_3' ),
    l1tResults = '',
    throw = False
    )

# accept if 'Path_1' succeeds, prescaled by 2
process.filter_1_pre = hlt.triggerResultsFilter.clone(
    triggerConditions =  ( '(Path_1) / 15', ),
    l1tResults = '',
    throw = False
    )

# accept if 'Path_2' succeeds, prescaled by 10
process.filter_2_pre = hlt.triggerResultsFilter.clone(
    triggerConditions = ( '(Path_2 / 10)', ),
    l1tResults = '',
    throw = False
    )

# accept if any path succeeds, with different prescales (explicit OR, prescaled)
process.filter_any_pre = hlt.triggerResultsFilter.clone(
    triggerConditions = ( 'Path_1 / 15', 'Path_2 / 10', 'Path_3 / 6', ),
    l1tResults = '',
    throw = False
    )

# accept if any path succeeds (wildcard, '*')
process.filter_any_star = hlt.triggerResultsFilter.clone(
    triggerConditions = ( '*', ),
    l1tResults = '',
    throw = False
    )

# accept if any path succeeds (wildcard, twice '*')
process.filter_any_doublestar = hlt.triggerResultsFilter.clone(
    triggerConditions = ( '*_*', ),
    l1tResults = '',
    throw = False
    )


# accept if any path succeeds (wildcard, '?')
process.filter_any_question = hlt.triggerResultsFilter.clone(
    triggerConditions = ( 'Path_?', ),
    l1tResults = '',
    throw = False
    )

# accept if all path succeed (explicit AND)
process.filter_all_explicit = hlt.triggerResultsFilter.clone(
    triggerConditions = ( 'Path_1 AND Path_2 AND Path_3', ),
    l1tResults = '',
    throw = False
)

# wrong path name (explicit)
process.filter_wrong_name = hlt.triggerResultsFilter.clone(
    triggerConditions = ( 'Wrong', ),
    l1tResults = '',
    throw = False
)

# wrong path name (wildcard)
process.filter_wrong_pattern = hlt.triggerResultsFilter.clone(
    triggerConditions = ( '*_Wrong', ),
    l1tResults = '',
    throw = False
)

# empty path list
process.filter_empty_pattern = hlt.triggerResultsFilter.clone(
    triggerConditions = ( ),
    l1tResults = '',
    throw = False
)

# L1-like path name
process.filter_l1path_pattern = hlt.triggerResultsFilter.clone(
    triggerConditions = ( 'L1_Path', ),
    l1tResults = '',
    throw = False
)

# real L1 trigger
process.filter_l1singlemuopen_pattern = hlt.triggerResultsFilter.clone(
    triggerConditions = ( 'L1_SingleMuOpen', ),
    l1tResults = '',
    throw = False
)

# TRUE
process.filter_true_pattern = hlt.triggerResultsFilter.clone(
    triggerConditions = ( 'TRUE', ),
    l1tResults = '',
    throw = False
)

# FALSE
process.filter_false_pattern = hlt.triggerResultsFilter.clone(
    triggerConditions = ( 'FALSE', ),
    l1tResults = '',
    throw = False
)


process.path_1 = cms.Path( process.filter_1 )
process.path_2 = cms.Path( process.filter_2 )
process.path_3 = cms.Path( process.filter_3 )

process.path_all_explicit = cms.Path( process.filter_all_explicit )

process.path_any_or   = cms.Path( process.filter_any_or )
process.path_any_star = cms.Path( process.filter_any_star )

process.path_1_pre    = cms.Path( process.filter_1_pre )
process.path_2_pre    = cms.Path( process.filter_2_pre )
process.path_any_pre  = cms.Path( process.filter_any_pre ) 

process.path_any_doublestar      = cms.Path( process.filter_any_doublestar )
process.path_any_question        = cms.Path( process.filter_any_question )
process.path_wrong_name          = cms.Path( process.filter_wrong_name )
process.path_wrong_pattern       = cms.Path( process.filter_wrong_pattern )
process.path_not_wrong_pattern   = cms.Path( ~ process.filter_wrong_pattern )
process.path_empty_pattern       = cms.Path( process.filter_empty_pattern )
process.path_l1path_pattern      = cms.Path( process.filter_l1path_pattern )
process.path_l1singlemuopen_pattern = cms.Path( process.filter_l1singlemuopen_pattern )
process.path_true_pattern        = cms.Path( process.filter_true_pattern )
process.path_false_pattern       = cms.Path( process.filter_false_pattern )

# define an EndPath to analyze all other path results
process.hltTrigReport = cms.EDAnalyzer( 'HLTrigReport',
    HLTriggerResults = cms.InputTag( 'TriggerResults','','TEST' )
)
process.HLTAnalyzerEndpath = cms.EndPath( process.hltTrigReport )
