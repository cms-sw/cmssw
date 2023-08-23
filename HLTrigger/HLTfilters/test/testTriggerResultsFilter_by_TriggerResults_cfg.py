import FWCore.ParameterSet.Config as cms

process = cms.Process('TEST')

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

# read back the trigger decisions
process.source = cms.Source('PoolSource',
    fileNames = cms.untracked.vstring('file:trigger.root')
)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as _trigResFilter
_triggerResultsFilter = _trigResFilter.clone( l1tResults = '' )

# accept if 'Path_1' succeeds
process.filter_1 = _triggerResultsFilter.clone(
    triggerConditions =  ( 'Path_1', )
)

# accept if 'Path_2' succeeds
process.filter_2 = _triggerResultsFilter.clone(
    triggerConditions = ( 'Path_2', )
)

# accept if 'Path_3' succeeds
process.filter_3 = _triggerResultsFilter.clone(
    triggerConditions = ( 'Path_3', )
)

# accept if any path succeeds (explicit OR)
process.filter_any_or = _triggerResultsFilter.clone(
    triggerConditions = ( 'Path_1', 'Path_2', 'Path_3' )
)

# accept if 'Path_1' succeeds, prescaled by 15
process.filter_1_pre = _triggerResultsFilter.clone(
    triggerConditions =  ( '(Path_1) / 15', )
)

# accept if 'Path_1' succeeds, prescaled by 15
# masking Path_2 (equivalent to filter_1_pre)
process.filter_1_pre_with_masks1 = _triggerResultsFilter.clone(
    triggerConditions =  ( '(Path_1 / 15 OR Path_2) MASKING Path_2', )
)

# accept if 'Path_1' succeeds, prescaled by 15
# masking Path_2 and Path_3 (equivalent to filter_1_pre)
process.filter_1_pre_with_masks2 = _triggerResultsFilter.clone(
    triggerConditions =  ( '(Path_? / 15) MASKING Path_2 MASKING Path_3', )
)

# accept if 'Path_1' prescaled by 15 does not succeed
process.filter_not_1_pre = _triggerResultsFilter.clone(
    triggerConditions =  ( 'NOT (Path_1 / 15)', )
)

# accept if 'Path_2' succeeds, prescaled by 10
process.filter_2_pre = _triggerResultsFilter.clone(
    triggerConditions = ( '(Path_2 / 10)', )
)

# accept if any path succeeds, with different prescales (explicit OR, prescaled)
process.filter_any_pre = _triggerResultsFilter.clone(
    triggerConditions = ( 'Path_1 / 15', 'Path_2 / 10', 'Path_3 / 6', )
)

# equivalent of filter_any_pre using NOT operator twice
process.filter_any_pre_doubleNOT = _triggerResultsFilter.clone(
    triggerConditions = ( 'NOT NOT (Path_1 / 15 OR Path_2 / 10 OR Path_3 / 6)', )
)

# opposite of filter_any_pre without whitespaces where possible
process.filter_not_any_pre = _triggerResultsFilter.clone(
    triggerConditions = ( 'NOT(Path_1/15)AND(NOT Path_2/10)AND(NOT Path_3/6)', )
)

# accept if Path_1 and Path_2 have different results (XOR) without using XOR operator
process.filter_1xor2_withoutXOR = _triggerResultsFilter.clone(
    triggerConditions = ( 'Path_1 AND NOT Path_2', 'NOT Path_1 AND Path_2', )
)

# accept if Path_1 and Path_2 have different results (XOR) using XOR operator
process.filter_1xor2_withXOR = _triggerResultsFilter.clone(
    triggerConditions = ( 'Path_1 XOR Path_2', )
)

# accept if any path succeeds (wildcard, '*')
process.filter_any_star = _triggerResultsFilter.clone(
    triggerConditions = ( '*', )
)

# accept if any path succeeds (wildcard, twice '*')
process.filter_any_doublestar = _triggerResultsFilter.clone(
    triggerConditions = ( '*_*', )
)

# accept if any path succeeds (wildcard, '?')
process.filter_any_question = _triggerResultsFilter.clone(
    triggerConditions = ( 'Path_?', )
)

# accept if all path succeed (explicit AND)
process.filter_all_explicit = _triggerResultsFilter.clone(
    triggerConditions = ( 'Path_1 AND Path_2 AND Path_3', )
)

# wrong path name (explicit)
process.filter_wrong_name = _triggerResultsFilter.clone(
    triggerConditions = ( 'Wrong', ),
    throw = False
)

# wrong path name (wildcard)
process.filter_wrong_pattern = _triggerResultsFilter.clone(
    triggerConditions = ( '*_Wrong', ),
    throw = False
)

# empty path list
process.filter_empty_pattern = _triggerResultsFilter.clone(
    triggerConditions = ( )
)

# L1-like path name
process.filter_l1path_pattern = _triggerResultsFilter.clone(
    # this returns False for every event without throwing exceptions,
    # because here l1tResults is an empty InputTag
    # (patterns starting with "L1_" are used exclusively to check the L1-Trigger decisions)
    triggerConditions = ( 'L1_Path', )
)

# real L1 trigger
process.filter_l1singlemuopen_pattern = _triggerResultsFilter.clone(
    # this returns False for every event without throwing exceptions,
    # because here l1tResults is an empty InputTag
    # (patterns starting with "L1_" are used exclusively to check the L1-Trigger decisions)
    triggerConditions = ( 'L1_SingleMuOpen', )
)

# TRUE
process.filter_true_pattern = _triggerResultsFilter.clone(
    triggerConditions = ( 'TRUE', )
)

# FALSE
process.filter_false_pattern = _triggerResultsFilter.clone(
    triggerConditions = ( 'FALSE', )
)


process.path_1 = cms.Path( process.filter_1 )
process.path_2 = cms.Path( process.filter_2 )
process.path_3 = cms.Path( process.filter_3 )

process.path_all_explicit = cms.Path( process.filter_all_explicit )

process.path_any_or   = cms.Path( process.filter_any_or )
process.path_any_star = cms.Path( process.filter_any_star )

process.path_1_pre             = cms.Path( process.filter_1_pre )
process.path_1_pre_with_masks1 = cms.Path( process.filter_1_pre_with_masks1 )
process.path_1_pre_with_masks2 = cms.Path( process.filter_1_pre_with_masks2 )
process.path_not_1_pre         = cms.Path( process.filter_not_1_pre )
process.path_2_pre             = cms.Path( process.filter_2_pre )
process.path_any_pre           = cms.Path( process.filter_any_pre )
process.path_any_pre_doubleNOT = cms.Path( process.filter_any_pre_doubleNOT )
process.path_not_any_pre       = cms.Path( process.filter_not_any_pre )
process.Check_1xor2_withoutXOR = cms.Path( process.filter_1xor2_withoutXOR )
process.Check_1xor2_withXOR    = cms.Path( process.filter_1xor2_withXOR )

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
    HLTriggerResults = cms.InputTag( 'TriggerResults', '', '@currentProcess' )
)
process.HLTAnalyzerEndpath = cms.EndPath( process.hltTrigReport )
