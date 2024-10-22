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
process.L1_Path = cms.Path(process.success)
# Path names containing a special keyword (TRUE, FALSE, AND, OR, NOT)
process.AlwaysNOTFalse = cms.Path(process.success)
process.AlwaysFALSE    = cms.Path(process.fail)

# define the TriggerResultsFilters based on the status of the previous paths
from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as _trigResFilter
_triggerResultsFilter = _trigResFilter.clone(
  usePathStatus = True,
  hltResults = '',
  l1tResults = ''
)

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
    triggerConditions = ( 'Path_*', )
)

# accept if any path succeeds (wildcard, '?')
process.filter_any_question = _triggerResultsFilter.clone(
    triggerConditions = ( 'Path_?', )
)

# accept if any path succeeds (double wildcard, '*_?')
process.filter_any_starquestion = _triggerResultsFilter.clone(
    triggerConditions = ( '*_?', )
)

# accept if all path succeed (explicit AND)
process.filter_all_explicit = _triggerResultsFilter.clone(
    triggerConditions = ( 'Path_1 AND Path_2 AND Path_3', )
)

# wrong path name (explicit)
process.filter_wrong_name = _triggerResultsFilter.clone(
    triggerConditions = ( 'Wrong', ),
    throw = False # do not throw, and return False for every event
)

# wrong path name (wildcard)
process.filter_wrong_pattern = _triggerResultsFilter.clone(
    triggerConditions = ( '*_Wrong', ),
    throw = False # do not throw, and return False for every event
)

# empty path list
process.filter_empty_pattern = _triggerResultsFilter.clone(
    triggerConditions = ( )
)

# L1-like path name
process.filter_l1path_pattern = _triggerResultsFilter.clone(
    # if usePathStatus=True, this returns False for every event without throwing exceptions,
    # because patterns starting with "L1_" are used exclusively to check the L1-Trigger decisions
    triggerConditions = ( 'L1_Path', )
)

# real L1 trigger
process.filter_l1singlemuopen_pattern = _triggerResultsFilter.clone(
    # if usePathStatus=True, this returns False for every event without throwing exceptions,
    # because patterns starting with "L1_" are used exclusively to check the L1-Trigger decisions
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

# Path name containing special keyword NOT
process.filter_AlwaysNOTFalse_pattern = _triggerResultsFilter.clone(
    triggerConditions = ( 'AlwaysNOTFalse', )
)

# Path name containing special keyword FALSE
process.filter_NOTAlwaysFALSE_pattern = _triggerResultsFilter.clone(
    triggerConditions = ( 'NOT AlwaysFALSE', )
)


process.Check_1 = cms.Path( process.filter_1 )
process.Check_2 = cms.Path( process.filter_2 )
process.Check_3 = cms.Path( process.filter_3 )

process.Check_All_Explicit = cms.Path( process.filter_all_explicit )

process.Check_Any_Or   = cms.Path( process.filter_any_or )
process.Check_Any_Star = cms.Path( process.filter_any_star )

process.Check_1_Pre             = cms.Path( process.filter_1_pre )
process.Check_1_Pre_With_Masks1 = cms.Path( process.filter_1_pre_with_masks1 )
process.Check_1_Pre_With_Masks2 = cms.Path( process.filter_1_pre_with_masks2 )
process.Check_NOT_1_Pre         = cms.Path( process.filter_not_1_pre )
process.Check_2_Pre             = cms.Path( process.filter_2_pre )
process.Check_Any_Pre           = cms.Path( process.filter_any_pre )
process.Check_Any_Pre_DoubleNOT = cms.Path( process.filter_any_pre_doubleNOT )
process.Check_Not_Any_Pre       = cms.Path( process.filter_not_any_pre )
process.Check_1xor2_withoutXOR  = cms.Path( process.filter_1xor2_withoutXOR )
process.Check_1xor2_withXOR     = cms.Path( process.filter_1xor2_withXOR )

process.Check_Any_Question           = cms.Path( process.filter_any_question )
process.Check_Any_StarQuestion       = cms.Path( process.filter_any_starquestion )
process.Check_Wrong_Name             = cms.Path( process.filter_wrong_name )
process.Check_Wrong_Pattern          = cms.Path( process.filter_wrong_pattern )
process.Check_Not_Wrong_Pattern      = cms.Path( ~ process.filter_wrong_pattern )
process.Check_Empty_Pattern          = cms.Path( process.filter_empty_pattern )
process.Check_L1Path_Pattern         = cms.Path( process.filter_l1path_pattern )
process.Check_L1Singlemuopen_Pattern = cms.Path( process.filter_l1singlemuopen_pattern )
process.Check_True_Pattern           = cms.Path( process.filter_true_pattern )
process.Check_False_Pattern          = cms.Path( process.filter_false_pattern )
process.Check_AlwaysNOTFalse_Pattern = cms.Path( process.filter_AlwaysNOTFalse_pattern )
process.Check_NOTAlwaysFALSE_Pattern = cms.Path( process.filter_NOTAlwaysFALSE_pattern )

# define an EndPath to analyze all other path results
process.hltTrigReport = cms.EDAnalyzer( 'HLTrigReport',
    HLTriggerResults = cms.InputTag( 'TriggerResults', '', '@currentProcess' )
)
process.HLTAnalyzerEndpath = cms.EndPath( process.hltTrigReport )
