import FWCore.ParameterSet.Config as cms

process = cms.Process('HLT')

process.options.wantSummary = True

process.load('FWCore.MessageService.MessageLogger_cfi')

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

# load conditions from the global tag
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')

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

process.Path_1  = cms.Path(process.scale_1)
process.Path_2  = cms.Path(process.scale_2)
process.Path_3  = cms.Path(process.scale_3)
process.L1_Path = cms.Path(process.success)
# Path names containing a special keyword (TRUE, FALSE, AND, OR, NOT)
process.AlwaysNOTFalse = cms.Path(process.success)
process.AlwaysFALSE    = cms.Path(process.fail)

# define the TriggerResultsFilters based on the status of the previous paths
from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter as _triggerResultsFilter
triggerResultsFilter = _triggerResultsFilter.clone( usePathStatus = True )

# accept if 'Path_1' succeeds
process.filter_1 = triggerResultsFilter.clone(
    triggerConditions =  ( 'Path_1', ),
    l1tResults = '',
    throw = False
)

# accept if 'Path_2' succeeds
process.filter_2 = triggerResultsFilter.clone(
    triggerConditions = ( 'Path_2', ),
    l1tResults = '',
    throw = False
)

# accept if 'Path_3' succeeds
process.filter_3 = triggerResultsFilter.clone(
    triggerConditions = ( 'Path_3', ),
    l1tResults = '',
    throw = False
)

# accept if any path succeeds (explicit OR)
process.filter_any_or = triggerResultsFilter.clone(
    triggerConditions = ( 'Path_1', 'Path_2', 'Path_3' ),
    l1tResults = '',
    throw = False
)

# accept if 'Path_1' succeeds, prescaled by 15
process.filter_1_pre = triggerResultsFilter.clone(
    triggerConditions =  ( '(Path_1) / 15', ),
    l1tResults = '',
    throw = False
)

# accept if 'Path_1' prescaled by 15 does not succeed
process.filter_not_1_pre = triggerResultsFilter.clone(
    triggerConditions =  ( 'NOT (Path_1 / 15)', ),
    l1tResults = '',
    throw = False
)

# accept if 'Path_2' succeeds, prescaled by 10
process.filter_2_pre = triggerResultsFilter.clone(
    triggerConditions = ( '(Path_2 / 10)', ),
    l1tResults = '',
    throw = False
)

# accept if any path succeeds, with different prescales (explicit OR, prescaled)
process.filter_any_pre = triggerResultsFilter.clone(
    triggerConditions = ( 'Path_1 / 15', 'Path_2 / 10', 'Path_3 / 6', ),
    l1tResults = '',
    throw = False
)

# equivalent of filter_any_pre using NOT operator twice
process.filter_any_pre_doubleNOT = triggerResultsFilter.clone(
    triggerConditions = ( 'NOT NOT (Path_1 / 15 OR Path_2 / 10 OR Path_3 / 6)', ),
    l1tResults = '',
    throw = False
)

# opposite of filter_any_pre without whitespaces where possible
process.filter_not_any_pre = triggerResultsFilter.clone(
    triggerConditions = ( 'NOT(Path_1/15)AND(NOT Path_2/10)AND(NOT Path_3/6)', ),
    l1tResults = '',
    throw = False
)

# accept if any path succeeds (wildcard, '*')
process.filter_any_star = triggerResultsFilter.clone(
    triggerConditions = ( 'Path_*', ),
    l1tResults = '',
    throw = False
)

# accept if any path succeeds (wildcard, '?')
process.filter_any_question = triggerResultsFilter.clone(
    triggerConditions = ( 'Path_?', ),
    l1tResults = '',
    throw = False
)

# accept if any path succeeds (double wildcard, '*_?')
process.filter_any_starquestion = triggerResultsFilter.clone(
    triggerConditions = ( '*_?', ),
    l1tResults = '',
    throw = False
)

# accept if all path succeed (explicit AND)
process.filter_all_explicit = triggerResultsFilter.clone(
    triggerConditions = ( 'Path_1 AND Path_2 AND Path_3', ),
    l1tResults = '',
    throw = False
)

# wrong path name (explicit)
process.filter_wrong_name = triggerResultsFilter.clone(
    triggerConditions = ( 'Wrong', ),
    l1tResults = '',
    throw = False
)

# wrong path name (wildcard)
process.filter_wrong_pattern = triggerResultsFilter.clone(
    triggerConditions = ( '*_Wrong', ),
    l1tResults = '',
    throw = False
)

# empty path list
process.filter_empty_pattern = triggerResultsFilter.clone(
    triggerConditions = ( ),
    l1tResults = '',
    throw = False
)

# L1-like path name
process.filter_l1path_pattern = triggerResultsFilter.clone(
    triggerConditions = ( 'L1_Path', ),
    l1tResults = '',
    throw = False
)

# real L1 trigger
process.filter_l1singlemuopen_pattern = triggerResultsFilter.clone(
    triggerConditions = ( 'L1_SingleMuOpen', ),
    l1tResults = '',
    throw = False
)

# TRUE
process.filter_true_pattern = triggerResultsFilter.clone(
    triggerConditions = ( 'TRUE', ),
    l1tResults = '',
    throw = False
)

# FALSE
process.filter_false_pattern = triggerResultsFilter.clone(
    triggerConditions = ( 'FALSE', ),
    l1tResults = '',
    throw = False
)

# Path name containing special keyword NOT
process.filter_AlwaysNOTFalse_pattern = triggerResultsFilter.clone(
    triggerConditions = ( 'AlwaysNOTFalse', ),
    l1tResults = '',
    throw = False
)

# Path name containing special keyword FALSE
process.filter_NOTAlwaysFALSE_pattern = triggerResultsFilter.clone(
    triggerConditions = ( 'NOT AlwaysFALSE', ),
    l1tResults = '',
    throw = False
)


process.Check_1 = cms.Path( process.filter_1 )
process.Check_2 = cms.Path( process.filter_2 )
process.Check_3 = cms.Path( process.filter_3 )

process.Check_All_Explicit = cms.Path( process.filter_all_explicit )

process.Check_Any_Or   = cms.Path( process.filter_any_or )
process.Check_Any_Star = cms.Path( process.filter_any_star )

process.Check_1_Pre     = cms.Path( process.filter_1_pre )
process.Check_NOT_1_Pre = cms.Path( process.filter_not_1_pre )
process.Check_2_Pre     = cms.Path( process.filter_2_pre )
process.Check_Any_Pre   = cms.Path( process.filter_any_pre )
process.Check_Any_Pre_DoubleNOT = cms.Path( process.filter_any_pre_doubleNOT )
process.Check_Not_Any_Pre = cms.Path( process.filter_not_any_pre )

process.Check_Any_Question        = cms.Path( process.filter_any_question )
process.Check_Any_StarQuestion    = cms.Path( process.filter_any_starquestion )
process.Check_Wrong_Name          = cms.Path( process.filter_wrong_name )
process.Check_Wrong_Pattern       = cms.Path( process.filter_wrong_pattern )
process.Check_Not_Wrong_Pattern   = cms.Path( ~ process.filter_wrong_pattern )
process.Check_Empty_Pattern       = cms.Path( process.filter_empty_pattern )
process.Check_L1Path_Pattern      = cms.Path( process.filter_l1path_pattern )
process.Check_L1Singlemuopen_Pattern = cms.Path( process.filter_l1singlemuopen_pattern )
process.Check_True_Pattern        = cms.Path( process.filter_true_pattern )
process.Check_False_Pattern       = cms.Path( process.filter_false_pattern )
process.Check_AlwaysNOTFalse_Pattern = cms.Path( process.filter_AlwaysNOTFalse_pattern )
process.Check_NOTAlwaysFALSE_Pattern = cms.Path( process.filter_NOTAlwaysFALSE_pattern )

# define an EndPath to analyze all other path results
process.hltTrigReport = cms.EDAnalyzer( 'HLTrigReport',
    HLTriggerResults = cms.InputTag( 'TriggerResults', '', 'HLT' )
)
process.HLTAnalyzerEndpath = cms.EndPath( process.hltTrigReport )
