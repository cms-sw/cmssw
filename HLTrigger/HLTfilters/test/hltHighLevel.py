import FWCore.ParameterSet.Config as cms

process = cms.Process('TEST')

# read back the trigger decisions
process.source = cms.Source('PoolSource',
    fileNames = cms.untracked.vstring('file:trigger.root')
)

# accept if 'path_1' succeeds
process.filter_1 = cms.EDFilter('HLTHighLevel',
    TriggerResultsTag = cms.InputTag('TriggerResults', '', 'HLT'),
    HLTPaths = cms.vstring('path_1'),
    andOr = cms.bool(True),
    throw = cms.untracked.bool(False)
)

# accept if 'path_2' succeeds
process.filter_2 = cms.EDFilter('HLTHighLevel',
    TriggerResultsTag = cms.InputTag('TriggerResults', '', 'HLT'),
    HLTPaths = cms.vstring('path_2'),
    andOr = cms.bool(True),
    throw = cms.untracked.bool(False)
)

# accept if 'path_3' succeeds
process.filter_3 = cms.EDFilter('HLTHighLevel',
    TriggerResultsTag = cms.InputTag('TriggerResults', '', 'HLT'),
    HLTPaths = cms.vstring('path_3'),
    andOr = cms.bool(True),
    throw = cms.untracked.bool(False)
)

# accept if any path succeeds (implicit)
process.filter_any_implicit = cms.EDFilter('HLTHighLevel',
    TriggerResultsTag = cms.InputTag('TriggerResults', '', 'HLT'),
    HLTPaths = cms.vstring(),
    andOr = cms.bool(True),
    throw = cms.untracked.bool(False)
)

# accept if any path succeeds (explicit)
process.filter_any_explicit = cms.EDFilter('HLTHighLevel',
    TriggerResultsTag = cms.InputTag('TriggerResults', '', 'HLT'),
    HLTPaths = cms.vstring('path_1', 'path_2', 'path_3'),
    andOr = cms.bool(True),
    throw = cms.untracked.bool(False)
)

# accept if any path succeeds (wildcard, '*')
process.filter_any_star = cms.EDFilter('HLTHighLevel',
    TriggerResultsTag = cms.InputTag('TriggerResults', '', 'HLT'),
    HLTPaths = cms.vstring('p*'),
    andOr = cms.bool(True),
    throw = cms.untracked.bool(False)
)

# accept if any path succeeds (wildcard, '?')
process.filter_any_question = cms.EDFilter('HLTHighLevel',
    TriggerResultsTag = cms.InputTag('TriggerResults', '', 'HLT'),
    HLTPaths = cms.vstring('path_?'),
    andOr = cms.bool(True),
    throw = cms.untracked.bool(False)
)

# accept if all path succeed (implicit)
process.filter_all_implicit = cms.EDFilter('HLTHighLevel',
    TriggerResultsTag = cms.InputTag('TriggerResults', '', 'HLT'),
    HLTPaths = cms.vstring(),
    andOr = cms.bool(False),
    throw = cms.untracked.bool(False)
)

# accept if all path succeed (explicit)
process.filter_all_explicit = cms.EDFilter('HLTHighLevel',
    TriggerResultsTag = cms.InputTag('TriggerResults', '', 'HLT'),
    HLTPaths = cms.vstring('path_1', 'path_2', 'path_3'),
    andOr = cms.bool(False),
    throw = cms.untracked.bool(False)
)

# accept if all path succeed (wildcard, '*')
process.filter_all_star = cms.EDFilter('HLTHighLevel',
    TriggerResultsTag = cms.InputTag('TriggerResults', '', 'HLT'),
    HLTPaths = cms.vstring('p*'),
    andOr = cms.bool(False),
    throw = cms.untracked.bool(False)
)

# accept if all path succeed (wildcard, '?')
process.filter_all_question = cms.EDFilter('HLTHighLevel',
    TriggerResultsTag = cms.InputTag('TriggerResults', '', 'HLT'),
    HLTPaths = cms.vstring('path_?'),
    andOr = cms.bool(False),
    throw = cms.untracked.bool(False)
)

# wrong L1 name (explicit)
process.filter_wrong_name = cms.EDFilter('HLTHighLevel',
    TriggerResultsTag = cms.InputTag('TriggerResults', '', 'HLT'),
    HLTPaths = cms.vstring('path_wrong'),
    andOr = cms.bool(True),
    throw = cms.untracked.bool(False)
)

# wrong L1 name (wildcard)
process.filter_wrong_pattern = cms.EDFilter('HLTHighLevel',
    TriggerResultsTag = cms.InputTag('TriggerResults', '', 'HLT'),
    HLTPaths = cms.vstring('*_wrong'),
    andOr = cms.bool(True),
    throw = cms.untracked.bool(False)
)

process.end_1 = cms.Path( process.filter_1 )
process.end_2 = cms.Path( process.filter_2 )
process.end_3 = cms.Path( process.filter_3 )

process.end_any_implicit = cms.Path( process.filter_any_implicit )
process.end_any_explicit = cms.Path( process.filter_any_explicit )
process.end_any_star     = cms.Path( process.filter_any_star )
process.end_any_question = cms.Path( process.filter_any_question )
#process.end_any_filter   = cms.Path( ~ ( ~ process.filter_1 + ~ process.filter_2 + ~ process.filter_3) )

process.end_all_implicit = cms.Path( process.filter_all_implicit )
process.end_all_explicit = cms.Path( process.filter_all_explicit )
process.end_all_star     = cms.Path( process.filter_all_star )
process.end_all_question = cms.Path( process.filter_all_question )
process.end_all_filter   = cms.Path( process.filter_1 + process.filter_2 + process.filter_3 )

process.end_wrong_name    = cms.Path( process.filter_wrong_name )
process.end_wrong_pattern = cms.Path( process.filter_wrong_pattern )
  
# define and EndPath to analyze all other path results
process.hltTrigReport = cms.EDAnalyzer( 'HLTrigReport',
    HLTriggerResults = cms.InputTag( 'TriggerResults','','TEST' )
)
process.HLTAnalyzerEndpath = cms.EndPath( process.hltTrigReport )
