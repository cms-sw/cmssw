import FWCore.ParameterSet.Config as cms

process = cms.Process('TEST')

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)
process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.cerr.INFO = cms.untracked.PSet(
    reportEvery = cms.untracked.int32(1), # every!
    limit = cms.untracked.int32(-1)       # no limit!
    )
process.MessageLogger.cerr.FwkReport.reportEvery = 10 # only report every 10th event start
process.MessageLogger.cerr_stats.threshold = 'INFO' # also info in statistics

# read back the trigger decisions
process.source = cms.Source('PoolSource',
    fileNames = cms.untracked.vstring('file:trigger.root')
)

import HLTrigger.HLTfilters.hltHighLevel_cfi as hlt
# accept if 'path_1' succeeds
process.filter_1 = hlt.hltHighLevel.clone(
    HLTPaths = [ 'path_1'],
    throw = False
    )

# accept if 'path_2' succeeds
process.filter_2 = hlt.hltHighLevel.clone(
    HLTPaths = ['path_2'],
    throw = False
    )

# accept if 'path_3' succeeds
process.filter_3 = hlt.hltHighLevel.clone(
    HLTPaths = ['path_3'],
    throw = False
    )

# accept if any path succeeds (implicit)
process.filter_any_implicit = hlt.hltHighLevel.clone(
    # HLTPaths = [], # empty is default
    throw = False
    )

# accept if any path succeeds (explicit)
process.filter_any_explicit = hlt.hltHighLevel.clone(
    HLTPaths = ['path_1', 'path_2', 'path_3'],
    throw = False
    )

# accept if any path succeeds (wildcard, '*')
process.filter_any_star = hlt.hltHighLevel.clone(
    HLTPaths = ['p*'],
    throw = False
    )

# accept if any path succeeds (wildcard, twice '*')
process.filter_any_doublestar = hlt.hltHighLevel.clone(
    HLTPaths = ['p*t*'],
    throw = False
    )


# accept if any path succeeds (wildcard, '?')
process.filter_any_question = hlt.hltHighLevel.clone(
    HLTPaths = ['path_?'],
    throw = False
    )

# accept if all path succeed (implicit)
process.filter_all_implicit = hlt.hltHighLevel.clone(
    #HLTPaths = [], # empty is default
    andOr = False,
    throw = False
)

# accept if all path succeed (explicit)
process.filter_all_explicit = hlt.hltHighLevel.clone(
    HLTPaths = ['path_1', 'path_2', 'path_3'],
    andOr = False,
    throw = False
)

# accept if all path succeed (wildcard, '*')
process.filter_all_star = hlt.hltHighLevel.clone(
    HLTPaths = ['p*'],
    andOr = False,
    throw = False
)

# accept if all path succeed (wildcard, '*')
process.filter_all_doublestar = hlt.hltHighLevel.clone(
    HLTPaths = ['p*t*'],
    andOr = False,
    throw = False
)

# accept if all path succeed (wildcard, '?')
process.filter_all_question = hlt.hltHighLevel.clone(
    HLTPaths = ['path_?'],
    andOr = False,
    throw = False
)

# wrong L1 name (explicit)
process.filter_wrong_name = hlt.hltHighLevel.clone(
    HLTPaths = ['path_wrong'],
    throw = False
)

# wrong L1 name (wildcard)
process.filter_wrong_pattern = hlt.hltHighLevel.clone(
    HLTPaths = ['*_wrong'],
    throw = False
)

## start testing AlCaRecoTriggerBits ##############################
##
## This works after having run a modified version of
## cmsRun src/CondTools/HLT/test/AlCaRecoTriggerBitsRcdWrite_cfg.py
## Simply remove overwriting of
## process.AlCaRecoTriggerBitsRcdWrite.triggerLists ...
##
## AlCaRecoTriggerBits
#process.filter_AlCaRecoTriggerBits = hlt.hltHighLevel.clone(
#    eventSetupPathsKey = 'test13', #'TkAlMinBias',
#    throw = False # True
#)
#
## DB input
#import CondCore.DBCommon.CondDBSetup_cfi
#process.dbInput = cms.ESSource(
#    "PoolDBESSource",
#    CondCore.DBCommon.CondDBSetup_cfi.CondDBSetup,
#    connect = cms.string('sqlite_file:AlCaRecoTriggerBits.db'),
#    toGet = cms.VPSet(cms.PSet(
#        record = cms.string('AlCaRecoTriggerBitsRcd'),
#        tag = cms.string('TestTag') # choose tag you want
#        )
#                      )
#    )
#process.end_AlCaRecoTriggerBits = cms.Path( process.filter_AlCaRecoTriggerBits )
##
## end testing AlCaRecoTriggerBits ################################

process.end_1 = cms.Path( process.filter_1 )
process.end_2 = cms.Path( process.filter_2 )
process.end_3 = cms.Path( process.filter_3 )

process.end_any_implicit = cms.Path( process.filter_any_implicit )
process.end_any_explicit = cms.Path( process.filter_any_explicit )
process.end_any_star     = cms.Path( process.filter_any_star )
process.end_any_doublestar = cms.Path( process.filter_any_doublestar )
process.end_any_question = cms.Path( process.filter_any_question )
#process.end_any_filter   = cms.Path( ~ ( ~ process.filter_1 + ~ process.filter_2 + ~ process.filter_3) )

process.end_all_implicit = cms.Path( process.filter_all_implicit )
process.end_all_explicit = cms.Path( process.filter_all_explicit )
process.end_all_star     = cms.Path( process.filter_all_star )
process.end_all_doublestar = cms.Path( process.filter_all_doublestar )
process.end_all_question = cms.Path( process.filter_all_question )
process.end_all_filter   = cms.Path( process.filter_1 + process.filter_2 + process.filter_3 )

process.end_wrong_name    = cms.Path( process.filter_wrong_name )
process.end_wrong_pattern = cms.Path( process.filter_wrong_pattern )
process.end_not_wrong_pattern = cms.Path( ~process.filter_wrong_pattern )

# define an EndPath to analyze all other path results
process.hltTrigReport = cms.EDAnalyzer( 'HLTrigReport',
    HLTriggerResults = cms.InputTag( 'TriggerResults','','TEST' )
)
process.HLTAnalyzerEndpath = cms.EndPath( process.hltTrigReport )
