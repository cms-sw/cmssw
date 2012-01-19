import FWCore.ParameterSet.Config as cms

process = cms.Process( "TEST" )

## Logging
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.categories.append( 'GenericTriggerEventFlag' )
process.MessageLogger.cerr.threshold = 'INFO'
process.MessageLogger.cerr.GenericTriggerEventFlag = cms.untracked.PSet( limit = cms.untracked.int32( -1 ) )
process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool( True ) )

# Conditions
from Configuration.AlCa.autoCond import autoCond
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'GR_R_50_V6::All'

## Source
process.source = cms.Source( "PoolSource"
, fileNames  = cms.untracked.vstring( '/store/relval/CMSSW_5_0_0/SingleMu/RECO/GR_R_50_V6_RelVal_mu2011A-v3/0000/0026C043-F52A-E111-B383-001A9281173C.root'
                                    , '/store/relval/CMSSW_5_0_0/SingleMu/RECO/GR_R_50_V6_RelVal_mu2011B-v3/0000/06918CA3-9E2A-E111-8953-0018F34D0D62.root'
                                    )
, skipEvents = cms.untracked.uint32( 20200 )
)
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32( 100 ) )

# Test modules
process.genericTriggerEventFlagPass = cms.EDFilter( "GenericTriggerEventFlagTest"
, andOr          = cms.bool( False )
, verbosityLevel = cms.uint32( 2 )
, andOrHlt      = cms.bool( False )
, hltInputTag   = cms.InputTag( 'TriggerResults::HLT' )
, hltPaths      = cms.vstring( 'HLT_IsoMu24_eta2p1_v3'
                             )
, errorReplyHlt = cms.bool( False )
)
process.genericTriggerEventFlagFail      = process.genericTriggerEventFlagPass.clone( hltPaths = [ 'HLT_IsoMu24_eta2p1' ] )
process.genericTriggerEventFlagTestTight = process.genericTriggerEventFlagPass.clone( hltPaths = [ 'HLT_IsoMu24_eta2p1_v*' ] )
process.genericTriggerEventFlagTestLoose = process.genericTriggerEventFlagPass.clone( hltPaths = [ 'HLT_Mu*_* OR HLT_IsoMu*_*' ] )

# Paths
process.pPass = cms.Path(
  process.genericTriggerEventFlagPass
)
process.pFail = cms.Path(
  process.genericTriggerEventFlagFail
)
process.pTestTight = cms.Path(
  process.genericTriggerEventFlagTestTight
)
process.pTestLoose = cms.Path(
  process.genericTriggerEventFlagTestLoose
)
