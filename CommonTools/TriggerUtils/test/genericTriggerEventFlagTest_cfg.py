import FWCore.ParameterSet.Config as cms

process = cms.Process( "TEST" )

## Logging
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.categories.append( 'GenericTriggerEventFlag' )
process.MessageLogger.cerr.threshold = 'INFO'
process.MessageLogger.cerr.GenericTriggerEventFlag = cms.untracked.PSet( limit = cms.untracked.int32( -1 ) )
process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool( True ) )

# Conditions
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag( process.GlobalTag, 'auto:com10' )

## Source
process.source = cms.Source( "PoolSource"
#, fileNames  = cms.untracked.vstring( '/store/relval/CMSSW_6_1_0_pre3-GR_R_60_V7_RelVal_mu2011A/SingleMu/RECO/v1/00000/16461657-910C-E211-BD66-0026189438CB.root'
, fileNames  = cms.untracked.vstring( '/store/relval/CMSSW_6_1_0_pre3-GR_R_60_V7_RelVal_mu2012B/SingleMu/RECO/v1/00000/08491D7D-CE0C-E211-A9BB-003048678D86.root'
                                    )
#, skipEvents = cms.untracked.uint32( 0 )
)
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32( 10 ) )

# Test modules
# GT
process.genericTriggerEventFlagGTPass = cms.EDFilter( "GenericTriggerEventFlagTest"
, andOr          = cms.bool( False )
, verbosityLevel = cms.uint32( 2 )
, andOrGt        = cms.bool( False )
, gtInputTag     = cms.InputTag( 'gtDigis' )
, gtEvmInputTag  = cms.InputTag( 'gtEvmDigis' )
, gtStatusBits   = cms.vstring( 'PhysDecl'
                              )
, errorReplyGt   = cms.bool( False )
)
process.genericTriggerEventFlagGTFail     = process.genericTriggerEventFlagGTPass.clone( gtStatusBits = [ '900GeV' ] )
process.genericTriggerEventFlagGTTest     = process.genericTriggerEventFlagGTPass.clone( gtStatusBits = [ '8TeV' ] )
process.genericTriggerEventFlagGTTestFail = process.genericTriggerEventFlagGTPass.clone( gtInputTag   = 'whateverDigis' )
# L1
process.genericTriggerEventFlagL1Pass = cms.EDFilter( "GenericTriggerEventFlagTest"
, andOr          = cms.bool( False )
, verbosityLevel = cms.uint32( 2 )
, andOrL1      = cms.bool( False )
, l1Algorithms = cms.vstring( 'L1_SingleMu12 OR L1_SingleMu14_Eta2p1'
                            )
, errorReplyL1 = cms.bool( False )
)
process.genericTriggerEventFlagL1Fail     = process.genericTriggerEventFlagL1Pass.clone( l1Algorithms = [ 'L1_SingleMu12_' ] )
process.genericTriggerEventFlagL1Test     = process.genericTriggerEventFlagL1Pass.clone( l1Algorithms = [ 'L1_SingleMu12* OR L1_SingleMu14*' ] )
process.genericTriggerEventFlagL1TestFail = process.genericTriggerEventFlagL1Pass.clone( l1Algorithms = [ 'L1_SingleMu12_v*' ] )
# HLT
process.genericTriggerEventFlagHLTPass = cms.EDFilter( "GenericTriggerEventFlagTest"
, andOr          = cms.bool( False )
, verbosityLevel = cms.uint32( 2 )
, andOrHlt      = cms.bool( False )
, hltInputTag   = cms.InputTag( 'TriggerResults::HLT' )
, hltPaths      = cms.vstring( 'HLT_IsoMu24_eta2p1_v13' # only in 2012B
                             )
, errorReplyHlt = cms.bool( False )
)
process.genericTriggerEventFlagHLTFail      = process.genericTriggerEventFlagHLTPass.clone( hltPaths = [ 'HLT_IsoMu24_eta2p1' ] )
process.genericTriggerEventFlagHLTTestTight = process.genericTriggerEventFlagHLTPass.clone( hltPaths = [ 'HLT_IsoMu24_eta2p1_v*' ] )
process.genericTriggerEventFlagHLTTestLoose = process.genericTriggerEventFlagHLTPass.clone( hltPaths = [ 'HLT_IsoMu24_*' ] )
process.genericTriggerEventFlagHLTTestFail  = process.genericTriggerEventFlagHLTPass.clone( hltPaths = [ 'HLT_IsoMu2*_v*' ] )        # does not fail, in fact :-)

# Paths
# GT
process.pGTPass = cms.Path(
  process.genericTriggerEventFlagGTPass
)
process.pGTFail = cms.Path(
  process.genericTriggerEventFlagGTFail
)
process.pGTTest = cms.Path(
  process.genericTriggerEventFlagGTTest
)
process.pGTTestFail = cms.Path(
  process.genericTriggerEventFlagGTTestFail
)
# L1
process.pL1Pass = cms.Path(
  process.genericTriggerEventFlagL1Pass
)
process.pL1Fail = cms.Path(
  process.genericTriggerEventFlagL1Fail
)
process.pL1Test = cms.Path(
  process.genericTriggerEventFlagL1Test
)
process.pL1TestFail = cms.Path(
  process.genericTriggerEventFlagL1TestFail
)
# HLT
process.pHLTPass = cms.Path(
  process.genericTriggerEventFlagHLTPass
)
process.pHLTFail = cms.Path(
  process.genericTriggerEventFlagHLTFail
)
process.pHLTTestTight = cms.Path(
  process.genericTriggerEventFlagHLTTestTight
)
process.pHLTTestLoose = cms.Path(
  process.genericTriggerEventFlagHLTTestLoose
)
process.pHLTTestFail = cms.Path(
  process.genericTriggerEventFlagHLTTestFail
)
