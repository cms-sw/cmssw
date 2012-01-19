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
, fileNames = cms.untracked.vstring( '/store/relval/CMSSW_5_0_0/SingleMu/RECO/GR_R_50_V6_RelVal_mu2011B-v3/0000/06918CA3-9E2A-E111-8953-0018F34D0D62.root'
                                   , '/store/relval/CMSSW_5_0_0/SingleMu/RECO/GR_R_50_V6_RelVal_mu2011B-v3/0000/08A20D79-9E2A-E111-94E6-002618FDA21D.root'
                                   , '/store/relval/CMSSW_5_0_0/SingleMu/RECO/GR_R_50_V6_RelVal_mu2011B-v3/0000/1629187A-9E2A-E111-AC61-003048678B18.root'
                                   , '/store/relval/CMSSW_5_0_0/SingleMu/RECO/GR_R_50_V6_RelVal_mu2011B-v3/0000/1869A96A-9E2A-E111-A8A0-00261894393F.root'
                                   , '/store/relval/CMSSW_5_0_0/SingleMu/RECO/GR_R_50_V6_RelVal_mu2011B-v3/0000/1C3F2B6E-9E2A-E111-BAE7-0030486792A8.root'
                                   , '/store/relval/CMSSW_5_0_0/SingleMu/RECO/GR_R_50_V6_RelVal_mu2011B-v3/0000/1C61CF70-9E2A-E111-932A-00304867BFB2.root'
                                   , '/store/relval/CMSSW_5_0_0/SingleMu/RECO/GR_R_50_V6_RelVal_mu2011B-v3/0000/22AC6377-9E2A-E111-9427-00261894387C.root'
                                   , '/store/relval/CMSSW_5_0_0/SingleMu/RECO/GR_R_50_V6_RelVal_mu2011B-v3/0000/2A765A7C-9E2A-E111-80F4-002618943918.root'
                                   , '/store/relval/CMSSW_5_0_0/SingleMu/RECO/GR_R_50_V6_RelVal_mu2011B-v3/0000/4C871E67-9E2A-E111-A3E3-002618943978.root'
                                   , '/store/relval/CMSSW_5_0_0/SingleMu/RECO/GR_R_50_V6_RelVal_mu2011B-v3/0000/50106D72-9E2A-E111-B55E-00261894394D.root'
                                   , '/store/relval/CMSSW_5_0_0/SingleMu/RECO/GR_R_50_V6_RelVal_mu2011B-v3/0000/52583493-9E2A-E111-9BB4-00261894380A.root'
                                   , '/store/relval/CMSSW_5_0_0/SingleMu/RECO/GR_R_50_V6_RelVal_mu2011B-v3/0000/54888592-9E2A-E111-8832-0026189438F6.root'
                                   , '/store/relval/CMSSW_5_0_0/SingleMu/RECO/GR_R_50_V6_RelVal_mu2011B-v3/0000/60692C68-9E2A-E111-809B-0026189438C4.root'
                                   , '/store/relval/CMSSW_5_0_0/SingleMu/RECO/GR_R_50_V6_RelVal_mu2011B-v3/0000/6CBACF94-9E2A-E111-8DEF-0026189438FC.root'
                                   , '/store/relval/CMSSW_5_0_0/SingleMu/RECO/GR_R_50_V6_RelVal_mu2011B-v3/0000/7ECEAF75-9E2A-E111-9105-001BFCDBD15E.root'
                                   , '/store/relval/CMSSW_5_0_0/SingleMu/RECO/GR_R_50_V6_RelVal_mu2011B-v3/0000/86DE2F7E-9E2A-E111-856F-003048FFD7A2.root'
                                   , '/store/relval/CMSSW_5_0_0/SingleMu/RECO/GR_R_50_V6_RelVal_mu2011B-v3/0000/886A6579-9E2A-E111-9FB0-00304867906C.root'
                                   , '/store/relval/CMSSW_5_0_0/SingleMu/RECO/GR_R_50_V6_RelVal_mu2011B-v3/0000/AC93D1F2-9D2A-E111-AB02-0026189437FD.root'
                                   , '/store/relval/CMSSW_5_0_0/SingleMu/RECO/GR_R_50_V6_RelVal_mu2011B-v3/0000/C431F392-9E2A-E111-8056-003048FFD7BE.root'
                                   , '/store/relval/CMSSW_5_0_0/SingleMu/RECO/GR_R_50_V6_RelVal_mu2011B-v3/0000/D86B6E65-9E2A-E111-954B-0026189438D2.root'
                                   , '/store/relval/CMSSW_5_0_0/SingleMu/RECO/GR_R_50_V6_RelVal_mu2011B-v3/0000/E2980C72-9E2A-E111-A57C-001A92810AD4.root'
                                   )
)
#process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32( 1000 ) )
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32( 1 ) )

# Trigger analyzers
process.load( "HLTrigger.HLTcore.hltEventAnalyzerAOD_cfi" )
process.hltEventAnalyzerAOD.triggerName = cms.string( 'HLT_IsoMu24_eta2p1_v3' )

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
process.genericTriggerEventFlagTestLoose = process.genericTriggerEventFlagPass.clone( hltPaths = [ 'HLT_IsoMu24_*' ] )

# Paths
#process.p = cms.Path(
  #process.hltEventAnalyzerAOD
#)
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
