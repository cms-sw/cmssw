import FWCore.ParameterSet.Config as cms
process = cms.Process('openHLT')
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
    #SkipEvent = cms.untracked.vstring('ProductNotFound')
)

#####################################################################################
# Input source
#####################################################################################

process.source = cms.Source("PoolSource",
                            duplicateCheckMode = cms.untracked.string("noDuplicateCheck"),
                            #fileNames = cms.untracked.vstring("root://xrootd.cmsaf.mit.edu//store/user/pkurt/HidjetQuenchedMinBias/44X_Embeddeding_workflow_test_dijet80_RECO_v4/86245d447b905f80021c91863b830407/step4_RAW2DIGI_L1Reco_RECO_VALIDATION_DQM_4_1_lmt.root")
                            fileNames = cms.untracked.vstring("file:L1Emulator_HI_oldGCT.root")
                            #fileNames = cms.untracked.vstring("file:/net/hisrv0001/home/luck/UCT2015/CMSSW_7_0_0_pre8/src/L1Trigger/L1TCommon/demo/demo_output_full_norm.root")
                           )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)


#####################################################################################
# Load Global Tag, Geometry, etc.
#####################################################################################

process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgradePLS1', '')

#####################################################################################
# Define tree output
#####################################################################################

process.TFileService = cms.Service("TFileService",
                                   fileName=cms.string("hltana_HI_oldGCT.root"))

#process.load('HLTrigger.HLTanalyzers.hltanalysis_cff')
#from HLTrigger.HLTanalyzers.HLTBitAnalyser_cfi import *
process.load('HLTrigger.HLTanalyzers.HLTBitAnalyser_cfi')

process.hltbitanalysis.UseTFileService = cms.untracked.bool(True)
process.hltanalysis = process.hltbitanalysis.clone(
    dummyBranches = cms.untracked.vstring(
    ),
    # most important, contains sim L1 results
    l1GtReadoutRecord    = cms.InputTag("simGtDigis","","L1TEMULATION"),
    # the following are less important, does not affect triggers
    l1GctHFBitCounts     = cms.InputTag("GCTConverter","","L1TEMULATION"),
    l1GctHFRingSums      = cms.InputTag("GCTConverter","","L1TEMULATION"),
    l1extramu            = cms.string('l1extraParticles'),
    l1extramc            = cms.string('l1extraParticles'),
    # hlt info currently dropped, not important
    hltresults           = cms.InputTag("TriggerResults","","L1TEMULATION"),
    HLTProcessName       = cms.string("HLT")
    )

process.hltAna = cms.Path(process.hltanalysis)
