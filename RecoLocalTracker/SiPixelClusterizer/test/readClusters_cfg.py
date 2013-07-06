#
# Last update: new version for python
#
#
import FWCore.ParameterSet.Config as cms

process = cms.Process("cluTest")


import HLTrigger.HLTfilters.hltHighLevel_cfi as hlt
# accept if 'path_1' succeeds
process.hltfilter = hlt.hltHighLevel.clone(
# Min-Bias
# HLT_L1_BscMinBiasOR_BptxPlusORMinus, HLT_L1Tech_BSC_minBias, 	HLT_L1Tech_BSC_halo_forPhysicsBackground	
#    HLTPaths = ['HLT_Physics_v*'],
#    HLTPaths = ['HLT_L1Tech_BSC_minBias_threshold1_v*'],
    HLTPaths = ['HLT_Random_v*'],
#    HLTPaths = ['HLT_ZeroBias_v*'],
# old
#    HLTPaths = ['HLT_L1Tech_BSC_minBias'],
#    HLTPaths = ['HLT_L1Tech_BSC_minBias_OR'],
#    HLTPaths = ['HLT_L1Tech_BSC_halo_forPhysicsBackground'],
#    HLTPaths = ['HLT_L1Tech_BSC_HighMultiplicity'],
#    HLTPaths = ['HLT_L1_BPTX'],
#    HLTPaths = ['HLT_ZeroBias'],
#    HLTPaths = ['HLT_L1_BPTX_MinusOnly','HLT_L1_BPTX_PlusOnly'],
# Commissioning:
#    HLTPaths = ['HLT_L1_Interbunch_BSC_v*'],
#    HLTPaths = ['HLT_L1_PreCollisions_v*'],
#    HLTPaths = ['HLT_BeamGas_BSC_v*'],
#    HLTPaths = ['HLT_BeamGas_HF_v*'],
# old
#    HLTPaths = ['HLT_L1_BptxXOR_BscMinBiasOR'],
# Zero-Bias : HLT_L1_BPTX, HLT_L1_BPTX_PlusOnly, HLT_L1_BPTX_MinusOnly, HLT_ZeroBias
#    HLTPaths = ['HLT_L1_BPTX','HLT_ZeroBias','HLT_L1_BPTX_MinusOnly','HLT_L1_BPTX_PlusOnly'],
#    HLTPaths = ['p*'],
#    HLTPaths = ['path_?'],
    andOr = True,  # False = and, True=or
    throw = False
    )

# to select PhysicsBit
process.load('HLTrigger.special.hltPhysicsDeclared_cfi')
process.hltPhysicsDeclared.L1GtReadoutRecordTag = 'gtDigis'

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('siPixelClusters'),
    destinations = cms.untracked.vstring('cout'),
#    destinations = cms.untracked.vstring("log","cout"),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('ERROR')
    )
#    log = cms.untracked.PSet(
#        threshold = cms.untracked.string('DEBUG')
#    )
)

process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring(    
#  "/store/data/Run2011A/Commissioning/RECO/PromptReco-v1/000/160/497/102891A0-7250-E011-8061-0030487C90D4.root",
# R 187446
"/store/data/Commissioning12/MinimumBias/RECO/PromptReco-v1/000/187/446/FE7B607F-D76D-E111-993E-003048D37538.root",

  )

)

# process.source.lumisToProcess = cms.untracked.VLuminosityBlockRange('124230:26-124230:9999','124030:2-124030:9999')
# process.source.lumisToProcess = cms.untracked.VLuminosityBlockRange('133450:1-133450:657')
# process.source.lumisToProcess = cms.untracked.VLuminosityBlockRange('139239:160-139239:213')
# process.source.lumisToProcess = cms.untracked.VLuminosityBlockRange('142187:207-142187:9999')
#process.source.lumisToProcess = cms.untracked.VLuminosityBlockRange('160431:17-160431:9999')
# process.source.lumisToProcess = cms.untracked.VLuminosityBlockRange('160497:130-160497:9999')


process.TFileService = cms.Service("TFileService",
    fileName = cms.string('histo.root')
)

process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")

# what is this?
# process.load("Configuration.StandardSequences.Services_cff")

# what is this?
#process.load("SimTracker.Configuration.SimTracker_cff")

# needed for global transformation
# process.load("Configuration.StandardSequences.FakeConditions_cff")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")# Choose the global tag here:
process.GlobalTag.globaltag = "GR_P_V28::All"
# 2011
# process.GlobalTag.globaltag = "GR_P_V20::All"
#  process.GlobalTag.globaltag = "GR_R_311_V2::All"
# 2010
# process.GlobalTag.globaltag = 'GR10_P_V5::All'
# process.GlobalTag.globaltag = 'GR10_P_V4::All'
# OK for 2009 LHC data
#process.GlobalTag.globaltag = 'CRAFT09_R_V4::All'

process.analysis = cms.EDAnalyzer("ReadPixClusters",
    Verbosity = cms.untracked.bool(True),
    src = cms.InputTag("siPixelClusters"),
)

#process.p = cms.Path(process.hltfilter*process.analysis)
process.p = cms.Path(process.hltPhysicsDeclared*process.hltfilter*process.analysis)
#process.p = cms.Path(process.hltPhysicsDeclared*process.analysis)
#process.p = cms.Path(process.analysis)


# define an EndPath to analyze all other path results
#process.hltTrigReport = cms.EDAnalyzer( 'HLTrigReport',
#    HLTriggerResults = cms.InputTag( 'TriggerResults','','' )
#)
#process.HLTAnalyzerEndpath = cms.EndPath( process.hltTrigReport )
