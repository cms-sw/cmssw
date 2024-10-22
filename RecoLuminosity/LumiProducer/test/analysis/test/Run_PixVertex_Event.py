# ######################################################################
#
# pixelLumi.py
#
# ----------------------------------------------------------------------
import os
import FWCore.ParameterSet.Config as cms
process = cms.Process("Lumi")

# ----------------------------------------------------------------------
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = 'INFO'
process.MessageLogger.cerr.FwkReport.reportEvery = 1000
process.MessageLogger.HLTrigReport=dict()
process.MessageLogger.L1GtTrigReport=dict()
process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

# -- Database configuration
process.load("CondCore.CondDB.CondDB_cfi")

# -- Conditions

process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff") #
process.load("RecoVertex.BeamSpotProducer.BeamSpot_cfi")

process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag

#process.GlobalTag = GlobalTag(process.GlobalTag, '80X_mcRun2_asymptotic_v14', '')
process.GlobalTag = GlobalTag(process.GlobalTag, '90X_upgrade2023_realistic_v1', '')


process.load("Configuration.StandardSequences.Reconstruction_cff") # 
# -- number of events
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
    )

# -- skimming
process.zerobiasfilter = cms.EDFilter("HLTHighLevel",
   TriggerResultsTag = cms.InputTag("TriggerResults","","HLT"),
   #HLTPaths = cms.vstring("HLT_ZeroBias_v*"),
   HLTPaths = cms.vstring("*ZeroBias*"),
   eventSetupPathsKey = cms.string(""),
   andOr = cms.bool(True),
   throw = cms.bool(False)
    )

# the main Analyzer
process.lumi = cms.EDAnalyzer(
    "PCCNTupler",
    verbose                      = cms.untracked.int32(0),
    #rootFileName                 = cms.untracked.string(rootFileName),
    #type                         = cms.untracked.string(getDataset(process.source.fileNames[0])),
    globalTag                    = process.GlobalTag.globaltag,
    dumpAllEvents                = cms.untracked.int32(0),
    vertexCollLabel              = cms.untracked.InputTag('offlinePrimaryVertices'),
    pixelClusterLabel            = cms.untracked.InputTag('siPixelClusters'), # even in Phase2, for now.
    saveType                     = cms.untracked.string('Event'), # LumiSect, LumiNib, Event
    sampleType                   = cms.untracked.string('MC'), # MC, DATA
    includeVertexInformation     = cms.untracked.bool(True),
    includePixels                = cms.untracked.bool(True),
    splitByBX                    = cms.untracked.bool(True),
    L1GTReadoutRecordLabel       = cms.untracked.InputTag('gtDigis'), 
    hltL1GtObjectMap             = cms.untracked.InputTag('hltL1GtObjectMap'), 
    HLTResultsLabel              = cms.untracked.InputTag('TriggerResults::HLT'),
    pixelPhase2Geometry          = cms.untracked.bool(True),
    )

# -- Path
process.p = cms.Path(
    process.zerobiasfilter*
    process.lumi
    )


outFile = 'pcc_Data_PixVtx_Event_90X.root'
process.TFileService = cms.Service("TFileService",fileName = cms.string(outFile)) 
readFiles = cms.untracked.vstring() 
secFiles = cms.untracked.vstring() 
process.source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles) 
readFiles.extend([
# Min Bias 90X files with 2023D4 geometry and timing. no pu.
 '/store/relval/CMSSW_9_0_0_pre2/RelValMinBias_14TeV/GEN-SIM-RECO/90X_upgrade2023_realistic_v1_2023D4Timing-v1/10000/28088B65-66C2-E611-BF89-0CC47A7C347A.root',
 '/store/relval/CMSSW_9_0_0_pre2/RelValMinBias_14TeV/GEN-SIM-RECO/90X_upgrade2023_realistic_v1_2023D4Timing-v1/10000/20D68D58-3CC2-E611-B15B-0CC47A4C8F18.root',
 '/store/relval/CMSSW_9_0_0_pre2/RelValMinBias_14TeV/GEN-SIM-RECO/90X_upgrade2023_realistic_v1_2023D4Timing-v1/10000/94242929-30C3-E611-B3E0-0025905B85DC.root',
 '/store/relval/CMSSW_9_0_0_pre2/RelValMinBias_14TeV/GEN-SIM-RECO/90X_upgrade2023_realistic_v1_2023D4Timing-v1/10000/F46D98E7-EAC2-E611-936E-0CC47A7452D0.root',
 '/store/relval/CMSSW_9_0_0_pre2/RelValMinBias_14TeV/GEN-SIM-RECO/90X_upgrade2023_realistic_v1_2023D4Timing-v1/10000/36D3D8CD-3BC2-E611-908A-0025905A6088.root',
#'/store/mc/RunIISummer16DR80/MinBias_TuneCUETP8M1_13TeV-pythia8/GEN-SIM-RECO/NoPU_RECO_80X_mcRun2_asymptotic_v14-v1/100000/00150044-D075-E611-AAE8-001E67505A2D.root', # 80X file
#'/store/data/Run2015A/ZeroBias1/RECO/PromptReco-v1/000/250/786/00000/B4CDEBBC-F52A-E511-808D-02163E011CE8.root', 
])
