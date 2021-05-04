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
process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.load("CondCore.DBCommon.CondDBSetup_cfi")

# -- Conditions
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff") # works for MC & data
process.load("RecoVertex.BeamSpotProducer.BeamSpot_cfi")

process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
#process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_hlt_GRun', '')

process.load("Configuration.StandardSequences.Reconstruction_cff")

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
    pixelClusterLabel            = cms.untracked.InputTag('siPixelClusters'),
    saveType                     = cms.untracked.string('LumiSect'), # LumiSect, LumiNib, Event
    sampleType                   = cms.untracked.string('DATA'), # MC, DATA
    includeVertexInformation     = cms.untracked.bool(True),
    includePixels                = cms.untracked.bool(True),
    splitByBX                    = cms.untracked.bool(False),
    L1GTReadoutRecordLabel       = cms.untracked.InputTag('gtDigis'), 
    hltL1GtObjectMap             = cms.untracked.InputTag('hltL1GtObjectMap'), 
    HLTResultsLabel              = cms.untracked.InputTag('TriggerResults::HLT')
    )

# -- Path
process.p = cms.Path(
    process.zerobiasfilter*
    process.lumi
    )


outFile = 'pcc_Data_PixVtx_LS.root'
process.TFileService = cms.Service("TFileService",fileName = cms.string(outFile)) 
readFiles = cms.untracked.vstring() 
secFiles = cms.untracked.vstring() 
process.source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles) 
readFiles.extend([
'/store/express/Run2015C/ExpressPhysics_0T/FEVT/Express-v3/000/256/385/00000/02EC6A5B-5B59-E511-A555-02163E011A29.root',
'/store/express/Run2015C/ExpressPhysics_0T/FEVT/Express-v3/000/256/385/00000/0616947E-5959-E511-9D4F-02163E0132A0.root',
'/store/express/Run2015C/ExpressPhysics_0T/FEVT/Express-v3/000/256/385/00000/0C5246A7-6959-E511-A58E-02163E0124A9.root',
'/store/express/Run2015C/ExpressPhysics_0T/FEVT/Express-v3/000/256/385/00000/10DD1A69-5859-E511-B314-02163E0140DE.root',
])
