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
process.MessageLogger.categories.append('HLTrigReport')
process.MessageLogger.categories.append('L1GtTrigReport')
process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

# -- Database configuration
process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.load("CondCore.DBCommon.CondDBSetup_cfi")

# -- Conditions
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff") # works for MC & data
process.load("RecoVertex.BeamSpotProducer.BeamSpot_cfi")

process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff')
from Configuration.AlCa.GlobalTag_condDBv2 import GlobalTag
#process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')
process.GlobalTag = GlobalTag(process.GlobalTag, 'GR_E_V48', '')

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
    saveType                     = cms.untracked.string('Event'), # LumiSect, LumiNib, Event
    sampleType                   = cms.untracked.string('DATA'), # MC, DATA
    includeVertexInformation     = cms.untracked.bool(True),
    includePixels                = cms.untracked.bool(True),
    L1GTReadoutRecordLabel       = cms.untracked.InputTag('gtDigis'), 
    hltL1GtObjectMap             = cms.untracked.InputTag('hltL1GtObjectMap'), 
    HLTResultsLabel              = cms.untracked.InputTag('TriggerResults::HLT')
    )

# -- Path
process.p = cms.Path(
    process.zerobiasfilter*
    process.lumi
    )


outFile = 'pcc_Data_PixVtx_Event.root'
process.TFileService = cms.Service("TFileService",fileName = cms.string(outFile)) 
readFiles = cms.untracked.vstring() 
secFiles = cms.untracked.vstring() 
process.source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles) 
readFiles.extend([
'/store/data/Run2015A/ZeroBias1/RECO/PromptReco-v1/000/246/865/00000/C60316FA-150B-E511-AFCE-02163E0136E1.root',
'/store/data/Run2015A/ZeroBias1/RECO/PromptReco-v1/000/246/908/00000/0AE1AFA5-220B-E511-B7D4-02163E014239.root',
'/store/data/Run2015A/ZeroBias1/RECO/PromptReco-v1/000/246/908/00000/0CADC28A-390B-E511-853D-02163E0121C5.root',
'/store/data/Run2015A/ZeroBias1/RECO/PromptReco-v1/000/246/908/00000/10D9909F-260B-E511-9A53-02163E014113.root',
'/store/data/Run2015A/ZeroBias1/RECO/PromptReco-v1/000/246/908/00000/168661B7-300B-E511-870A-02163E013653.root',
'/store/data/Run2015A/ZeroBias1/RECO/PromptReco-v1/000/246/908/00000/1A1D4A18-2E0B-E511-9A5A-02163E012925.root',
'/store/data/Run2015A/ZeroBias1/RECO/PromptReco-v1/000/246/908/00000/1CC6C5E8-220B-E511-8640-02163E011ACE.root',
'/store/data/Run2015A/ZeroBias1/RECO/PromptReco-v1/000/246/908/00000/1E5ADEF5-200B-E511-B722-02163E01184D.root',
'/store/data/Run2015A/ZeroBias1/RECO/PromptReco-v1/000/246/908/00000/1E84CAD2-230B-E511-88F9-02163E013861.root',
'/store/data/Run2015A/ZeroBias1/RECO/PromptReco-v1/000/246/908/00000/20A71F54-550B-E511-B4DD-02163E0142F3.root',
'/store/data/Run2015A/ZeroBias1/RECO/PromptReco-v1/000/246/908/00000/20FBFCC6-240B-E511-A826-02163E012925.root',
'/store/data/Run2015A/ZeroBias1/RECO/PromptReco-v1/000/246/908/00000/2275F0CE-200B-E511-B714-02163E0143FC.root',
'/store/data/Run2015A/ZeroBias1/RECO/PromptReco-v1/000/246/908/00000/26CE2328-410B-E511-833A-02163E01184E.root',
'/store/data/Run2015A/ZeroBias1/RECO/PromptReco-v1/000/246/908/00000/2A151439-2C0B-E511-A1E1-02163E01383E.root',
'/store/data/Run2015A/ZeroBias1/RECO/PromptReco-v1/000/246/908/00000/2C44A971-230B-E511-B5E3-02163E012925.root',
'/store/data/Run2015A/ZeroBias1/RECO/PromptReco-v1/000/246/908/00000/34DC8712-220B-E511-8CE6-02163E0146EE.root',
'/store/data/Run2015A/ZeroBias1/RECO/PromptReco-v1/000/246/908/00000/3C37180B-220B-E511-AA12-02163E014220.root',
'/store/data/Run2015A/ZeroBias1/RECO/PromptReco-v1/000/246/908/00000/4862A3D9-220B-E511-A2CF-02163E013491.root',
'/store/data/Run2015A/ZeroBias1/RECO/PromptReco-v1/000/246/908/00000/4A84D28D-160B-E511-B9A2-02163E0142BF.root',
'/store/data/Run2015A/ZeroBias1/RECO/PromptReco-v1/000/246/908/00000/542C34A0-270B-E511-862E-02163E011BDB.root',
'/store/data/Run2015A/ZeroBias1/RECO/PromptReco-v1/000/246/908/00000/5C1F3477-230B-E511-BD27-02163E0142D7.root',
'/store/data/Run2015A/ZeroBias1/RECO/PromptReco-v1/000/246/908/00000/62A6E7A7-250B-E511-A56F-02163E011DBC.root',
'/store/data/Run2015A/ZeroBias1/RECO/PromptReco-v1/000/246/908/00000/64F30127-290B-E511-BF4E-02163E011A8B.root',
'/store/data/Run2015A/ZeroBias1/RECO/PromptReco-v1/000/246/908/00000/6CF3471A-420B-E511-BF19-02163E011ACE.root',
'/store/data/Run2015A/ZeroBias1/RECO/PromptReco-v1/000/246/908/00000/6E409396-170B-E511-9F1C-02163E01288E.root',
'/store/data/Run2015A/ZeroBias1/RECO/PromptReco-v1/000/246/908/00000/72942193-160B-E511-8134-02163E014565.root',
'/store/data/Run2015A/ZeroBias1/RECO/PromptReco-v1/000/246/908/00000/78691E83-170B-E511-9F99-02163E01262E.root',
'/store/data/Run2015A/ZeroBias1/RECO/PromptReco-v1/000/246/908/00000/7CC0D075-350B-E511-92A9-02163E0144DC.root',
'/store/data/Run2015A/ZeroBias1/RECO/PromptReco-v1/000/246/908/00000/7EAB39AD-2F0B-E511-A02B-02163E014565.root',
'/store/data/Run2015A/ZeroBias1/RECO/PromptReco-v1/000/246/908/00000/84E77A7A-250B-E511-A3E3-02163E011B55.root',
'/store/data/Run2015A/ZeroBias1/RECO/PromptReco-v1/000/246/908/00000/86F37DAA-250B-E511-B64E-02163E01475C.root',
'/store/data/Run2015A/ZeroBias1/RECO/PromptReco-v1/000/246/908/00000/88070E70-230B-E511-B786-02163E0133E6.root',
'/store/data/Run2015A/ZeroBias1/RECO/PromptReco-v1/000/246/908/00000/8A15372C-240B-E511-A816-02163E0118EC.root',
'/store/data/Run2015A/ZeroBias1/RECO/PromptReco-v1/000/246/908/00000/8C8C610C-280B-E511-BF59-02163E01475E.root',
'/store/data/Run2015A/ZeroBias1/RECO/PromptReco-v1/000/246/908/00000/8E2E9AAA-250B-E511-95CE-02163E013617.root',
'/store/data/Run2015A/ZeroBias1/RECO/PromptReco-v1/000/246/908/00000/8E40A5F7-290B-E511-A108-02163E0139C0.root',
'/store/data/Run2015A/ZeroBias1/RECO/PromptReco-v1/000/246/908/00000/94505B6B-1C0B-E511-90AC-02163E01396D.root',
'/store/data/Run2015A/ZeroBias1/RECO/PromptReco-v1/000/246/908/00000/9A8C4DF4-220B-E511-9340-02163E011C14.root',
'/store/data/Run2015A/ZeroBias1/RECO/PromptReco-v1/000/246/908/00000/9CFA6FCA-230B-E511-BCC9-02163E0134CC.root',
'/store/data/Run2015A/ZeroBias1/RECO/PromptReco-v1/000/246/908/00000/A082D468-260B-E511-B4C1-02163E0121DA.root',
'/store/data/Run2015A/ZeroBias1/RECO/PromptReco-v1/000/246/908/00000/A693F915-240B-E511-8F86-02163E011A86.root',
'/store/data/Run2015A/ZeroBias1/RECO/PromptReco-v1/000/246/908/00000/A8C0018C-170B-E511-B139-02163E013604.root',
'/store/data/Run2015A/ZeroBias1/RECO/PromptReco-v1/000/246/908/00000/B0472053-160B-E511-9FC9-02163E01288E.root',
'/store/data/Run2015A/ZeroBias1/RECO/PromptReco-v1/000/246/908/00000/B628561E-240B-E511-8D9A-02163E014215.root',
'/store/data/Run2015A/ZeroBias1/RECO/PromptReco-v1/000/246/908/00000/B89BB985-200B-E511-B12C-02163E014239.root',
'/store/data/Run2015A/ZeroBias1/RECO/PromptReco-v1/000/246/908/00000/BC57DB97-180B-E511-A70E-02163E0138BD.root',
'/store/data/Run2015A/ZeroBias1/RECO/PromptReco-v1/000/246/908/00000/C446636A-2B0B-E511-9DD7-02163E014110.root',
'/store/data/Run2015A/ZeroBias1/RECO/PromptReco-v1/000/246/908/00000/CE49A1A0-310B-E511-A0B1-02163E0134D2.root',
'/store/data/Run2015A/ZeroBias1/RECO/PromptReco-v1/000/246/908/00000/D41DC168-240B-E511-A96D-02163E01338A.root',
'/store/data/Run2015A/ZeroBias1/RECO/PromptReco-v1/000/246/908/00000/D6C1241C-260B-E511-9142-02163E0134CC.root',
'/store/data/Run2015A/ZeroBias1/RECO/PromptReco-v1/000/246/908/00000/DA7C76C1-2E0B-E511-86C9-02163E0140E8.root',
'/store/data/Run2015A/ZeroBias1/RECO/PromptReco-v1/000/246/908/00000/DA8EC182-240B-E511-9D19-02163E0141D2.root',
'/store/data/Run2015A/ZeroBias1/RECO/PromptReco-v1/000/246/908/00000/DE790703-1A0B-E511-8520-02163E012028.root',
'/store/data/Run2015A/ZeroBias1/RECO/PromptReco-v1/000/246/908/00000/E692A589-240B-E511-89BE-02163E011BD8.root',
'/store/data/Run2015A/ZeroBias1/RECO/PromptReco-v1/000/246/908/00000/EADC9DE0-270B-E511-BE4F-02163E012028.root',
'/store/data/Run2015A/ZeroBias1/RECO/PromptReco-v1/000/246/908/00000/EC2AF0F4-250B-E511-9248-02163E0136A3.root',
'/store/data/Run2015A/ZeroBias1/RECO/PromptReco-v1/000/246/908/00000/EE3D9096-240B-E511-B113-02163E014267.root',
'/store/data/Run2015A/ZeroBias1/RECO/PromptReco-v1/000/246/908/00000/F06C8362-2A0B-E511-9C99-02163E0135BC.root',
'/store/data/Run2015A/ZeroBias1/RECO/PromptReco-v1/000/246/908/00000/F28FBC09-250B-E511-A98B-02163E014204.root',
'/store/data/Run2015A/ZeroBias1/RECO/PromptReco-v1/000/246/908/00000/F2E1E279-230B-E511-8F7C-02163E0139C0.root',
'/store/data/Run2015A/ZeroBias1/RECO/PromptReco-v1/000/246/908/00000/F4E11D80-2B0B-E511-AE21-02163E0141D2.root',
'/store/data/Run2015A/ZeroBias1/RECO/PromptReco-v1/000/246/908/00000/FC11E464-160B-E511-B2DC-02163E0135CA.root',
'/store/data/Run2015A/ZeroBias1/RECO/PromptReco-v1/000/246/908/00000/FE0F20A6-200B-E511-A4F4-02163E011DA4.root',
])
