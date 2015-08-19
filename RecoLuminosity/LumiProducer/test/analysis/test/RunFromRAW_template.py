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
###process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff") # works for MC & data
process.load("RecoVertex.BeamSpotProducer.BeamSpot_cfi")

process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff')
from Configuration.AlCa.GlobalTag_condDBv2 import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')

###process.load("Configuration.StandardSequences.Reconstruction_cff")

###### ADDED FOR RAW
#### standard includes
process.load('Configuration/StandardSequences/Services_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_PostLS1_cff')
process.load("Configuration.StandardSequences.RawToDigi_cff")
process.load("Configuration.EventContent.EventContent_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Configuration.StandardSequences.PostRecoGenerator_cff")
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.L1Reco_cff')
#process.load("RecoLocalCalo.Configuration.hcalLocalReco_cff")
#process.load("RecoLocalCalo.Configuration.ecalLocalRecoSequence_cff")

process.particleFlowClusterECAL.energyCorrector.autoDetectBunchSpacing = False
process.particleFlowClusterECAL.energyCorrector.bunchSpacing = cms.int32(25)

process.ecalMultiFitUncalibRecHit.algoPSet = cms.PSet( #for CMSSW >=750pre1
    activeBXs = cms.vint32(-5,-4,-3,-2,-1,0,1,2,3,4),
    useLumiInfoRunHeader = cms.bool(False)
)

# Path and EndPath definitions
process.raw2digi_step = cms.Path(process.RawToDigi)
process.L1Reco_step = cms.Path(process.L1Reco)
process.reconstruction_step = cms.Path(process.reconstruction)

# Automatic addition of the customisation function from SLHCUpgradeSimulations.Configuration.postLS1Customs
from SLHCUpgradeSimulations.Configuration.postLS1Customs import customisePostLS1

#call to customisation function customisePostLS1 imported from SLHCUpgradeSimulations.Configuration.postLS1Customs
process = customisePostLS1(process)

# for HLT
if hasattr(process, 'hltCsc2DRecHits'):
    process.hltCsc2DRecHits.wireDigiTag  = cms.InputTag("simMuonCSCDigis","MuonCSCWireDigi")
    process.hltCsc2DRecHits.stripDigiTag = cms.InputTag("simMuonCSCDigis","MuonCSCStripDigi")
# for the L1 emulator
if hasattr(process, 'cscReEmulTriggerPrimitiveDigis'):
    process.cscReEmulTriggerPrimitiveDigis.CSCComparatorDigiProducer = cms.InputTag("simMuonCSCDigis","MuonCSCComparatorDigi")
    process.cscReEmulTriggerPrimitiveDigis.CSCWireDigiProducer = cms.InputTag("simMuonCSCDigis","MuonCSCWireDigi")

## END ADDED FOR RAW


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
    sampleType                   = cms.untracked.string('MC'), # MC, DATA
    includeVertexInformation     = cms.untracked.bool(True),
    includePixels                = cms.untracked.bool(True),
    L1GTReadoutRecordLabel       = cms.untracked.InputTag('gtDigis'), 
    hltL1GtObjectMap             = cms.untracked.InputTag('hltL1GtObjectMap'), 
    HLTResultsLabel              = cms.untracked.InputTag('TriggerResults::HLT')
    )

process.endjob_step = cms.EndPath(process.lumi*process.endOfProcess)
process.schedule = cms.Schedule(process.raw2digi_step,process.L1Reco_step,process.reconstruction_step,process.endjob_step)

# -- Path
#process.p = cms.Path(
#    process.raw2cluster
#    *process.clustToHits
#    *process.tracking
##    process.zerobiasfilter*
#    )


outFile = 'pcc_MC_fromRAW.root'
process.TFileService = cms.Service("TFileService",fileName = cms.string(outFile)) 
readFiles = cms.untracked.vstring() 
secFiles = cms.untracked.vstring() 
process.source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles) 
readFiles.extend([
'/store/mc/Spring14dr/Neutrino_Pt-2to20_gun/GEN-SIM-RAW/Flat0to10_POSTLS170_V5-v2/10000/00111A6D-3D53-E411-BA00-E0CB4E29C4F7.root',
'/store/mc/Spring14dr/Neutrino_Pt-2to20_gun/GEN-SIM-RAW/Flat0to10_POSTLS170_V5-v2/10000/0013DC3F-5153-E411-8D28-20CF3027A5C9.root',
'/store/mc/Spring14dr/Neutrino_Pt-2to20_gun/GEN-SIM-RAW/Flat0to10_POSTLS170_V5-v2/10000/004B5217-3053-E411-83F6-E0CB4E29C503.root',
'/store/mc/Spring14dr/Neutrino_Pt-2to20_gun/GEN-SIM-RAW/Flat0to10_POSTLS170_V5-v2/10000/00649445-6553-E411-8BFD-00259073E4E8.root',
'/store/mc/Spring14dr/Neutrino_Pt-2to20_gun/GEN-SIM-RAW/Flat0to10_POSTLS170_V5-v2/10000/0081E66A-EA52-E411-A1F4-002590D0AF78.root',
'/store/mc/Spring14dr/Neutrino_Pt-2to20_gun/GEN-SIM-RAW/Flat0to10_POSTLS170_V5-v2/10000/0082EB83-DF52-E411-AF61-20CF30561701.root',
'/store/mc/Spring14dr/Neutrino_Pt-2to20_gun/GEN-SIM-RAW/Flat0to10_POSTLS170_V5-v2/10000/00C2EC36-8653-E411-9EF2-E0CB4EA0A908.root',
'/store/mc/Spring14dr/Neutrino_Pt-2to20_gun/GEN-SIM-RAW/Flat0to10_POSTLS170_V5-v2/10000/00C3DEC7-DF52-E411-92F4-00259074AE80.root',
'/store/mc/Spring14dr/Neutrino_Pt-2to20_gun/GEN-SIM-RAW/Flat0to10_POSTLS170_V5-v2/10000/022323AE-8953-E411-AA9A-002590D0AFE8.root',
'/store/mc/Spring14dr/Neutrino_Pt-2to20_gun/GEN-SIM-RAW/Flat0to10_POSTLS170_V5-v2/10000/0247B968-1A53-E411-8B24-0025907B4F9E.root',
'/store/mc/Spring14dr/Neutrino_Pt-2to20_gun/GEN-SIM-RAW/Flat0to10_POSTLS170_V5-v2/10000/02592BCF-2053-E411-B33A-00259073E364.root',
'/store/mc/Spring14dr/Neutrino_Pt-2to20_gun/GEN-SIM-RAW/Flat0to10_POSTLS170_V5-v2/10000/025E2F1B-7B53-E411-B7B7-002590D0B0AA.root',
'/store/mc/Spring14dr/Neutrino_Pt-2to20_gun/GEN-SIM-RAW/Flat0to10_POSTLS170_V5-v2/10000/025ED27A-9653-E411-96A6-20CF3027A639.root',
'/store/mc/Spring14dr/Neutrino_Pt-2to20_gun/GEN-SIM-RAW/Flat0to10_POSTLS170_V5-v2/10000/02606AC7-5F53-E411-88CB-20CF3027A630.root',
#'/store/relval/CMSSW_7_4_1/RelValProdMinBias_13/GEN-SIM-RAW/MCRUN2_74_V9_gensim71X-v1/00000/6EC23239-AEEC-E411-B4E4-0025905B860C.root'
#'/store/mc/Spring14dr/Neutrino_Pt-2to20_gun/AODSIM/Flat0to10_POSTLS170_V5-v2/10000/00AE7E7E-6153-E411-9565-002590D0AFBE.root'
#'file:/tmp/capalmer/00111A6D-3D53-E411-BA00-E0CB4E29C4F7.root'
#'/store/mc/Spring14dr/Neutrino_Pt-2to20_gun/GEN-SIM-RAW/Flat0to10_POSTLS170_V5-v2/10000/00111A6D-3D53-E411-BA00-E0CB4E29C4F7.root'
#'/store/data/Run2012D/ZeroBias1/RECO/PromptReco-v1/000/206/251/F28DAF8D-7723-E211-80A1-BCAEC5364C4C.root'
#'/store/relval/CMSSW_7_4_0_pre8/RelValMinBias_13/GEN-SIM-RECO/MCRUN2_74_V7-v1/00000/08A7F47B-B9BD-E411-97B0-0025905B85D6.root'
])
