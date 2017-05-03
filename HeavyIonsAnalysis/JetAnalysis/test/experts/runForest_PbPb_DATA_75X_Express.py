import FWCore.ParameterSet.Config as cms
process = cms.Process('HiForest')
process.options = cms.untracked.PSet(
    # wantSummary = cms.untracked.bool(True)
    #SkipEvent = cms.untracked.vstring('ProductNotFound')
)

#parse command line arguments
from FWCore.ParameterSet.VarParsing import VarParsing
options = VarParsing('analysis')
options.register ('isPP',
                  False,
                  VarParsing.multiplicity.singleton,
                  VarParsing.varType.bool,
                  "Flag if this is a pp simulation")
options.parseArguments()

#####################################################################################
# HiForest labelling info
#####################################################################################

process.load("HeavyIonsAnalysis.JetAnalysis.HiForest_cff")
process.HiForest.inputLines = cms.vstring("HiForest V3",)
import subprocess
version = subprocess.Popen(["(cd $CMSSW_BASE/src && git describe --tags)"], stdout=subprocess.PIPE, shell=True).stdout.read()
if version == '':
    version = 'no git info'
process.HiForest.HiForestVersion = cms.string(version)

#####################################################################################
# Input source
#####################################################################################

process.source = cms.Source("PoolSource",
                            duplicateCheckMode = cms.untracked.string("noDuplicateCheck"),
                           fileNames = cms.untracked.vstring(options.inputFiles[0])                            
# fileNames = cms.untracked.vstring("file:/mnt/hadoop/cms/store/user/dgulhan/HIHighPt/HIHighPt_photon20and30_HIRun2011-v1_RECO_753_patch1/fd44351629dd155a25de2b4c109c824c/RECO_100_1_Uk0.root")                        )
                            # fileNames = cms.untracked.vstring("/store/express/Run2015E/ExpressPhysics/FEVT/Express-v1/000/261/544/00000//22D08F8A-2E8D-E511-BF87-02163E011965.root")                        
)


# Number of events we want to process, -1 = all events
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(options.maxEvents))

import FWCore.PythonUtilities.LumiList as LumiList
process.source.lumisToProcess = LumiList.LumiList(filename = '/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions15/HI/DCSOnly/json_DCSONLY.txt').getVLuminosityBlockRange()
# process.source.lumisToProcess = LumiList.LumiList(filename = '/afs/cern.ch/user/v/velicanu/mylist.json').getVLuminosityBlockRange()
# process.source.lumisToProcess = LumiList.LumiList(filename = '/afs/cern.ch/cms/CAF/CMSPHYS/PHYS_HEAVYIONS/cms/CMSSW_7_5_5_patch4/src/production/HIRun2015ForestingSetup_v0/ls100.txt').getVLuminosityBlockRange()

#####################################################################################
# Load Global Tag, Geometry, etc.
#####################################################################################

process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.Geometry.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff')
process.load('Configuration.StandardSequences.Digi_cff')
process.load('Configuration.StandardSequences.SimL1Emulator_cff')
process.load('Configuration.StandardSequences.DigiToRaw_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Configuration.StandardSequences.ReconstructionHeavyIons_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')

from Configuration.AlCa.GlobalTag_condDBv2 import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, '75X_dataRun2_ExpressHI_v2', '')



##########################################JEC##########################################
from HeavyIonsAnalysis.Configuration.CommonFunctions_cff import overrideJEC_PbPb5020
process = overrideJEC_PbPb5020(process)
##########################################JEC##########################################


##########################################UE##########################################
from CondCore.DBCommon.CondDBSetup_cfi import *
process.uetable = cms.ESSource("PoolDBESSource",
      DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(0)
        ),
      timetype = cms.string('runnumber'),
      toGet = cms.VPSet(
          cms.PSet(record = cms.string("JetCorrectionsRecord"),
                   tag = cms.string("UETableCompatibilityFormat_Calo_v00_express"),
                   label = cms.untracked.string("UETable_PF")
          ),
          cms.PSet(record = cms.string("JetCorrectionsRecord"),
                   tag = cms.string("UETableCompatibilityFormat_PF_v00_express"),
                   label = cms.untracked.string("UETable_Calo")
          )
      ), 
      connect = cms.string("frontier://FrontierProd/CMS_CONDITIONS")
)
process.es_prefer_uetable = cms.ESPrefer('PoolDBESSource','uetable')
##########################################UE##########################################



process.load("RecoHI.HiCentralityAlgos.CentralityBin_cfi")
process.centralityBin.Centrality = cms.InputTag("hiCentrality")
process.centralityBin.centralityVariable = cms.string("HFtowers")

# process.GlobalTag.snapshotTime = cms.string("9999-12-31 23:59:59.000")
# process.GlobalTag.toGet.extend([
  # cms.PSet(record = cms.string("HeavyIonRcd"),
     # tag = cms.string("CentralityTable_HFtowers200_Glauber2015A_v750x01_offline"),
     # connect = cms.string("frontier://FrontierProd/CMS_CONDITIONS"),
     # label = cms.untracked.string("HFtowers")
  # ),
# ])

#####################################################################################
# Define tree output
#####################################################################################

process.TFileService = cms.Service("TFileService",
                                  fileName=cms.string(options.outputFile))                                   
# fileName=cms.string("HiForest_test_pp_Express.root"))

#####################################################################################
# Additional Reconstruction and Analysis: Main Body
#####################################################################################



process.load('HeavyIonsAnalysis.JetAnalysis.jets.akPu3CaloJetSequence_PbPb_data_bTag_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akVs3CaloJetSequence_PbPb_data_bTag_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akPu3PFJetSequence_PbPb_data_bTag_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akVs3PFJetSequence_PbPb_data_bTag_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akPu4CaloJetSequence_PbPb_data_bTag_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akVs4CaloJetSequence_PbPb_data_bTag_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akPu4PFJetSequence_PbPb_data_bTag_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akVs4PFJetSequence_PbPb_data_bTag_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akPu5CaloJetSequence_PbPb_data_bTag_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akVs5CaloJetSequence_PbPb_data_bTag_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akPu5PFJetSequence_PbPb_data_bTag_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akVs5PFJetSequence_PbPb_data_bTag_cff')

process.PureTracks = cms.EDFilter("TrackSelector",
                       src = cms.InputTag("hiGeneralTracks"),
                       cut = cms.string('quality("highPurity")'))



process.jetSequences = cms.Sequence(
# process.ak3CaloJetSequence +
                                    # process.ak3PFJetSequence +
                                    process.PureTracks +
                                    process.akVs3CaloJetSequence +
                                    process.akPu3CaloJetSequence +
                                    process.akVs3PFJetSequence +
                                    process.akPu3PFJetSequence +
                                    process.akVs4CaloJetSequence +
                                    process.akPu4CaloJetSequence +
                                    process.akVs4PFJetSequence +
                                    process.akPu4PFJetSequence +
                                    process.akVs5CaloJetSequence +
                                    process.akPu5CaloJetSequence +
                                    process.akVs5PFJetSequence +
                                    process.akPu5PFJetSequence
                                    # process.ak4CaloJetSequence +
                                    # process.ak4PFJetSequence

                                    # process.akPu5CaloJetSequence +
                                    # process.akVs5CaloJetSequence +
                                    # process.akVs5PFJetSequence +
                                    # process.akPu5PFJetSequence

                                    )
                                    
                                    
process.load('HeavyIonsAnalysis.EventAnalysis.hievtanalyzer_data_cfi')
process.load('HeavyIonsAnalysis.EventAnalysis.hltanalysis_cff')
from HeavyIonsAnalysis.EventAnalysis.dummybranches_cff import addHLTdummybranches
addHLTdummybranches(process)

############ hlt oject
process.load("HeavyIonsAnalysis.EventAnalysis.hltobject_cfi")
process.load("HLTrigger.HLTanalyzers.HLTBitAnalyser_cfi")

#####################################################################################
# To be cleaned

process.load('HeavyIonsAnalysis.JetAnalysis.ExtraTrackReco_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.TrkAnalyzers_cff')
process.load("HeavyIonsAnalysis.TrackAnalysis.METAnalyzer_cff")
process.load("HeavyIonsAnalysis.JetAnalysis.pfcandAnalyzer_cfi")
process.pfcandAnalyzer.skipCharged = False
process.pfcandAnalyzer.pfPtMin = 0

#####################################################################################
#########################
# HLTMuTree Analyzer
#########################
process.load('HeavyIonsAnalysis.MuonAnalysis.hltMuTree_cfi')
# process.hltMuTree.vertices = cms.InputTag("offlinePrimaryVertices")
process.hltMuTree.vertices = cms.InputTag("hiSelectedVertex")

#########################
# Track Analyzer
#########################
process.anaTrack.qualityStrings = cms.untracked.vstring(['highPurity','tight','loose'])

process.pixelTrack.qualityStrings = cms.untracked.vstring('highPurity')
process.hiTracks.cut = cms.string('quality("highPurity")')

# set track collection to iterative tracking
process.anaTrack.trackSrc = cms.InputTag("hiGeneralTracks")


#####################
# L1 Digis
#####################

process.load('L1Trigger.L1TCalorimeter.caloConfigStage1HI_cfi')
process.load('L1Trigger.L1TCalorimeter.L1TCaloStage1_cff')
process.load('L1Trigger.L1TCalorimeter.caloStage1Params_HI_cfi')
process.caloStage1Params.minimumBiasThresholds = cms.vint32(1,1,2,2)

process.simRctUpgradeFormatDigis.emTag = cms.InputTag("caloStage1Digis")
process.simRctUpgradeFormatDigis.regionTag = cms.InputTag("caloStage1Digis")

process.load('EventFilter.L1TRawToDigi.caloStage1Digis_cfi')
process.caloStage1Digis.InputLabel = cms.InputTag("rawDataRepacker")

process.L1Sequence = cms.Sequence(
    process.caloStage1Digis +
    process.simRctUpgradeFormatDigis +
    process.simCaloStage1Digis +
    process.simCaloStage1FinalDigis
    )



process.EmulatorResults = cms.EDAnalyzer('l1t::L1UpgradeAnalyzer',
                                         InputLayer2Collection = cms.InputTag("simCaloStage1FinalDigis"),
                                         InputLayer2TauCollection = cms.InputTag("simCaloStage1FinalDigis:rlxTaus"),
                                         InputLayer2IsoTauCollection = cms.InputTag("simCaloStage1FinalDigis:isoTaus"),
                                         InputLayer2CaloSpareCollection = cms.InputTag("simCaloStage1FinalDigis:HFRingSums"),
                                         InputLayer2HFBitCountCollection = cms.InputTag("simCaloStage1FinalDigis:HFBitCounts"),
                                         InputLayer1Collection = cms.InputTag("None"),
                                         legacyRCTDigis = cms.InputTag("None")
)

process.UnpackerResults = cms.EDAnalyzer('l1t::L1UpgradeAnalyzer',
                                         InputLayer2Collection = cms.InputTag("caloStage1Digis"),
                                         InputLayer2TauCollection = cms.InputTag("caloStage1Digis:rlxTaus"),
                                         InputLayer2IsoTauCollection = cms.InputTag("caloStage1Digis:isoTaus"),
                                         InputLayer2CaloSpareCollection = cms.InputTag("caloStage1Digis:HFRingSums"),
                                         InputLayer2HFBitCountCollection = cms.InputTag("caloStage1Digis:HFBitCounts"),
                                         InputLayer1Collection = cms.InputTag("simRctUpgradeFormatDigis"),
                                         legacyRCTDigis = cms.InputTag("caloStage1Digis")
)


process.L1EmulatorUnpacker = cms.Sequence(process.UnpackerResults + process.EmulatorResults)

AddCaloMuon = False
runOnMC = False
HIFormat = False
UseGenPlusSim = False
VtxLabel = "hiSelectedVertex"
TrkLabel = "hiGeneralTracks"
from Bfinder.finderMaker.finderMaker_75X_cff import finderMaker_75X
finderMaker_75X(process, AddCaloMuon, runOnMC, HIFormat, UseGenPlusSim, VtxLabel, TrkLabel)


process.load('HeavyIonsAnalysis.JetAnalysis.rechitanalyzer_cfi')
process.rechitanalyzer.doVS = cms.untracked.bool(True)
process.rechitanalyzer.useJets = cms.untracked.bool(False)
process.rechitanalyzer.EBTreePtMin = -9999
process.rechitanalyzer.EETreePtMin = -9999
process.rechitanalyzer.HBHETreePtMin = -9999
process.rechitanalyzer.HFTreePtMin = -9999
process.rechitanalyzer.HFlongMin = -9999
process.rechitanalyzer.HFshortMin = -9999
process.rechitanalyzer.HFtowerMin = -9999

process.load("HeavyIonsAnalysis.JetAnalysis.hcalNoise_cff")




#####################

# photons
process.load('HeavyIonsAnalysis.PhotonAnalysis.ggHiNtuplizer_cfi')
process.ggHiNtuplizer.doGenParticles = False
process.ggHiNtuplizerGED = process.ggHiNtuplizer.clone(recoPhotonSrc = cms.InputTag('gedPhotonsTmp'),
                                                       recoPhotonHiIsolationMap = cms.InputTag('photonIsolationHIProducerGED')
                                                       )


# re-reco zdc from the new castorDigis
process.load('RecoLocalCalo.HcalRecProducers.HcalHitReconstructor_zdc_cfi')
process.zdcreco.digiLabel = cms.InputTag("castorDigis") # was hcalDigis
# redo centrality
process.load('RecoHI.HiCentralityAlgos.HiCentrality_cfi')
process.newHiCentrality = process.hiCentrality.clone()
process.newHiCentrality.produceHFhits = cms.bool(False)
process.newHiCentrality.produceHFtowers = cms.bool(False)
process.newHiCentrality.produceEcalhits = cms.bool(False)
process.newHiCentrality.produceZDChits = cms.bool(True)
process.newHiCentrality.produceETmidRapidity = cms.bool(False)
process.newHiCentrality.producePixelhits = cms.bool(False)
process.newHiCentrality.produceTracks = cms.bool(False)
process.newHiCentrality.producePixelTracks = cms.bool(False)
process.newHiCentrality.reUseCentrality = cms.bool(True)
process.newHiCentrality.srcReUse = cms.InputTag("hiCentrality")

process.hiEvtAnalyzer.CentralitySrc = cms.InputTag("newHiCentrality")
                                                       
                                                       
process.ana_step = cms.Path(
                            process.hltanalysis *
                            # process.hltobject +
                            process.zdcreco *
                            process.newHiCentrality * 
                            process.centralityBin * 
                            process.hiEvtAnalyzer*
                            process.jetSequences +
                            process.hcalNoise +
                            process.ggHiNtuplizer +
                            process.ggHiNtuplizerGED +
                            process.pfcandAnalyzer +
                            process.L1Sequence +
                            process.L1EmulatorUnpacker +
                            process.finderSequence +
                            process.rechitanalyzer +
                            process.hltMuTree + 
                            process.HiForest +
                            process.anaTrack
                            )


#####################################################################################
# PAcollisionEventSelection stuff
#####################################################################################
# process.load('HeavyIonsAnalysis.Configuration.hfCoincFilter_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.EventSelection_cff')
process.load('HeavyIonsAnalysis.Configuration.collisionEventSelection_cff')
# process.PAprimaryVertexFilter = cms.EDFilter("VertexSelector",
    # src = cms.InputTag("offlinePrimaryVertices"),
    # cut = cms.string("!isFake && abs(z) <= 25 && position.Rho <= 2 && tracksSize >= 2"),
    # filter = cms.bool(True), # otherwise it won't filter the events
# )

# process.NoScraping = cms.EDFilter("FilterOutScraping",
 # applyfilter = cms.untracked.bool(True),
 # debugOn = cms.untracked.bool(False),
 # numtrack = cms.untracked.uint32(10),
 # thresh = cms.untracked.double(0.25)
# )

# process.load('RecoHI.HiCentralityAlgos.HiClusterCompatibility_cfi')
# process.load('HeavyIonsAnalysis.EventAnalysis.HIClusterCompatibilityFilter_cfi')
process.clusterCompatibilityFilter.clusterPars = cms.vdouble(0.0,0.006)

# process.PAcollisionEventSelection = cms.Sequence(process.hfCoincFilter *
                                         # process.PAprimaryVertexFilter *
                                         # process.NoScraping 
                                         # )

process.phltJetHI = cms.Path( process.hltJetHI )
# process.primaryVertexFilter.src = cms.InputTag("offlinePrimaryVertices")
process.pcollisionEventSelection = cms.Path(process.collisionEventSelection)
process.load('CommonTools.RecoAlgos.HBHENoiseFilterResultProducer_cfi')
process.pHBHENoiseFilterResultProducer = cms.Path( process.HBHENoiseFilterResultProducer )

process.pClusterCompaitiblityFilter = cms.Path( process.clusterCompatibilityFilter)
process.pprimaryVertexFilter = cms.Path(process.primaryVertexFilter)

process.phfCoincFilter = cms.Path(process.hfCoincFilter )
process.phfCoincFilter3 = cms.Path(process.hfCoincFilter3 )

process.phfPosFilter3 = cms.Path(process.towersAboveThreshold+process.hfPosTowers+process.hfNegTowers+process.hfPosFilter3)
process.phfNegFilter3 = cms.Path(process.towersAboveThreshold+process.hfPosTowers+process.hfNegTowers+process.hfNegFilter3)
process.hfPosFilter2 = process.hfPosFilter.clone(minNumber=cms.uint32(2))
process.hfNegFilter2 = process.hfNegFilter.clone(minNumber=cms.uint32(2))
process.phfPosFilter2 = cms.Path(process.towersAboveThreshold+process.hfPosTowers+process.hfNegTowers+process.hfPosFilter2)
process.phfNegFilter2 = cms.Path(process.towersAboveThreshold+process.hfPosTowers+process.hfNegTowers+process.hfNegFilter2)
process.phfPosFilter1 = cms.Path(process.towersAboveThreshold+process.hfPosTowers+process.hfNegTowers+process.hfPosFilter)
process.phfNegFilter1 = cms.Path(process.towersAboveThreshold+process.hfPosTowers+process.hfNegTowers+process.hfNegFilter)





process.pAna = cms.EndPath(process.skimanalysis)
# process.pAna = cms.Path(process.skimanalysis)

# Customization

# import HLTrigger.HLTfilters.hltHighLevel_cfi

# process.hltMinBias = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
# process.hltMinBias.HLTPaths = ["HLT_HIL1MinimumBiasHF1AND_v1"]


# filter all path with the production filter sequence
# for path in process.paths:
    # getattr(process,path)._seq = process.hltMinBias * getattr(process,path)._seq

    
    
