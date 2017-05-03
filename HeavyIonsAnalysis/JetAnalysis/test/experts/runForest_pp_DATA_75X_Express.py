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
process.HiForest.HiForestVersion = cms.untracked.string(version)

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

from HeavyIonsAnalysis.Configuration.CommonFunctions_cff import overrideJEC_pp5020
process = overrideJEC_pp5020(process)


#for pp data create centrality object and bin
process.load("RecoHI.HiCentralityAlgos.pACentrality_cfi")
process.pACentrality.producePixelTracks = cms.bool(False)
process.load("RecoHI.HiCentralityAlgos.CentralityBin_cfi")
process.centralityBin.Centrality = cms.InputTag("pACentrality")
process.centralityBin.centralityVariable = cms.string("HFtowers")

#####################################################################################
# Define tree output
#####################################################################################

process.TFileService = cms.Service("TFileService",
                                  fileName=cms.string(options.outputFile))                                   
# fileName=cms.string("HiForest_test_pp_Express.root"))

#####################################################################################
# Additional Reconstruction and Analysis: Main Body
#####################################################################################



# process.load('HeavyIonsAnalysis.JetAnalysis.jets.ak3CaloJetSequence_pp_data_cff')
# process.load('HeavyIonsAnalysis.JetAnalysis.jets.ak4CaloJetSequence_pp_data_cff')
# process.load('HeavyIonsAnalysis.JetAnalysis.jets.ak5CaloJetSequence_pp_data_cff')

# process.load('HeavyIonsAnalysis.JetAnalysis.jets.ak3PFJetSequence_pp_data_cff')
# process.load('HeavyIonsAnalysis.JetAnalysis.jets.ak4PFJetSequence_pp_data_cff')
# process.load('HeavyIonsAnalysis.JetAnalysis.jets.ak5PFJetSequence_pp_data_cff')

process.load('HeavyIonsAnalysis.JetAnalysis.jets.ak4CaloJetSequence_pp_data_bTag_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.ak4PFJetSequence_pp_data_bTag_cff')

process.jetSequences = cms.Sequence(
# process.ak3CaloJetSequence +
                                    # process.ak3PFJetSequence +

                                    process.ak4CaloJetSequence +
                                    process.ak4PFJetSequence

                                    # process.akPu5CaloJetSequence +
                                    # process.akVs5CaloJetSequence +
                                    # process.akVs5PFJetSequence +
                                    # process.akPu5PFJetSequence

                                    )
                                    
                                    
process.load('HeavyIonsAnalysis.EventAnalysis.hievtanalyzer_data_cfi')
process.hiEvtAnalyzer.CentralitySrc = cms.InputTag("pACentrality")
process.hiEvtAnalyzer.Vertex = cms.InputTag("offlinePrimaryVertices")
process.hiEvtAnalyzer.doEvtPlane = cms.bool(False)
process.load('HeavyIonsAnalysis.EventAnalysis.hltanalysis_cff')
############ hlt oject
process.load("HeavyIonsAnalysis.EventAnalysis.hltobject_cfi")
process.load("HLTrigger.HLTanalyzers.HLTBitAnalyser_cfi")
process.hltbitanalysis.l1GtReadoutRecord = cms.InputTag("gtDigis","","HLT")


#####################                                                                      \               
#TUPEL                                                                                                     
#####################                                                                      \              


process.load('HeavyIonsAnalysis.VectorBosonAnalysis.tupel_cfi')
process.tupel.vtxSrc     = cms.untracked.InputTag("offlinePrimaryVertices")
process.tupel.jetSrc      = cms.untracked.InputTag("ak4PFpatJetsWithBtagging")

#####################

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
process.hltMuTree.vertices = cms.InputTag("offlinePrimaryVertices")

#########################
# Track Analyzer
#########################
# process.ppTrack.qualityStrings = cms.untracked.vstring(['highPurity'])
process.ppTrack.trackSrc = cms.InputTag("generalTracks")
# process.ppTrack.doMVA = cms.untracked.bool(False)
process.ppTrack.mvaSrc = cms.string("generalTracks")
# process.ppTrack.doPFMatching = cms.untracked.bool(True)
process.ppTrack.pfCandSrc = cms.InputTag('particleFlow')

#####################
# L1 Digis
#####################

process.load('EventFilter.L1TRawToDigi.caloStage1Digis_cfi')

process.load('L1Trigger.L1TCalorimeter.caloConfigStage1HI_cfi')
process.load('L1Trigger.L1TCalorimeter.L1TCaloStage1_PPFromRaw_cff')

### nominal
process.load('L1Trigger.L1TCalorimeter.caloStage1Params_cfi')
### PUS mask
process.caloStage1Params.jetRegionMask = cms.int32(0b0000100000000000010000)
#process.caloStage1Params.jetRegionMask = cms.int32(0)
### EG 'iso' (eta) mask
process.caloStage1Params.egEtaCut = cms.int32(0b0000001111111111000000)
### Single track eta mask
process.caloStage1Params.tauRegionMask = cms.int32(0b1111111100000011111111)
### Centrality eta mask
process.caloStage1Params.centralityRegionMask = cms.int32(0b0000111111111111110000)
### jet seed threshold for 3x3 step of jet finding
process.caloStage1Params.jetSeedThreshold = cms.double(0)
### HTT settings (this won't match anyway yet)
process.caloStage1Params.etSumEtThreshold        = cms.vdouble(0., 7.) #ET, HT
### Minimum Bias thresholds
process.caloStage1Params.minimumBiasThresholds = cms.vint32(4,4,6,6)
### Centrality LUT
# process.caloStage1Params.centralityLUTFile = cms.FileInPath("L1Trigger/L1TCalorimeter/data/centrality_extended_LUT_preRun.txt")

process.L1Sequence = cms.Sequence(
    process.L1TCaloStage1_PPFromRaw +
    process.caloStage1Digis
    )

process.EmulatorResults = cms.EDAnalyzer('l1t::L1UpgradeAnalyzer',
                                         InputLayer2Collection = cms.InputTag("simCaloStage1FinalDigis"),
                                         InputLayer2TauCollection = cms.InputTag("simCaloStage1FinalDigis:rlxTaus"),
                                         InputLayer2IsoTauCollection = cms.InputTag("simCaloStage1FinalDigis:isoTaus"),
                                         InputLayer2CaloSpareCollection = cms.InputTag("simCaloStage1FinalDigis:HFRingSums"),
                                         InputLayer2HFBitCountCollection = cms.InputTag("simCaloStage1FinalDigis:HFBitCounts"),
                                         InputLayer1Collection = cms.InputTag("simRctUpgradeFormatDigis"),
                                         legacyRCTDigis = cms.InputTag("simRctDigis")
)

process.UnpackerResults = cms.EDAnalyzer('l1t::L1UpgradeAnalyzer',
                                         InputLayer2Collection = cms.InputTag("caloStage1Digis"),
                                         InputLayer2TauCollection = cms.InputTag("caloStage1Digis:rlxTaus"),
                                         InputLayer2IsoTauCollection = cms.InputTag("caloStage1Digis:isoTaus"),
                                         InputLayer2CaloSpareCollection = cms.InputTag("caloStage1Digis:HFRingSums"),
                                         InputLayer2HFBitCountCollection = cms.InputTag("caloStage1Digis:HFBitCounts"),
                                         InputLayer1Collection = cms.InputTag("None"),
                                         legacyRCTDigis = cms.InputTag("caloStage1Digis")
)


process.L1EmulatorUnpacker = cms.Sequence(process.EmulatorResults + process.UnpackerResults)

AddCaloMuon = False
runOnMC = False
HIFormat = False
UseGenPlusSim = False
VtxLabel = "offlinePrimaryVerticesWithBS"
TrkLabel = "generalTracks"
from Bfinder.finderMaker.finderMaker_75X_cff import finderMaker_75X
finderMaker_75X(process, AddCaloMuon, runOnMC, HIFormat, UseGenPlusSim, VtxLabel, TrkLabel)


process.load('HeavyIonsAnalysis.JetAnalysis.rechitanalyzer_pp_cfi')
process.rechitanalyzer.doVS = cms.untracked.bool(False)
process.rechitanalyzer.useJets = cms.untracked.bool(False)


#####################

# photons
process.load('HeavyIonsAnalysis.PhotonAnalysis.ggHiNtuplizer_cfi')
process.ggHiNtuplizer.gsfElectronLabel   = cms.InputTag("gedGsfElectrons")
process.ggHiNtuplizer.useValMapIso       = cms.bool(False)
process.ggHiNtuplizer.VtxLabel           = cms.InputTag("offlinePrimaryVerticesWithBS")
process.ggHiNtuplizer.particleFlowCollection = cms.InputTag("particleFlow")
process.ggHiNtuplizer.doVsIso            = cms.bool(False)
process.ggHiNtuplizer.doGenParticles = False
process.ggHiNtuplizerGED = process.ggHiNtuplizer.clone(recoPhotonSrc = cms.InputTag('gedPhotons'))

###############################################################
process.pfcandAnalyzer.pfCandidateLabel = cms.InputTag("particleFlow")
process.pfcandAnalyzer.doVS = cms.untracked.bool(False)
process.pfcandAnalyzer.doUEraw_ = cms.untracked.bool(False)
process.pfcandAnalyzer.genLabel = cms.InputTag("genParticles")


process.ana_step = cms.Path(
                            process.hltanalysis *
                            process.hltobject +
                            process.siPixelRecHits * process.pACentrality * process.centralityBin * #for pp data only on reco
                            process.hiEvtAnalyzer*
                            process.jetSequences +
                            process.ggHiNtuplizer +
                            process.ggHiNtuplizerGED +
                            process.tupel +
                            process.pfcandAnalyzer +
                            process.L1Sequence +
                            process.L1EmulatorUnpacker +
                            process.finderSequence +
                            process.rechitanalyzer +
                            process.hltMuTree + 
                            process.HiForest +
                            process.ppTrack
                            )


#####################################################################################
# PAcollisionEventSelection stuff
#####################################################################################
process.load('HeavyIonsAnalysis.Configuration.hfCoincFilter_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.EventSelection_cff')
process.PAprimaryVertexFilter = cms.EDFilter("VertexSelector",
    src = cms.InputTag("offlinePrimaryVertices"),
    cut = cms.string("!isFake && abs(z) <= 25 && position.Rho <= 2 && tracksSize >= 2"),
    filter = cms.bool(True), # otherwise it won't filter the events
)

process.NoScraping = cms.EDFilter("FilterOutScraping",
 applyfilter = cms.untracked.bool(True),
 debugOn = cms.untracked.bool(False),
 numtrack = cms.untracked.uint32(10),
 thresh = cms.untracked.double(0.25)
)

process.load('RecoHI.HiCentralityAlgos.HiClusterCompatibility_cfi')
process.load('HeavyIonsAnalysis.EventAnalysis.HIClusterCompatibilityFilter_cfi')
process.clusterCompatibilityFilter.clusterPars = cms.vdouble(0.0,0.006)

process.PAcollisionEventSelection = cms.Sequence(process.hfCoincFilter *
                                         process.PAprimaryVertexFilter *
                                         process.NoScraping 
                                         )

process.phltJetHI = cms.Path( process.hltJetHI )
# process.primaryVertexFilter.src = cms.InputTag("offlinePrimaryVertices")
process.PAcollisionEventSelection = cms.Path(process.PAcollisionEventSelection)
process.load('CommonTools.RecoAlgos.HBHENoiseFilterResultProducer_cfi')
process.pHBHENoiseFilterResultProducer = cms.Path( process.HBHENoiseFilterResultProducer )

process.pClusterCompaitiblityFilter = cms.Path(process.siPixelRecHits*process.hiClusterCompatibility * process.clusterCompatibilityFilter)
process.pPAprimaryVertexFilter = cms.Path(process.PAprimaryVertexFilter)
process.phltPixelClusterShapeFilter = cms.Path(process.siPixelRecHits*process.hltPixelClusterShapeFilter )
process.pBeamScrapingFilter=cms.Path(process.NoScraping)

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




process.load("HeavyIonsAnalysis.VertexAnalysis.PAPileUpVertexFilter_cff")

process.pVertexFilterCutG = cms.Path(process.pileupVertexFilterCutG)
process.pVertexFilterCutGloose = cms.Path(process.pileupVertexFilterCutGloose)
process.pVertexFilterCutGtight = cms.Path(process.pileupVertexFilterCutGtight)
process.pVertexFilterCutGplus = cms.Path(process.pileupVertexFilterCutGplus)
process.pVertexFilterCutE = cms.Path(process.pileupVertexFilterCutE)
process.pVertexFilterCutEandG = cms.Path(process.pileupVertexFilterCutEandG)

process.pAna = cms.EndPath(process.skimanalysis)

# Customization
