#!/usr/bin/env python2
# Run the foresting configuration on PbPb in CMSSW_5_3_X, using the new HF/Voronoi jets
# Author: Alex Barbieri
# Date: 2013-10-15

# change this to true to switch to the Pbp JEC
secondHalfpPbJEC = False

import FWCore.ParameterSet.Config as cms
process = cms.Process('HiForest')
process.options = cms.untracked.PSet(
    # wantSummary = cms.untracked.bool(True)
    #SkipEvent = cms.untracked.vstring('ProductNotFound')
)

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
                            fileNames = cms.untracked.vstring(
                                "/store/hidata/HIRun2013A/PAHighPt/RECO/PromptReco-v1/000/210/354/CC8F3FCD-EB62-E211-ABBF-BCAEC5329716.root"
                            ))

# Number of events we want to process, -1 = all events
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10))


#####################################################################################
# Load Global Tag, Geometry, etc.
#####################################################################################

process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.Geometry.GeometryDB_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('Configuration.StandardSequences.Digi_cff')
process.load('Configuration.StandardSequences.SimL1Emulator_cff')
process.load('Configuration.StandardSequences.DigiToRaw_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Configuration.StandardSequences.ReconstructionHeavyIons_cff')
process.load('RecoHI.HiCentralityAlgos.pACentrality_cfi')
process.load('RecoHI.HiCentralityAlgos.CentralityBin_cfi')

process.load('FWCore.MessageService.MessageLogger_cfi')

# PbPb 53X Data
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'GR_P_V43D::All', '')

from HeavyIonsAnalysis.Configuration.CommonFunctions_cff import overrideCentrality
from HeavyIonsAnalysis.Configuration.CommonFunctions_cff import overrideJEC_pPb5020
from HeavyIonsAnalysis.Configuration.CommonFunctions_cff import overrideJEC_Pbp5020
overrideCentrality(process)
if secondHalfpPbJEC:
    overrideJEC_Pbp5020(process)
else :
    overrideJEC_pPb5020(process)

process.HeavyIonGlobalParameters = cms.PSet(
    centralityVariable = cms.string("HFtowersTrunc"),
    nonDefaultGlauberModel = cms.string("Hijing"),
    centralitySrc = cms.InputTag("pACentrality"),
    pPbRunFlip = cms.untracked.uint32(211313)
    )

process.pACentrality.producePixelhits = cms.bool(False)

#####################################################################################
# Define tree output
#####################################################################################

process.TFileService = cms.Service("TFileService",
                                   fileName=cms.string("HiForest.root"))

#####################################################################################
# Additional Reconstruction and Analysis: Main Body
#####################################################################################

process.hiEvtPlane.vtxCollection_ = cms.InputTag("offlinePrimaryVerticesWithBS")
process.hiEvtPlane.trackCollection_ = cms.InputTag("generalTracks")

process.load('HeavyIonsAnalysis.JetAnalysis.jets.akPu3CaloJetSequence_pPb_data_cff')
#process.load('HeavyIonsAnalysis.JetAnalysis.jets.akVs3CaloJetSequence_pPb_data_cff')
#process.load('HeavyIonsAnalysis.JetAnalysis.jets.akVs3PFJetSequence_pPb_data_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akPu3PFJetSequence_pPb_data_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.ak3PFJetSequence_pPb_data_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.ak3CaloJetSequence_pPb_data_cff')

process.load('HeavyIonsAnalysis.JetAnalysis.jets.akPu4CaloJetSequence_pPb_data_cff')
#process.load('HeavyIonsAnalysis.JetAnalysis.jets.akVs4CaloJetSequence_pPb_data_cff')
#process.load('HeavyIonsAnalysis.JetAnalysis.jets.akVs4PFJetSequence_pPb_data_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akPu4PFJetSequence_pPb_data_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.ak4PFJetSequence_pPb_data_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.ak4CaloJetSequence_pPb_data_cff')

process.load('HeavyIonsAnalysis.JetAnalysis.jets.akPu5CaloJetSequence_pPb_data_cff')
#process.load('HeavyIonsAnalysis.JetAnalysis.jets.akVs5CaloJetSequence_pPb_data_cff')
#process.load('HeavyIonsAnalysis.JetAnalysis.jets.akVs5PFJetSequence_pPb_data_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akPu5PFJetSequence_pPb_data_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.ak5PFJetSequence_pPb_data_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.ak5CaloJetSequence_pPb_data_cff')

# Difference between HiReRecoJets_pp_cff and HiReRecoJets_HI_cff is
# particle flow collection used.
process.load('HeavyIonsAnalysis.JetAnalysis.jets.HiReRecoJets_pp_cff')

process.voronoiBackgroundPF.src = cms.InputTag("particleFlow")
process.PFTowers.src = cms.InputTag("particleFlow")

process.jetSequences = cms.Sequence(process.voronoiBackgroundCalo +
                                    process.voronoiBackgroundPF +
                                    process.PFTowers +
                                    process.hiReRecoCaloJets +
                                    process.hiReRecoPFJets +

                                    process.akPu3CaloJetSequence +
                                    #process.akVs3CaloJetSequence +
                                    #process.akVs3PFJetSequence +
                                    process.akPu3PFJetSequence +
                                    process.ak3PFJetSequence +
                                    process.ak3CaloJetSequence +

                                    process.akPu4CaloJetSequence +
                                    #process.akVs4CaloJetSequence +
                                    #process.akVs4PFJetSequence +
                                    process.akPu4PFJetSequence +
                                    process.ak4PFJetSequence +
                                    process.ak4CaloJetSequence +

                                    process.akPu5CaloJetSequence +
                                    #process.akVs5CaloJetSequence +
                                    #process.akVs5PFJetSequence +
                                    process.akPu5PFJetSequence +
                                    process.ak5PFJetSequence +
                                    process.ak5CaloJetSequence

                                    )

process.load('HeavyIonsAnalysis.EventAnalysis.hievtanalyzer_data_cfi')
process.hiEvtAnalyzer.Vertex = cms.InputTag("offlinePrimaryVerticesWithBS")

#####################################################################################
# To be cleaned

process.load('HeavyIonsAnalysis.JetAnalysis.ExtraTrackReco_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.TrkAnalyzers_MC_cff')
process.load("HeavyIonsAnalysis.TrackAnalysis.METAnalyzer_cff")
process.load("HeavyIonsAnalysis.JetAnalysis.pfcandAnalyzer_pp_cfi")
process.load('HeavyIonsAnalysis.JetAnalysis.rechitanalyzer_pp_cfi')
process.rechitAna = cms.Sequence(process.rechitanalyzer+process.pfTowers)
process.pfcandAnalyzer.skipCharged = False
process.pfcandAnalyzer.pfPtMin = 0

#####################################################################################

#########################
# Track Analyzer
#########################
process.hiTracks.cut = cms.string('quality("highPurity")')

# clusters missing in recodebug - to be resolved
process.anaTrack.doPFMatching = False
process.ppTrack.doPFMatching = False
process.ppTrack.pfCandSrc = cms.InputTag("particleFlow")
process.ppTrack.doSimTrack = False

#####################
# photons
process.load('HeavyIonsAnalysis.JetAnalysis.EGammaAnalyzers_cff')
process.photonStep.remove(process.photonMatch)
process.multiPhotonAnalyzer.pfCandidateLabel = cms.InputTag("particleFlow")
process.multiPhotonAnalyzer.VertexProducer = cms.InputTag("offlinePrimaryVerticesWithBS")
process.hiGoodTracks.src = cms.InputTag("generalTracks")
process.hiGoodTracks.vertices = cms.InputTag("offlinePrimaryVerticesWithBS")
process.RandomNumberGeneratorService.multiPhotonAnalyzer = process.RandomNumberGeneratorService.generator.clone()

#####################
# muons
######################
process.load("HeavyIonsAnalysis.MuonAnalysis.hltMuTree_cfi")
process.hltMuTree.doGen = cms.untracked.bool(True)
process.load("RecoHI.HiMuonAlgos.HiRecoMuon_cff")
process.muons.JetExtractorPSet.JetCollectionLabel = cms.InputTag("akVs3PFJets")
process.globalMuons.TrackerCollectionLabel = "generalTracks"
process.muons.TrackExtractorPSet.inputTrackCollection = "generalTracks"
process.muons.inputCollectionLabels = ["generalTracks", "globalMuons", "standAloneMuons:UpdatedAtVtx", "tevMuons:firstHit", "tevMuons:picky", "tevMuons:dyt"]

process.load('HeavyIonsAnalysis.EventAnalysis.hltanalysis_cff')

#Filtering
#############################################################
# To filter on an HLT trigger path, uncomment the lines below, add the
# HLT path you would like to filter on to 'HLTPaths' and also
# uncomment the snippet at the end of the configuration.
#############################################################
# Minimum bias trigger selection (later runs)
#process.load("HLTrigger.HLTfilters.hltHighLevel_cfi")
#process.skimFilter = process.hltHighLevel.clone()
#process.skimFilter.HLTPaths = ["HLT_PixelTracks_Multiplicity*"]

#process.superFilterSequence = cms.Sequence(process.skimFilter)
#process.superFilterPath = cms.Path(process.superFilterSequence)
#process.skimanalysis.superFilters = cms.vstring("superFilterPath")
################################################################

process.hltobject.triggerNames = cms.vstring("HLT_PAJet100_NoJetID_v1","HLT_PAJet80_NoJetID_v1","HLT_PAJet60_NoJetID_v1","HLT_PAJet40_NoJetID_v1","HLT_PAJet20_NoJetID_v1")

process.ana_step = cms.Path(process.hltanalysis +
                            process.hltobject +
                            process.pACentrality +
                            process.centralityBin +
                            process.hiEvtPlane +
                            process.hiEvtAnalyzer*
                            process.jetSequences +
                            process.photonStep +
                            process.pfcandAnalyzer +
                            process.rechitAna +
#temp                            process.hltMuTree +
                            process.HiForest +
                            process.ppTrack)

process.load('HeavyIonsAnalysis.JetAnalysis.EventSelection_cff')
process.primaryVertexFilter.src = cms.InputTag("offlinePrimaryVerticesWithBS")
process.PAprimaryVertexFilter.src = cms.InputTag("offlinePrimaryVerticesWithBS")

process.phltJetHI = cms.Path( process.hltJetHI )
process.pcollisionEventSelection = cms.Path(process.collisionEventSelection)
process.pPAcollisionEventSelectionPA = cms.Path(process.PAcollisionEventSelection)
process.pHBHENoiseFilter = cms.Path( process.HBHENoiseFilter )
process.phfCoincFilter = cms.Path(process.hfCoincFilter )
process.phfCoincFilter3 = cms.Path(process.hfCoincFilter3 )
process.pprimaryVertexFilter = cms.Path(process.primaryVertexFilter )
process.pPAprimaryVertexFilter = cms.Path(process.PAprimaryVertexFilter)
process.phltPixelClusterShapeFilter = cms.Path(process.siPixelRecHits*process.hltPixelClusterShapeFilter )
process.phiEcalRecHitSpikeFilter = cms.Path(process.hiEcalRecHitSpikeFilter )
process.phfPosFilter3 = cms.Path(process.towersAboveThreshold+process.hfPosTowers+process.hfNegTowers+process.hfPosFilter3)
process.phfNegFilter3 = cms.Path(process.towersAboveThreshold+process.hfPosTowers+process.hfNegTowers+process.hfNegFilter3)
#process.phfPosFilter2 = cms.Path(process.towersAboveThreshold+process.hfPosTowers+process.hfNegTowers+process.hfPosFilter2)
#process.phfNegFilter2 = cms.Path(process.towersAboveThreshold+process.hfPosTowers+process.hfNegTowers+process.hfNegFilter2)
#process.phfPosFilter1 = cms.Path(process.towersAboveThreshold+process.hfPosTowers+process.hfNegTowers+process.hfPosFilter)
#process.phfNegFilter1 = cms.Path(process.towersAboveThreshold+process.hfPosTowers+process.hfNegTowers+process.hfNegFilter)
process.pBeamScrapingFilter=cms.Path(process.NoScraping)

process.load("HeavyIonsAnalysis.VertexAnalysis.PAPileUpVertexFilter_cff")

process.pVertexFilterCutG = cms.Path(process.pileupVertexFilterCutG)
process.pVertexFilterCutGloose = cms.Path(process.pileupVertexFilterCutGloose)
process.pVertexFilterCutGtight = cms.Path(process.pileupVertexFilterCutGtight)
process.pVertexFilterCutGplus = cms.Path(process.pileupVertexFilterCutGplus)
process.pVertexFilterCutE = cms.Path(process.pileupVertexFilterCutE)
process.pVertexFilterCutEandG = cms.Path(process.pileupVertexFilterCutEandG)

# Customization

process.pAna = cms.EndPath(process.skimanalysis)

#Filtering
#for path in process.paths:
#    getattr(process,path)._seq = process.superFilterSequence*getattr(process,path)._seq
