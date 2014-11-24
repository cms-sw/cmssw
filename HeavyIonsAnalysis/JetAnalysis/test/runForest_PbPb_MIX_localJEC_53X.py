#!/usr/bin/env python2
# Run the foresting configuration on PbPb in CMSSW_5_3_X, using the new HF/Voronoi jets
# Author: Alex Barbieri
# Date: 2013-10-15

hiTrackQuality = "highPurity"              # iterative tracks
#hiTrackQuality = "highPuritySetWithPV"    # calo-matched tracks

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
    "/store/himc/HiFall13DR53X/Pyquen_DiJet_Pt370_TuneZ2_Unquenched_Hydjet1p8_2760GeV/GEN-SIM-RECO/NoPileUp_STARTHI53_LV1-v4/00000/024C7240-DCDD-E311-882D-7845C4FC3B1B.root"
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
process.load('FWCore.MessageService.MessageLogger_cfi')

# PbPb 53X MC
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'STARTHI53_LV1::All', '')

from HeavyIonsAnalysis.Configuration.CommonFunctionsLocalDB_cff import *
overrideGT_PbPb2760(process)

process.HeavyIonGlobalParameters = cms.PSet(
    centralityVariable = cms.string("HFtowers"),
    nonDefaultGlauberModel = cms.string("Hydjet_Drum"),
    centralitySrc = cms.InputTag("hiCentrality")
    )

#####################################################################################
# Define tree output
#####################################################################################

process.TFileService = cms.Service("TFileService",
                                   fileName=cms.string("hiForest_DijetpT370_Hydjet1p8_STARTHI53_LV1_v15.root"))

#####################################################################################
# Additional Reconstruction and Analysis: Main Body
#####################################################################################

process.load('HeavyIonsAnalysis.JetAnalysis.jets.HiGenJetsCleaned_cff')

process.load('HeavyIonsAnalysis.JetAnalysis.jets.akPu1CaloJetSequence_PbPb_mix_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akVs1CaloJetSequence_PbPb_mix_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akVs1PFJetSequence_PbPb_mix_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akPu1PFJetSequence_PbPb_mix_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.ak1PFJetSequence_PbPb_mix_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.ak1CaloJetSequence_PbPb_mix_cff')

process.load('HeavyIonsAnalysis.JetAnalysis.jets.akPu2CaloJetSequence_PbPb_mix_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akVs2CaloJetSequence_PbPb_mix_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akVs2PFJetSequence_PbPb_mix_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akPu2PFJetSequence_PbPb_mix_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.ak2PFJetSequence_PbPb_mix_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.ak2CaloJetSequence_PbPb_mix_cff')

process.load('HeavyIonsAnalysis.JetAnalysis.jets.akPu3CaloJetSequence_PbPb_mix_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akVs3CaloJetSequence_PbPb_mix_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akVs3PFJetSequence_PbPb_mix_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akPu3PFJetSequence_PbPb_mix_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.ak3PFJetSequence_PbPb_mix_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.ak3CaloJetSequence_PbPb_mix_cff')

process.load('HeavyIonsAnalysis.JetAnalysis.jets.akPu4CaloJetSequence_PbPb_mix_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akVs4CaloJetSequence_PbPb_mix_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akVs4PFJetSequence_PbPb_mix_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akPu4PFJetSequence_PbPb_mix_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.ak4PFJetSequence_PbPb_mix_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.ak4CaloJetSequence_PbPb_mix_cff')

process.load('HeavyIonsAnalysis.JetAnalysis.jets.akPu5CaloJetSequence_PbPb_mix_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akVs5CaloJetSequence_PbPb_mix_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akVs5PFJetSequence_PbPb_mix_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akPu5PFJetSequence_PbPb_mix_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.ak5PFJetSequence_PbPb_mix_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.ak5CaloJetSequence_PbPb_mix_cff')

process.load('HeavyIonsAnalysis.JetAnalysis.jets.akPu6CaloJetSequence_PbPb_mix_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akVs6CaloJetSequence_PbPb_mix_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akVs6PFJetSequence_PbPb_mix_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akPu6PFJetSequence_PbPb_mix_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.ak6PFJetSequence_PbPb_mix_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.ak6CaloJetSequence_PbPb_mix_cff')

process.load('HeavyIonsAnalysis.JetAnalysis.jets.akPu7CaloJetSequence_PbPb_mix_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akVs7CaloJetSequence_PbPb_mix_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akVs7PFJetSequence_PbPb_mix_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akPu7PFJetSequence_PbPb_mix_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.ak7PFJetSequence_PbPb_mix_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.ak7CaloJetSequence_PbPb_mix_cff')

process.load('HeavyIonsAnalysis.JetAnalysis.jets.HiReRecoJets_HI_cff')

process.jetSequences = cms.Sequence(process.hiReRecoCaloJets +
                                    process.hiReRecoPFJets +

                                    process.akPu1CaloJetSequence +
                                    process.akVs1CaloJetSequence +
                                    process.akVs1PFJetSequence +
                                    process.akPu1PFJetSequence +
                                    process.ak1PFJetSequence +
                                    process.ak1CaloJetSequence +

                                    process.akPu2CaloJetSequence +
                                    process.akVs2CaloJetSequence +
                                    process.akVs2PFJetSequence +
                                    process.akPu2PFJetSequence +
                                    process.ak2PFJetSequence +
                                    process.ak2CaloJetSequence +

                                    process.akPu3CaloJetSequence +
                                    process.akVs3CaloJetSequence +
                                    process.akVs3PFJetSequence +
                                    process.akPu3PFJetSequence +
                                    process.ak3PFJetSequence +
                                    process.ak3CaloJetSequence +

                                    process.akPu4CaloJetSequence +
                                    process.akVs4CaloJetSequence +
                                    process.akVs4PFJetSequence +
                                    process.akPu4PFJetSequence +
                                    process.ak4PFJetSequence +
                                    process.ak4CaloJetSequence +

                                    process.akPu5CaloJetSequence +
                                    process.akVs5CaloJetSequence +
                                    process.akVs5PFJetSequence +
                                    process.akPu5PFJetSequence +
                                    process.ak5PFJetSequence +
                                    process.ak5CaloJetSequence +

                                    process.akPu6CaloJetSequence +
                                    process.akVs6CaloJetSequence +
                                    process.akVs6PFJetSequence +
                                    process.akPu6PFJetSequence +
                                    process.ak6PFJetSequence +
                                    process.ak6CaloJetSequence +

                                    process.akPu7CaloJetSequence +
                                    process.akVs7CaloJetSequence +
                                    process.akVs7PFJetSequence +
                                    process.akPu7PFJetSequence +
                                    process.ak7PFJetSequence +
                                    process.ak7CaloJetSequence
                                    )

process.load('HeavyIonsAnalysis.EventAnalysis.hievtanalyzer_mc_cfi')
process.load('HeavyIonsAnalysis.EventAnalysis.hltanalysis_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.HiGenAnalyzer_cfi')

#####################################################################################
# To be cleaned

process.load('HeavyIonsAnalysis.JetAnalysis.ExtraTrackReco_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.TrkAnalyzers_MC_cff')
process.load("HeavyIonsAnalysis.TrackAnalysis.METAnalyzer_cff")
process.load("HeavyIonsAnalysis.JetAnalysis.pfcandAnalyzer_cfi")
process.load('HeavyIonsAnalysis.JetAnalysis.rechitanalyzer_cfi')
process.rechitAna = cms.Sequence(process.rechitanalyzer+process.pfTowers)
process.pfcandAnalyzer.skipCharged = False
process.pfcandAnalyzer.pfPtMin = 0

#####################################################################################

#########################
# Track Analyzer
#########################
process.anaTrack.qualityStrings = cms.untracked.vstring('highPurity','highPuritySetWithPV')
process.pixelTrack.qualityStrings = cms.untracked.vstring('highPurity','highPuritySetWithPV')
process.hiTracks.cut = cms.string('quality("highPurity")')

# set track collection to iterative tracking
process.anaTrack.trackSrc = cms.InputTag("hiGeneralTracks")

# clusters missing in recodebug - to be resolved
process.anaTrack.doPFMatching = False
process.pixelTrack.doPFMatching = False

#####################
# photons
process.load('HeavyIonsAnalysis.JetAnalysis.EGammaAnalyzers_cff')
process.multiPhotonAnalyzer.GenEventScale = cms.InputTag("hiSignal")
process.multiPhotonAnalyzer.HepMCProducer = cms.InputTag("hiSignal")
process.RandomNumberGeneratorService.multiPhotonAnalyzer = process.RandomNumberGeneratorService.generator.clone()

#####################
# muons
######################
process.load("HeavyIonsAnalysis.MuonAnalysis.hltMuTree_cfi")
process.hltMuTree.doGen = cms.untracked.bool(True)
process.load("RecoHI.HiMuonAlgos.HiRecoMuon_cff")
process.muons.JetExtractorPSet.JetCollectionLabel = cms.InputTag("akVs3PFJets")
process.globalMuons.TrackerCollectionLabel = "hiGeneralTracks"
process.muons.TrackExtractorPSet.inputTrackCollection = "hiGeneralTracks"
process.muons.inputCollectionLabels = ["hiGeneralTracks", "globalMuons", "standAloneMuons:UpdatedAtVtx", "tevMuons:firstHit", "tevMuons:picky", "tevMuons:dyt"]

# HYDJET RECO file didn't have ak1GenJets
process.load('Configuration.StandardSequences.Generator_cff')

# required to re-run ak1HiGenJets
process.load('RecoHI.HiJetAlgos.HiGenJets_cff')
process.genStep = cms.Path(process.hiGenParticlesForJets +
                           process.ak1HiGenJets)

process.ana_step = cms.Path(process.heavyIon*
                            process.hltanalysis *
#temp                            process.hltobject *
                            process.hiEvtAnalyzer*
                            process.HiGenParticleAna*
                            process.hiGenJetsCleaned*
                            process.jetSequences +
                            process.photonStep_withReco +
                            process.pfcandAnalyzer +
                            process.rechitAna +
#temp                            process.hltMuTree +
                            process.HiForest +
                            process.cutsTPForFak +
                            process.cutsTPForEff +
                            process.anaTrack +
                            process.pixelTrack
                            )

process.load('HeavyIonsAnalysis.JetAnalysis.EventSelection_cff')
process.phltJetHI = cms.Path( process.hltJetHI )
process.pcollisionEventSelection = cms.Path(process.collisionEventSelection)
process.pHBHENoiseFilter = cms.Path( process.HBHENoiseFilter )
process.phfCoincFilter = cms.Path(process.hfCoincFilter )
process.phfCoincFilter3 = cms.Path(process.hfCoincFilter3 )
process.pprimaryVertexFilter = cms.Path(process.primaryVertexFilter )
process.phltPixelClusterShapeFilter = cms.Path(process.siPixelRecHits*process.hltPixelClusterShapeFilter )
process.phiEcalRecHitSpikeFilter = cms.Path(process.hiEcalRecHitSpikeFilter )

process.pAna = cms.EndPath(process.skimanalysis)

# Customization
