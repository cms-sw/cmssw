#!/usr/bin/env python2
# Run the foresting configuration on PbPb in CMSSW_5_3_X, using the new HF/Voronoi jets
# Author: Alex Barbieri
# Date: 2013-10-15

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
    "file:MC_RECO_1_1_3t3.root"
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
process.load('RecoHI.HiCentralityAlgos.CentralityBin_cfi')

process.load('FWCore.MessageService.MessageLogger_cfi')

# PbPb 53X MC
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'STARTHI53_V28::All', '')

from HeavyIonsAnalysis.Configuration.CommonFunctionsLocalDBforPPJEC_cff import *
overrideGT_pp2760(process)
overrideJEC_pp2760(process)

#####################################################################################
# Define tree output
#####################################################################################

process.TFileService = cms.Service("TFileService",
                                   fileName=cms.string("HiForest.root"))

#####################################################################################
# Additional Reconstruction and Analysis: Main Body
#####################################################################################

process.load('Configuration.StandardSequences.Generator_cff')
process.load('RecoJets.Configuration.GenJetParticles_cff')
#process.load('RecoHI.HiJetAlgos.HiGenJets_cff')
#process.load('RecoHI.HiJetAlgos.HiRecoJets_cff')
#process.load('RecoHI.HiJetAlgos.HiRecoPFJets_cff')

process.hiGenParticles.srcVector = cms.vstring('generator')

process.HeavyIonGlobalParameters = cms.PSet(
    centralityVariable = cms.string("HFtowers"),
    nonDefaultGlauberModel = cms.string("Hydjet_Drum"),
    centralitySrc = cms.InputTag("hiCentrality")
    )


process.hiCentrality.producePixelhits = False
process.hiCentrality.producePixelTracks = False
process.hiCentrality.srcTracks = cms.InputTag("hiGeneralTracks")
process.hiCentrality.srcVertex = cms.InputTag("hiSelectedVertex")
process.hiEvtPlane.vtxCollection_ = cms.InputTag("hiSelectedVertex")
process.hiEvtPlane.trackCollection_ = cms.InputTag("hiGeneralTracks")

process.load('HeavyIonsAnalysis.JetAnalysis.jets.HiGenJetsCleaned_JEC_cff')

process.load('HeavyIonsAnalysis.JetAnalysis.jets.akPu3CaloJetSequence_pp_mc_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akVs3CaloJetSequence_pp_mc_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akVs3PFJetSequence_pp_mc_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akPu3PFJetSequence_pp_mc_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.ak3PFJetSequence_pp_mc_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.ak3CaloJetSequence_pp_mc_cff')

process.load('HeavyIonsAnalysis.JetAnalysis.jets.akPu4CaloJetSequence_pp_mc_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akVs4CaloJetSequence_pp_mc_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akVs4PFJetSequence_pp_mc_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akPu4PFJetSequence_pp_mc_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.ak4PFJetSequence_pp_mc_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.ak4CaloJetSequence_pp_mc_cff')

process.load('HeavyIonsAnalysis.JetAnalysis.jets.akPu5CaloJetSequence_pp_mc_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akVs5CaloJetSequence_pp_mc_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akVs5PFJetSequence_pp_mc_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akPu5PFJetSequence_pp_mc_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.ak5PFJetSequence_pp_mc_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.ak5CaloJetSequence_pp_mc_cff')

# process.load('HeavyIonsAnalysis.JetAnalysis.jets.HiReRecoJets_PbPb_cff')
# process.voronoiBackgroundPF.src = cms.InputTag("particleFlow")
# process.PFTowers.src = cms.InputTag("particleFlow")

process.ak3PFJetAnalyzer.pfCandidateLabel=cms.untracked.InputTag('particleFlowTmp')
process.akVs3PFJetAnalyzer.pfCandidateLabel=cms.untracked.InputTag('particleFlowTmp')
process.akPu3PFJetAnalyzer.pfCandidateLabel=cms.untracked.InputTag('particleFlowTmp')

process.ak3CaloJetAnalyzer.pfCandidateLabel=cms.untracked.InputTag('particleFlowTmp')
process.akVs3CaloJetAnalyzer.pfCandidateLabel=cms.untracked.InputTag('particleFlowTmp')
process.akPu3CaloJetAnalyzer.pfCandidateLabel=cms.untracked.InputTag('particleFlowTmp')

process.ak4PFJetAnalyzer.pfCandidateLabel=cms.untracked.InputTag('particleFlowTmp')
process.akVs4PFJetAnalyzer.pfCandidateLabel=cms.untracked.InputTag('particleFlowTmp')
process.akPu4PFJetAnalyzer.pfCandidateLabel=cms.untracked.InputTag('particleFlowTmp')

process.ak4CaloJetAnalyzer.pfCandidateLabel=cms.untracked.InputTag('particleFlowTmp')
process.akVs4CaloJetAnalyzer.pfCandidateLabel=cms.untracked.InputTag('particleFlowTmp')
process.akPu4CaloJetAnalyzer.pfCandidateLabel=cms.untracked.InputTag('particleFlowTmp')

process.ak5PFJetAnalyzer.pfCandidateLabel=cms.untracked.InputTag('particleFlowTmp')
process.akVs5PFJetAnalyzer.pfCandidateLabel=cms.untracked.InputTag('particleFlowTmp')
process.akPu5PFJetAnalyzer.pfCandidateLabel=cms.untracked.InputTag('particleFlowTmp')

process.ak5CaloJetAnalyzer.pfCandidateLabel=cms.untracked.InputTag('particleFlowTmp')
process.akVs5CaloJetAnalyzer.pfCandidateLabel=cms.untracked.InputTag('particleFlowTmp')
process.akPu5CaloJetAnalyzer.pfCandidateLabel=cms.untracked.InputTag('particleFlowTmp')

process.ak3PFJetAnalyzer.trackTag=cms.InputTag('hiGeneralTracks')
process.akVs3PFJetAnalyzer.trackTag=cms.InputTag('hiGeneralTracks')
process.akPu3PFJetAnalyzer.trackTag=cms.InputTag('hiGeneralTracks')

process.ak3CaloJetAnalyzer.trackTag=cms.InputTag('hiGeneralTracks')
process.akVs3CaloJetAnalyzer.trackTag=cms.InputTag('hiGeneralTracks')
process.akPu3CaloJetAnalyzer.trackTag=cms.InputTag('hiGeneralTracks')

process.ak4PFJetAnalyzer.trackTag=cms.InputTag('hiGeneralTracks')
process.akVs4PFJetAnalyzer.trackTag=cms.InputTag('hiGeneralTracks')
process.akPu4PFJetAnalyzer.trackTag=cms.InputTag('hiGeneralTracks')

process.ak4CaloJetAnalyzer.trackTag=cms.InputTag('hiGeneralTracks')
process.akVs4CaloJetAnalyzer.trackTag=cms.InputTag('hiGeneralTracks')
process.akPu4CaloJetAnalyzer.trackTag=cms.InputTag('hiGeneralTracks')

process.ak5PFJetAnalyzer.trackTag=cms.InputTag('hiGeneralTracks')
process.akVs5PFJetAnalyzer.trackTag=cms.InputTag('hiGeneralTracks')
process.akPu5PFJetAnalyzer.trackTag=cms.InputTag('hiGeneralTracks')

process.ak5CaloJetAnalyzer.trackTag=cms.InputTag('hiGeneralTracks')
process.akVs5CaloJetAnalyzer.trackTag=cms.InputTag('hiGeneralTracks')
process.akPu5CaloJetAnalyzer.trackTag=cms.InputTag('hiGeneralTracks')


process.jetSequences = cms.Sequence(
                                    process.voronoiBackgroundCalo +
                                    process.voronoiBackgroundPF +
                                    process.PFTowers +
                                    # process.hiReRecoCaloJets +
                                    # process.hiReRecoPFJets +

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
                                    process.ak5CaloJetSequence

                                    )

process.load('HeavyIonsAnalysis.EventAnalysis.hievtanalyzer_mc_cfi')
process.load('HeavyIonsAnalysis.JetAnalysis.HiGenAnalyzer_cfi')

process.hiEvtAnalyzer.Vertex = cms.InputTag("hiSelectedVertex")

#####################################################################################
# To be cleaned

process.load('HeavyIonsAnalysis.JetAnalysis.ExtraTrackReco_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.TrkAnalyzers_MC_cff')
process.load("HeavyIonsAnalysis.TrackAnalysis.METAnalyzer_cff")
process.load("HeavyIonsAnalysis.JetAnalysis.pfcandAnalyzer_pp_cfi")
process.pfcandAnalyzer.pfCandidateLabel = cms.InputTag("particleFlowTmp")

process.load('HeavyIonsAnalysis.JetAnalysis.rechitanalyzer_pp_cfi')
process.rechitanalyzer.vtxSrc = cms.untracked.InputTag("hiSelectedVertex")
process.pfTowers.vtxSrc = cms.untracked.InputTag("hiSelectedVertex")
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

#####################
# photons
process.load('HeavyIonsAnalysis.JetAnalysis.EGammaAnalyzers_cff')
process.multiPhotonAnalyzer.GenEventScale = cms.InputTag("generator")
process.multiPhotonAnalyzer.HepMCProducer = cms.InputTag("generator")
process.hiGoodTracks.src = cms.InputTag("hiGeneralTracks")
process.hiGoodTracks.vertices = cms.InputTag("hiSelectedVertex")
process.photonMatch.matched = cms.InputTag("genParticles")
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

process.ppTrack.doSimTrack = cms.untracked.bool(True)

process.temp_step = cms.Path(process.hiGenParticles *
                             process.hiGenParticlesForJets
                             *
                             process.ak1HiGenJets +
                             process.ak2HiGenJets +
                             process.ak3HiGenJets +
                             process.ak4HiGenJets +
                             process.ak5HiGenJets +
                             process.ak6HiGenJets +
                             process.ak7HiGenJets)

process.ana_step = cms.Path(
                            process.hiCentrality +
                            process.centralityBin +
                            process.hiEvtPlane +
                            process.heavyIon*
                            process.hiEvtAnalyzer*
                            process.HiGenParticleAna*
                            process.hiGenJetsCleaned*
                            process.rechitAna +
                            process.jetSequences +
                            process.photonStep +
                            process.pfcandAnalyzer +
#temp                            process.hltMuTree +
                            process.HiForest +
                            process.cutsTPForFak +
                            process.cutsTPForEff +
                            process.anaTrack)

process.load('HeavyIonsAnalysis.JetAnalysis.EventSelection_cff')
process.phltJetHI = cms.Path( process.hltJetHI )
process.PAprimaryVertexFilter.src = cms.InputTag("hiSelectedVertex")
process.NoScraping.src = cms.untracked.InputTag("hiGeneralTracks")
process.pPAcollisionEventSelectionPA = cms.Path(process.PAcollisionEventSelection)
process.pHBHENoiseFilter = cms.Path( process.HBHENoiseFilter )
process.phfCoincFilter = cms.Path(process.hfCoincFilter )
process.phfCoincFilter3 = cms.Path(process.hfCoincFilter3 )
#process.pprimaryVertexFilter = cms.Path(process.primaryVertexFilter )
process.phltPixelClusterShapeFilter = cms.Path(process.siPixelRecHits*process.hltPixelClusterShapeFilter )
process.phiEcalRecHitSpikeFilter = cms.Path(process.hiEcalRecHitSpikeFilter )

# Customization
process.load('HeavyIonsAnalysis.EventAnalysis.hltanalysis_cff')

process.hltAna = cms.Path(process.hltanalysis)
process.pAna = cms.EndPath(process.skimanalysis)
