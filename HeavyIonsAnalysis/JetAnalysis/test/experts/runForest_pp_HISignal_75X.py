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
                                "file:step3_1.root"
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
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff')
process.load('Configuration.StandardSequences.Digi_cff')
process.load('Configuration.StandardSequences.SimL1Emulator_cff')
process.load('Configuration.StandardSequences.DigiToRaw_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Configuration.StandardSequences.ReconstructionHeavyIons_cff')
process.load('RecoHI.HiCentralityAlgos.CentralityBin_cfi')

process.load('FWCore.MessageService.MessageLogger_cfi')

# pp 75X MC
from Configuration.AlCa.GlobalTag_condDBv2 import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')

#from HeavyIonsAnalysis.Configuration.CommonFunctionsLocalDB_cff import overrideJEC_HI_PythiaCUETP8M1_5020GeV_753p1_v3_db
#process = overrideJEC_HI_PythiaCUETP8M1_5020GeV_753p1_v3_db(process)

# Customization
from HeavyIonsAnalysis.Configuration.CommonFunctions_cff import overrideJEC_PbPb5020
process = overrideJEC_PbPb5020(process)


process.HeavyIonGlobalParameters = cms.PSet(
    centralityVariable = cms.string("HFtowersTrunc"),
    nonDefaultGlauberModel = cms.string("Hijing"),
    centralitySrc = cms.InputTag("hiCentrality")
)

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

process.hiCentrality.producePixelhits = False
process.hiCentrality.producePixelTracks = False
process.hiCentrality.srcTracks = cms.InputTag("hiGeneralTracks")
process.hiCentrality.srcVertex = cms.InputTag("offlinePrimaryVerticesWithBS")

process.load('RecoHI.HiJetAlgos.HiGenJets_cff')

process.load('HeavyIonsAnalysis.JetAnalysis.jets.ak1PFJetSequence_PbPb_jec_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.ak1CaloJetSequence_PbPb_jec_cff')

process.load('HeavyIonsAnalysis.JetAnalysis.jets.ak2PFJetSequence_PbPb_jec_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.ak2CaloJetSequence_PbPb_jec_cff')

process.load('HeavyIonsAnalysis.JetAnalysis.jets.ak3PFJetSequence_PbPb_jec_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.ak3CaloJetSequence_PbPb_jec_cff')

process.load('HeavyIonsAnalysis.JetAnalysis.jets.ak4PFJetSequence_PbPb_jec_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.ak4CaloJetSequence_PbPb_jec_cff')

process.load('HeavyIonsAnalysis.JetAnalysis.jets.ak5PFJetSequence_PbPb_jec_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.ak5CaloJetSequence_PbPb_jec_cff')

process.load('HeavyIonsAnalysis.JetAnalysis.jets.ak6PFJetSequence_PbPb_jec_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.ak6CaloJetSequence_PbPb_jec_cff')


process.load('HeavyIonsAnalysis.JetAnalysis.jets.akPu1PFJetSequence_PbPb_jec_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akPu1CaloJetSequence_PbPb_jec_cff')

process.load('HeavyIonsAnalysis.JetAnalysis.jets.akPu2PFJetSequence_PbPb_jec_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akPu2CaloJetSequence_PbPb_jec_cff')

process.load('HeavyIonsAnalysis.JetAnalysis.jets.akPu3PFJetSequence_PbPb_jec_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akPu3CaloJetSequence_PbPb_jec_cff')

process.load('HeavyIonsAnalysis.JetAnalysis.jets.akPu4PFJetSequence_PbPb_jec_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akPu4CaloJetSequence_PbPb_jec_cff')

process.load('HeavyIonsAnalysis.JetAnalysis.jets.akPu5PFJetSequence_PbPb_jec_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akPu5CaloJetSequence_PbPb_jec_cff')

process.load('HeavyIonsAnalysis.JetAnalysis.jets.akPu6PFJetSequence_PbPb_jec_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akPu6CaloJetSequence_PbPb_jec_cff')

process.akPu1PFJetAnalyzer.doSubEvent = True
process.akPu1CaloJetAnalyzer.doSubEvent = True

process.akPu2PFJetAnalyzer.doSubEvent = True
process.akPu2CaloJetAnalyzer.doSubEvent = True

process.akPu3PFJetAnalyzer.doSubEvent = True
process.akPu3CaloJetAnalyzer.doSubEvent = True

process.akPu4PFJetAnalyzer.doSubEvent = True
process.akPu4CaloJetAnalyzer.doSubEvent = True

process.akPu5PFJetAnalyzer.doSubEvent = True
process.akPu5CaloJetAnalyzer.doSubEvent = True

process.akPu6PFJetAnalyzer.doSubEvent = True
process.akPu6CaloJetAnalyzer.doSubEvent = True

process.akPu1PFJetAnalyzer.jetPtMin = cms.untracked.double(1)
process.akPu1CaloJetAnalyzer.jetPtMin = cms.untracked.double(1)

process.akPu2PFJetAnalyzer.jetPtMin = cms.untracked.double(1)
process.akPu2CaloJetAnalyzer.jetPtMin = cms.untracked.double(1)

process.akPu3PFJetAnalyzer.jetPtMin = cms.untracked.double(1)
process.akPu3CaloJetAnalyzer.jetPtMin = cms.untracked.double(1)

process.akPu4PFJetAnalyzer.jetPtMin = cms.untracked.double(1)
process.akPu4CaloJetAnalyzer.jetPtMin = cms.untracked.double(1)

process.akPu5PFJetAnalyzer.jetPtMin = cms.untracked.double(1)
process.akPu5CaloJetAnalyzer.jetPtMin = cms.untracked.double(1)

process.akPu6PFJetAnalyzer.jetPtMin = cms.untracked.double(1)
process.akPu6CaloJetAnalyzer.jetPtMin = cms.untracked.double(1)

process.ak1PFJetAnalyzer.doSubEvent = True
process.ak1CaloJetAnalyzer.doSubEvent = True

process.ak2PFJetAnalyzer.doSubEvent = True
process.ak2CaloJetAnalyzer.doSubEvent = True

process.ak3PFJetAnalyzer.doSubEvent = True
process.ak3CaloJetAnalyzer.doSubEvent = True

process.ak4PFJetAnalyzer.doSubEvent = True
process.ak4CaloJetAnalyzer.doSubEvent = True

process.ak5PFJetAnalyzer.doSubEvent = True
process.ak5CaloJetAnalyzer.doSubEvent = True

process.ak6PFJetAnalyzer.doSubEvent = True
process.ak6CaloJetAnalyzer.doSubEvent = True

process.ak1PFJetAnalyzer.jetPtMin = cms.untracked.double(1)
process.ak1CaloJetAnalyzer.jetPtMin = cms.untracked.double(1)

process.ak2PFJetAnalyzer.jetPtMin = cms.untracked.double(1)
process.ak2CaloJetAnalyzer.jetPtMin = cms.untracked.double(1)

process.ak3PFJetAnalyzer.jetPtMin = cms.untracked.double(1)
process.ak3CaloJetAnalyzer.jetPtMin = cms.untracked.double(1)

process.ak4PFJetAnalyzer.jetPtMin = cms.untracked.double(1)
process.ak4CaloJetAnalyzer.jetPtMin = cms.untracked.double(1)

process.ak5PFJetAnalyzer.jetPtMin = cms.untracked.double(1)
process.ak5CaloJetAnalyzer.jetPtMin = cms.untracked.double(1)

process.ak6PFJetAnalyzer.jetPtMin = cms.untracked.double(1)
process.ak6CaloJetAnalyzer.jetPtMin = cms.untracked.double(1)

process.load('HeavyIonsAnalysis.JetAnalysis.jets.HiReRecoJets_HI_cff')


process.voronoiBackgroundPF.src = cms.InputTag("particleFlow")
process.PFTowers.src = cms.InputTag("particleFlow")


process.jetSequences = cms.Sequence(
    process.ak1PFJetSequence +
    process.ak1CaloJetSequence +
    process.ak2PFJetSequence +
    process.ak2CaloJetSequence +
    process.ak3PFJetSequence +
    process.ak3CaloJetSequence +
    process.ak4PFJetSequence +
    process.ak4CaloJetSequence +
    process.ak5PFJetSequence +
    process.ak5CaloJetSequence +
    process.ak6PFJetSequence +
    process.ak6CaloJetSequence 
)

process.ak1PFJetAnalyzer.doSubEvent = True
process.ak1CaloJetAnalyzer.doSubEvent = True

process.load('HeavyIonsAnalysis.EventAnalysis.hievtanalyzer_mc_cfi')
process.load('HeavyIonsAnalysis.JetAnalysis.HiGenAnalyzer_cfi')
process.load('HeavyIonsAnalysis.EventAnalysis.runanalyzer_cff')

process.hiEvtAnalyzer.Vertex = cms.InputTag("offlinePrimaryVerticesWithBS")


#####################################################################################
# To be cleaned

process.load('HeavyIonsAnalysis.JetAnalysis.ExtraTrackReco_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.TrkAnalyzers_MC_cff')
process.load("HeavyIonsAnalysis.TrackAnalysis.METAnalyzer_cff")
process.load("HeavyIonsAnalysis.JetAnalysis.pfcandAnalyzer_pp_cfi")
process.load('HeavyIonsAnalysis.JetAnalysis.rechitanalyzer_pp_cfi')
# process.rechitAna = cms.Sequence(process.rechitanalyzer+process.pfTowers)
process.pfcandAnalyzer.skipCharged = False
process.pfcandAnalyzer.pfPtMin = 0
process.pfcandAnalyzer.pfCandidateLabel = cms.InputTag("particleFlow")
process.pfcandAnalyzer.doVS = cms.untracked.bool(False)
process.pfcandAnalyzer.doUEraw_ = cms.untracked.bool(False)
process.pfcandAnalyzer.genLabel = cms.InputTag("genParticles")

#####################################################################################

#########################
# Track Analyzer
#########################
process.hiTracks.cut = cms.string('quality("highPurity")')

#########################
# Track Analyzer
#########################
process.ppTrack.qualityStrings = cms.untracked.vstring(['highPurity','tight','loose'])

process.hiTracks.cut = cms.string('quality("highPurity")')

# set track collection to iterative tracking
process.load("SimTracker.TrackAssociation.trackingParticleRecoTrackAsssociation_cff")
process.tpRecoAssochiGeneralTracks = process.trackingParticleRecoTrackAsssociation.clone()
process.tpRecoAssochiGeneralTracks.label_tr = cms.InputTag("hiGeneralTracks")
process.quickTrackAssociatorByHits.ComponentName = cms.string('quickTrackAssociatorByHits')
process.quickTrackAssociatorByHits.SimToRecoDenominator = cms.string('reco')
# process.quickTrackAssociatorByHits.Cut_RecoToSim = cms.double(0.75)
# process.quickTrackAssociatorByHits.Quality_SimToReco = cms.double(0.5)


#####################
# photons
######################
process.load('RecoJets.Configuration.GenJetParticles_cff')
process.genParticlesForJets.ignoreParticleIDs += cms.vuint32( 12,14,16)

process.load('HeavyIonsAnalysis.PhotonAnalysis.ggHiNtuplizer_cfi')
process.load('HeavyIonsAnalysis.PhotonAnalysis.ggHiNtuplizer_cfi')
process.ggHiNtuplizer.genParticleSrc = cms.InputTag("genParticles")
process.ggHiNtuplizerGED = process.ggHiNtuplizer.clone(recoPhotonSrc = cms.InputTag('gedPhotonsTmp'),
                                                       recoPhotonHiIsolationMap = cms.InputTag('photonIsolationHIProducerGED')
                                                       )
process.akHiGenJets = cms.Sequence(
    process.genParticlesForJets +
    process.ak1HiGenJets +
    process.ak2HiGenJets +
    process.ak3HiGenJets +
    process.ak4HiGenJets +
    process.ak5HiGenJets +
    process.ak6HiGenJets)

process.load('HeavyIonsAnalysis.EventAnalysis.hltanalysis_cff')

process.HiGenParticleAna.genParticleSrc = cms.untracked.InputTag("genParticles")
process.HiGenParticleAna.doHI = False

process.PFTowers.src = cms.InputTag("particleFlowTmp")

process.ana_step = cms.Path(process.hltanalysis *
                            process.HiGenParticleAna*
                            process.PFTowers +
                            process.hiReRecoPFJets+
                            process.hiReRecoCaloJets+
			                process.akHiGenJets  +
                            process.jetSequences +
                            process.ggHiNtuplizer +
                            process.ggHiNtuplizerGED +
                            # process.pfcandAnalyzer +
   						    process.quickTrackAssociatorByHits +
			                process.tpRecoAssochiGeneralTracks +
                            process.HiForest +
                            process.anaTrack 
                            # process.runAnalyzer
)


# Customization
process.skimanalysis.useHBHENoiseProducer = cms.bool(False)

process.pAna = cms.EndPath(process.skimanalysis)
