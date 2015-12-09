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
    process.HiForest.HiForestVersion = cms.string(version)

#####################################################################################
# Input source
#####################################################################################

process.source = cms.Source("PoolSource",
                            duplicateCheckMode = cms.untracked.string("noDuplicateCheck"),
                            fileNames = cms.untracked.vstring(
                                "file:/afs/cern.ch/work/m/mnguyen/public/testPPfile/step3.root"  # replace me 
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
#process.load('RecoHI.HiCentralityAlgos.CentralityBin_cfi')

process.load('FWCore.MessageService.MessageLogger_cfi')

# pp 75X MC
from Configuration.AlCa.GlobalTag_condDBv2 import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')
process.HiForest.GlobalTagLabel = process.GlobalTag.globaltag


# Customization
from HeavyIonsAnalysis.Configuration.CommonFunctions_cff import overrideJEC_pp5020
process = overrideJEC_pp5020(process)


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

process.load('RecoHI.HiJetAlgos.HiGenJets_cff')

process.load('HeavyIonsAnalysis.JetAnalysis.jets.ak4PFJetSequence_pp_mc_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.ak4CaloJetSequence_pp_mc_cff')


process.highPurityTracks = cms.EDFilter("TrackSelector",
                                        src = cms.InputTag("generalTracks"),
                                        cut = cms.string('quality("highPurity")')
                                        )

# Other radii jets and calo jets need to be reconstructed
process.jetSequences = cms.Sequence(
    process.highPurityTracks +
    #process.ak1PFJetSequence +
    #process.ak1CaloJetSequence +
    #process.ak2PFJetSequence +
    #process.ak2CaloJetSequence +
    #process.ak3PFJetSequence +
    #process.ak3CaloJetSequence +
    process.ak4PFJetSequence +
    process.ak4CaloJetSequence #+
    #process.ak5PFJetSequence +
    #process.ak5CaloJetSequence +
    #process.ak6PFJetSequence +
    #process.ak6CaloJetSequence
)


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
process.ppTrack.doSimVertex = False
process.ppTrack.doSimTrack = False
process.ppTrack.pfCandSrc = cms.InputTag("particleFlow")
process.ppTrack.doPFMatching = cms.untracked.bool(True)


#####################
# photons
######################
process.load('RecoJets.Configuration.GenJetParticles_cff')
process.genParticlesForJets.ignoreParticleIDs += cms.vuint32( 12,14,16)

process.load('HeavyIonsAnalysis.PhotonAnalysis.ggHiNtuplizer_cfi')
process.ggHiNtuplizer.gsfElectronLabel   = cms.InputTag("gedGsfElectrons")
process.ggHiNtuplizer.useValMapIso       = cms.bool(False)
process.ggHiNtuplizer.VtxLabel           = cms.InputTag("offlinePrimaryVerticesWithBS")
process.ggHiNtuplizer.particleFlowCollection = cms.InputTag("particleFlow")
process.ggHiNtuplizer.doVsIso            = cms.bool(False)
process.ggHiNtuplizer.doGenParticles = False
process.ggHiNtuplizerGED = process.ggHiNtuplizer.clone(recoPhotonSrc = cms.InputTag('gedPhotons'))

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

process.ana_step = cms.Path(process.hltanalysis *
                            process.HiGenParticleAna*
                            #process.PFTowers +
                            #process.hiReRecoPFJets+
                            #process.hiReRecoCaloJets+
                            process.akHiGenJets  +
                            process.jetSequences +
                            process.ggHiNtuplizer +
                            process.ggHiNtuplizerGED +
                            # process.pfcandAnalyzer +
                            process.HiForest +
                            process.ppTrack +
                            process.runAnalyzer
)


process.load("HeavyIonsAnalysis.VertexAnalysis.PAPileUpVertexFilter_cff")

process.pVertexFilterCutG = cms.Path(process.pileupVertexFilterCutG)
process.pVertexFilterCutGloose = cms.Path(process.pileupVertexFilterCutGloose)
process.pVertexFilterCutGtight = cms.Path(process.pileupVertexFilterCutGtight)
process.pVertexFilterCutGplus = cms.Path(process.pileupVertexFilterCutGplus)
process.pVertexFilterCutE = cms.Path(process.pileupVertexFilterCutE)
process.pVertexFilterCutEandG = cms.Path(process.pileupVertexFilterCutEandG)


# Customization

process.pAna = cms.EndPath(process.skimanalysis)
process.skimanalysis.useHBHENoiseProducer = False   # not present in AOD
