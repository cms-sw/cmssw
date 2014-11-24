#!/usr/bin/env python2
# Run the foresting configuration on PbPb in CMSSW_5_3_X, using the new HF/Voronoi jets
# Author: Alex Barbieri
# Date: 2013-10-15

hiTrackQuality = "highPurity"              # iterative tracks
#hiTrackQuality = "highPuritySetWithPV"    # calo-matched tracks
hltProcess="HISIGNAL" #some embedding has this as HLT instead

# change this to true to switch to the Pbp JEC
secondHalfpPbJEC = True

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
                                    #"/store/user/kjung/Hijing_PPb502_MinimumBias/pPb_BFilterSampleRECO_pthat15/5dc89fb1319c58a400229c5d020a3799/RecoPythiaBJet_Apr14Prod_15_22_1_7MC.root"
                                    "file:/home/jung68/MCProjects/CMSSW_5_3_19/src/HIJINGemb_Dzero_TuneZ2star_5TeV_RECO.root"
                                    #"file:/home/jung68/CMSSW_5_3_15/src/step3_RAW"
                                    #"/store/user/kjung/Hijing_PPb502_MinimumBias/HIJING_D02KpiEmbed_502TeV_July14Prod_v1_RECO/fe29b51ccee4ce924105ae4d13f1a84a/HIJINGemb_Dzero_TuneZ2star_5TeV_RECO_35_1_PwG.root"
                                    ))

# Number of events we want to process, -1 = all events
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1))


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

# PbPb 53X MC
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'STARTHI53_V28::All', '')

from HeavyIonsAnalysis.Configuration.CommonFunctions_cff import *
overrideCentrality(process)
if secondHalfpPbJEC:
    overrideJEC_MC_Pbp5020(process)
    #overrideJEC_Pbp5020(process)
else:
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

#OpenHF Additional Loads
process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.load("CondCore.DBCommon.CondDBSetup_cfi")

#back to originally scheduled programming...
process.load('Configuration.StandardSequences.Generator_cff')
process.load('RecoJets.Configuration.GenJetParticles_cff')

process.hiGenParticles.srcVector = cms.vstring('hiSignal')

#process.hiCentrality.producePixelhits = False
#process.hiCentrality.producePixelTracks = False
#process.hiCentrality.srcTracks = cms.InputTag("generalTracks")
#process.hiCentrality.srcVertex = cms.InputTag("offlinePrimaryVerticesWithBS")
process.hiEvtPlane.vtxCollection_ = cms.InputTag("offlinePrimaryVerticesWithBS")
process.hiEvtPlane.trackCollection_ = cms.InputTag("generalTracks")

#process.load('HeavyIonsAnalysis.JetAnalysis.jets.HiGenJetsCleaned_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.bTaggers_cff')

process.load('HeavyIonsAnalysis.JetAnalysis.jets.akPu3CaloJetSequence_pPb_mc_bTag_cff')
#process.load('HeavyIonsAnalysis.JetAnalysis.jets.akVs3CaloJetSequence_pPb_mc_bTag_cff')
#process.load('HeavyIonsAnalysis.JetAnalysis.jets.akVs3PFJetSequence_pPb_mc_bTag_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akPu3PFJetSequence_pPb_mc_bTag_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.ak3PFJetSequence_pPb_mc_bTag_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.ak3CaloJetSequence_pPb_mc_bTag_cff')

process.load('HeavyIonsAnalysis.JetAnalysis.jets.akPu4CaloJetSequence_pPb_mc_bTag_cff')
#process.load('HeavyIonsAnalysis.JetAnalysis.jets.akVs4CaloJetSequence_pPb_mc_bTag_cff')
#process.load('HeavyIonsAnalysis.JetAnalysis.jets.akVs4PFJetSequence_pPb_mc_bTag_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akPu4PFJetSequence_pPb_mc_bTag_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.ak4PFJetSequence_pPb_mc_bTag_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.ak4CaloJetSequence_pPb_mc_bTag_cff')

process.load('HeavyIonsAnalysis.JetAnalysis.jets.akPu5CaloJetSequence_pPb_mc_bTag_cff')
#process.load('HeavyIonsAnalysis.JetAnalysis.jets.akVs5CaloJetSequence_pPb_mc_bTag_cff')
#process.load('HeavyIonsAnalysis.JetAnalysis.jets.akVs5PFJetSequence_pPb_mc_bTag_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akPu5PFJetSequence_pPb_mc_bTag_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.ak5PFJetSequence_pPb_mc_bTag_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.ak5CaloJetSequence_pPb_mc_bTag_cff')

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

process.load('HeavyIonsAnalysis.EventAnalysis.hievtanalyzer_mc_cfi')
process.hiEvtAnalyzer.Vertex = cms.InputTag("offlinePrimaryVerticesWithBS")

process.load('HeavyIonsAnalysis.JetAnalysis.HiGenAnalyzer_cfi')

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

process.ppTrack.pfCandSrc = cms.InputTag("particleFlow")

# Disable this for now, causes problems.
process.ppTrack.doPFMatching = cms.untracked.bool(False)
process.ppTrack.doSimVertex = True
process.ppTrack.doSimTrack = True
process.ppTrack.fillSimTrack = True

#####################
# photons
process.load('HeavyIonsAnalysis.JetAnalysis.EGammaAnalyzers_cff')
process.multiPhotonAnalyzer.GenEventScale = cms.InputTag("generator")
process.multiPhotonAnalyzer.HepMCProducer = cms.InputTag("generator")
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

## Additional Tweaks for c-jets
process.hiPartons.ptCut = cms.double(10)
process.HiGenParticleAna.stableOnly = cms.untracked.bool(False)

##more tweaks for OpenHF stuff
process.HFtree = cms.EDAnalyzer(
        "HFTree",
        verbose      = cms.untracked.int32(1),
        printFrequency = cms.untracked.int32(1000),
        requireCand  =  cms.untracked.bool(True),
        fReducedTree  =  cms.untracked.bool(True),
        isMC = cms.untracked.bool(True)
        )

process.load("UserCode.OpenHF.HFRecoStuff_cff")
process.load("UserCode.OpenHF.HFCharm_cff")

process.OpenHfTree_step = cms.Path(
        process.recoStuffSequence*
        process.charmSequence*
        process.HFtree
        )

##---------------------------------------------

process.genStep = cms.Path(process.hiGenParticles *
                           process.hiGenParticlesForJets *
                           process.genPartons *
                           process.hiPartons)

process.temp_step = cms.Path(
                             process.ak1HiGenJets +
                             process.ak2HiGenJets +
                             process.ak3HiGenJets +
                             process.ak4HiGenJets +
                             process.ak5HiGenJets +
                             process.ak6HiGenJets +
                             process.ak7HiGenJets)

process.ana_step = cms.Path(process.pACentrality +
                            process.centralityBin +
                            process.hiEvtPlane +
                            process.heavyIon*
                            process.hiEvtAnalyzer*
                            process.HiGenParticleAna*
#                            process.hiGenJetsCleaned*
                            process.jetSequences +
                            process.photonStep +
                            process.pfcandAnalyzer +
                            process.rechitAna +
#temp                            process.hltMuTree +
                            process.HiForest +
                            process.cutsTPForFak +
                            process.cutsTPForEff +
                            process.ppTrack)

process.load('HeavyIonsAnalysis.JetAnalysis.EventSelection_cff')
process.hltJetHI.TriggerResultsTag = cms.InputTag("TriggerResults","",hltProcess)
process.phltJetHI = cms.Path( process.hltJetHI )
process.pcollisionEventSelection = cms.Path(process.collisionEventSelection*process.PAcollisionEventSelection)
process.pHBHENoiseFilter = cms.Path( process.HBHENoiseFilter )
process.phfCoincFilter = cms.Path(process.hfCoincFilter )
process.phfCoincFilter3 = cms.Path(process.hfCoincFilter3 )
process.primaryVertexFilter.src = "offlinePrimaryVerticesWithBS"
process.pprimaryVertexFilter = cms.Path(process.primaryVertexFilter )
process.phltPixelClusterShapeFilter = cms.Path(process.siPixelRecHits*process.hltPixelClusterShapeFilter )
process.phiEcalRecHitSpikeFilter = cms.Path(process.hiEcalRecHitSpikeFilter )

# Customization
process.load('HeavyIonsAnalysis.EventAnalysis.hltanalysis_cff')

process.hltanalysis.hltresults = cms.InputTag("TriggerResults","",hltProcess)
process.hltanalysis.HLTProcessName = hltProcess
process.hltanalysis.dummyBranches = []
process.hltanalysis.l1GtObjectMapRecord = cms.InputTag("hltL1GtObjectMap","",hltProcess)
process.hltanalysis.mctruth = cms.InputTag("hiGenParticles","",hltProcess)

process.hltobject.processName = cms.string(hltProcess)
process.hltobject.treeName = cms.string("JetTriggers")
process.hltobject.triggerNames = cms.vstring("HLT_PAJet20_NoJetID_v1","HLT_PAJet40_NoJetID_v1","HLT_PAJet60_NoJetID_v1","HLT_PAJet80_NoJetID_v1","HLT_PAJet100_NoJetID_v1")
process.hltobject.triggerResults = process.hltanalysis.hltresults
process.hltobject.triggerEvent = cms.InputTag("hltTriggerSummaryAOD","",hltProcess)

process.photonHltObject = process.hltobject.clone()
process.photonHltObject.treeName = cms.string("PhotonTriggers")
process.photonHltObject.triggerNames = cms.vstring("HLT_PAPhoton10_NoCaloIdVL_v1","HLT_PAPhoton15_NoCaloIdVL_v1","HLT_PAPhoton20_NoCaloIdVL_v1","HLT_PAPhoton30_NoCaloIdVL_v1")

process.trackHltObject = process.hltobject.clone()
process.trackHltObject.treeName = cms.string("TrackTriggers")
process.trackHltObject.triggerNames = cms.vstring("HLT_PAFullTrack12_v2","HLT_PAFullTrack20_v2","HLT_PAFullTrack30_v2")

process.hltAna = cms.Path(process.hltanalysis*process.hltobject*process.photonHltObject*process.trackHltObject)
process.pAna = cms.EndPath(process.skimanalysis)
