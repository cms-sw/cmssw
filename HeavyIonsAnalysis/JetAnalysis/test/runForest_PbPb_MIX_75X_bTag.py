#!/usr/bin/env python2
# Run the foresting configuration on PbPb in CMSSW_5_3_X, using the new HF/Voronoi jets
# Author: Alex Barbieri
# Date: 2013-10-15


import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing


## parse some command line arguments

options = VarParsing.VarParsing ('analysis')

options.inputFiles = '/store/user/rkunnawa/Run2_PbPbpthat120_RECO_HCAL_Method0_40kevents/HydjetNcoll_Pyquen_DiJet_pt120to9999_5020GeV_cfi_GEN_SIM_750_run2_mc_HIon_BS_v2/Run2_PbPbpthat120_RECO_HCAL_Method0_40kevents/151021_183818/0000/step3_RAW2DIGI_L1Reco_RECO_PU_116.root'#'/store/user/mnguyen/PyquenUnquenched_BJetLO_pt30_5TeV_GEN-SIM/PyquenUnquenched_BJetLO_pthat30_PbPb_5TeV_embedded_740_MCHI2_74_V4_RECO/87777ff9102bfbec971698893bf3d6db/step3_RAW2DIGI_L1Reco_RECO_3_1_7H9.root'



options.outputFile = 'HiForest.root'
options.maxEvents = -1

options.parseArguments()

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
                            #secondaryFileNames = cms.untracked.vstring(options.secondaryInputFiles),
                            fileNames = cms.untracked.vstring(
        options.inputFiles
        ),
                            #eventsToProcess = cms.untracked.VEventRange("1:88-1:88")
                            )

# Number of events we want to process, -1 = all events
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(options.maxEvents))


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
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc_HIon', '')
process.HiForest.GlobalTagLabel = process.GlobalTag.globaltag

process.GlobalTag.snapshotTime = cms.string("9999-12-31 23:59:59.000")
process.GlobalTag.toGet.extend([
    cms.PSet(record = cms.string("HeavyIonRcd"),
             tag = cms.string("CentralityTable_HFtowers200_HydjetDrum5_v755x01_mc"),
             connect = cms.string("frontier://FrontierProd/CMS_CONDITIONS"),
             label = cms.untracked.string("HFtowers")
    ),
])


from HeavyIonsAnalysis.Configuration.CommonFunctions_cff import overrideJEC_PbPb5020
process = overrideJEC_PbPb5020(process)

process.load("RecoHI.HiCentralityAlgos.CentralityBin_cfi")

process.centralityBin.Centrality = cms.InputTag("hiCentrality")
process.centralityBin.centralityVariable = cms.string("HFtowers")
#process.centralityBin.nonDefaultGlauberModel = cms.string("HydjetDrum5")

#process.HeavyIonGlobalParameters = cms.PSet(
#    centralityVariable = cms.string("HFtowers"),
#    nonDefaultGlauberModel = cms.string("Hydjet_Drum"),
#    centralitySrc = cms.InputTag("hiCentrality")
#    )

#####################################################################################
# Define tree output
#####################################################################################

process.TFileService = cms.Service("TFileService",
                                   fileName=cms.string(options.outputFile))

#####################################################################################
# Additional Reconstruction and Analysis: Main Body
#####################################################################################
'''
process.allOutput = cms.OutputModule("PoolOutputModule",
                                     splitLevel = cms.untracked.int32(0),
                                     eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
                                     outputCommands = cms.untracked.vstring(['keep *']),
                                     fileName = cms.untracked.string("testOutput.root"),
                                     dataset = cms.untracked.PSet(
        filterName = cms.untracked.string(''),
        dataTier = cms.untracked.string('GEN-SIM-RECO')
        )
                                     )
'''

process.load('HeavyIonsAnalysis.JetAnalysis.jets.akPu3CaloJetSequence_PbPb_mix_bTag_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akPu3PFJetSequence_PbPb_mix_bTag_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akPu4CaloJetSequence_PbPb_mix_bTag_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akPu4PFJetSequence_PbPb_mix_bTag_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akVs3CaloJetSequence_PbPb_mix_bTag_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akVs3PFJetSequence_PbPb_mix_bTag_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akVs4CaloJetSequence_PbPb_mix_bTag_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akVs4PFJetSequence_PbPb_mix_bTag_cff')

process.load("RecoHI.HiJetAlgos.ParticleTowerProducer_cfi")
process.load('HeavyIonsAnalysis.JetAnalysis.bTaggingTracks_cff')

process.jetSequences = cms.Sequence(process.PureTracks + process.offlinePrimaryVertices +
                                    process.offlinePrimaryVertices +
                                    process.akPu3CaloJetSequence +
                                    process.akPu3PFJetSequence +
                                    process.akPu4CaloJetSequence +
                                    process.akPu4PFJetSequence +
                                    process.akVs3CaloJetSequence +
                                    process.akVs3PFJetSequence +
                                    process.akVs4CaloJetSequence +
                                    process.akVs4PFJetSequence
                                    )

process.load('HeavyIonsAnalysis.EventAnalysis.hievtanalyzer_mc_cfi')
process.hiEvtAnalyzer.doMC = cms.bool(False) #the gen info dataformat has changed in 73X, we need to update hiEvtAnalyzer code
process.load('HeavyIonsAnalysis.EventAnalysis.hltanalysis_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.HiGenAnalyzer_cfi')

hltProcName = "HLT"
process.hltanalysis.HLTProcessName = cms.string(hltProcName)
process.hltanalysis.hltresults = cms.InputTag("TriggerResults","",hltProcName)
process.hltanalysis.l1GtObjectMapRecord = cms.InputTag("hltL1GtObjectMap","",hltProcName)

#####################################################################################
# To be cleaned

process.load('HeavyIonsAnalysis.JetAnalysis.ExtraTrackReco_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.TrkAnalyzers_MC_cff')
process.load("HeavyIonsAnalysis.TrackAnalysis.METAnalyzer_cff")
process.load("HeavyIonsAnalysis.JetAnalysis.pfcandAnalyzer_cfi")
process.load('HeavyIonsAnalysis.JetAnalysis.rechitanalyzer_cfi')
process.pfcandAnalyzer.skipCharged = False
process.pfcandAnalyzer.pfPtMin = 0

#####################################################################################

#########################
# Track Analyzer
#########################
process.anaTrack.qualityStrings = cms.untracked.vstring('highPurity')
process.pixelTrack.qualityStrings = cms.untracked.vstring('highPurity')
process.hiTracks.cut = cms.string('quality("highPurity")')

# set track collection to iterative tracking
process.anaTrack.trackSrc = cms.InputTag("hiGeneralTracks")

# clusters missing in recodebug - to be resolved
process.anaTrack.doPFMatching = False
process.pixelTrack.doPFMatching = False

process.anaTrack.doSimVertex = False
process.anaTrack.doSimTrack = False
process.anaTrack.fillSimTrack = False

process.load("SimTracker.TrackAssociation.trackingParticleRecoTrackAsssociation_cff")
process.tpRecoAssocGeneralTracks = process.trackingParticleRecoTrackAsssociation.clone()
process.tpRecoAssocGeneralTracks.label_tr = cms.InputTag("hiGeneralTracks")
process.quickTrackAssociatorByHits.ComponentName = cms.string('quickTrackAssociatorByHits')



#####################
# muons
######################
process.load("HeavyIonsAnalysis.MuonAnalysis.hltMuTree_cfi")
process.hltMuTree.doGen = cms.untracked.bool(True)

process.load("PhysicsTools.JetMCAlgos.SelectPartons_cff")

process.load("RecoJets.Configuration.GenJetParticles_cff")
process.genParticlesForJets.ignoreParticleIDs += cms.vuint32( 12,14,16)
process.load("RecoHI.HiJetAlgos.HiGenJets_cff")


process.hiExtraGenSequence = cms.Sequence(
    process.myPartons*
    process.genParticlesForJets*
    process.ak3HiGenJets*
    process.ak3HiGenJets
    )

process.load("RecoHI.HiJetAlgos.HiRecoPFJets_cff")
process.PFTowers.useHF = True

process.reRecoJets = cms.Sequence(
#    process.PFTowers
#    *process.akPu3PFJets
    )

process.load("GeneratorInterface.HiGenCommon.HeavyIon_cff")

process.ana_step = cms.Path(#process.heavyIon*
                            process.centralityBin*
                            process.hltanalysis *
#temp                            process.hltobject *
                            process.hiEvtAnalyzer* #evt analyzer is fundamentally broken with the new consumes model
                            process.hiExtraGenSequence*
                            process.reRecoJets*
                            #process.HiGenParticleAna*
                            #process.quickTrackAssociatorByHits *
                            #process.tpRecoAssocGeneralTracks +
                            process.jetSequences +
                            #process.photonStep_withReco +
                            #process.pfcandAnalyzer +
#temp                            process.hltMuTree +
                            process.anaTrack +
                            process.HiForest
                            )

process.load('HeavyIonsAnalysis.JetAnalysis.EventSelection_cff')
#process.phltJetHI = cms.Path( process.hltJetHI )
process.pcollisionEventSelection = cms.Path(process.collisionEventSelectionAOD)
process.load('CommonTools.RecoAlgos.HBHENoiseFilterResultProducer_cfi')
process.pHBHENoiseFilterResultProducer = cms.Path( process.HBHENoiseFilterResultProducer )
#process.pHBHENoiseFilter = cms.Path( process.HBHENoiseFilter )
#process.phfCoincFilter = cms.Path(process.hfCoincFilter )
#process.phfCoincFilter3 = cms.Path(process.hfCoincFilter3 )
process.pprimaryVertexFilter = cms.Path(process.primaryVertexFilter )
#process.phltPixelClusterShapeFilter = cms.Path(process.siPixelRecHits*process.hltPixelClusterShapeFilter )
#process.phiEcalRecHitSpikeFilter = cms.Path(process.hiEcalRecHitSpikeFilter )

process.pAna = cms.EndPath(process.skimanalysis)
#process.output_step = cms.EndPath(process.allOutput)

# Customization
process.HiGenParticleAna.ptMin = 2.
process.HiGenParticleAna.genParticleSrc = cms.untracked.InputTag("genParticles")
process.HiGenParticleAna.stableOnly = cms.untracked.bool(False)

oldGenParticleTag=cms.InputTag("hiGenParticles")
newGenParticleTag=cms.InputTag("genParticles")
oldProcLabelTag=cms.InputTag("hiSignal")
newProcLabelTag=cms.InputTag("generator")

from PhysicsTools.PatAlgos.tools.helpers import massSearchReplaceAnyInputTag
for s in process.paths_().keys():
    massSearchReplaceAnyInputTag(getattr(process,s),oldGenParticleTag,newGenParticleTag)
    #massSearchReplaceAnyInputTag(getattr(process,s),newGenParticleTag,oldGenParticleTag)  // go back to hiGenParticles
    massSearchReplaceAnyInputTag(getattr(process,s),oldProcLabelTag,newProcLabelTag)
