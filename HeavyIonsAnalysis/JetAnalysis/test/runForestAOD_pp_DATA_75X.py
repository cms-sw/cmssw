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
                            # fileNames = cms.untracked.vstring("file:/mnt/hadoop/cms/store/user/dgulhan/HIHighPt/HIHighPt_photon20and30_HIRun2011-v1_RECO_753_patch1/fd44351629dd155a25de2b4c109c824c/RECO_100_1_Uk0.root")                        )
                            fileNames = cms.untracked.vstring("file:/afs/cern.ch/user/d/dgulhan/workDir/public/forAlex/160BBE1D-D36B-E511-A651-02163E01441A.root")                        )


# Number of events we want to process, -1 = all events
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10))



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

# PbPb 53X MC

from Configuration.AlCa.GlobalTag_condDBv2 import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run1_data', '')
# set snapshot to future to allow centrality table payload.
process.GlobalTag.snapshotTime = cms.string("9999-12-31 23:59:59.000")

process.GlobalTag.toGet.extend([
 cms.PSet(record = cms.string("HeavyIonRcd"),
 connect = cms.string("frontier://FrontierProd/CMS_CONDITIONS"),
 ## 5.02 TeV Centrality Tables
 #tag = cms.string("CentralityTable_HFtowers200_HydjetDrum5_v740x01_mc"),
 #label = cms.untracked.string("HFtowersHydjetDrum5")
 ## 2.76 TeV Centrality Tables for data
 tag = cms.string("CentralityTable_HFtowers200_Glauber2010A_eff99_run1v750x01_offline"),
 label = cms.untracked.string("HFtowers")
 ),
])

from HeavyIonsAnalysis.Configuration.CommonFunctionsLocalDB_cff import overrideJEC_HI_PythiaCUETP8M1_5020GeV_753p1_v3_db
process = overrideJEC_HI_PythiaCUETP8M1_5020GeV_753p1_v3_db(process)

process.load("RecoHI.HiCentralityAlgos.CentralityBin_cfi")
process.centralityBin.Centrality = cms.InputTag("hiCentrality")
process.centralityBin.centralityVariable = cms.string("HFtowers")
# 5.02 table
#process.centralityBin.nonDefaultGlauberModel = cms.string("HydjetDrum5")
# 2.76 table
process.centralityBin.nonDefaultGlauberModel = cms.string("")

#####################################################################################
# Define tree output
#####################################################################################

process.TFileService = cms.Service("TFileService",
                                   fileName=cms.string("HiForestAOD.root"))

#####################################################################################
# Additional Reconstruction and Analysis: Main Body
#####################################################################################


process.load('HeavyIonsAnalysis.JetAnalysis.jets.ak3CaloJetSequence_pp_data_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.ak4CaloJetSequence_pp_data_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.ak5CaloJetSequence_pp_data_cff')

process.load('HeavyIonsAnalysis.JetAnalysis.jets.ak3PFJetSequence_pp_data_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.ak4PFJetSequence_pp_data_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.ak5PFJetSequence_pp_data_cff')

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
process.load('HeavyIonsAnalysis.EventAnalysis.hltanalysis_cff')

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
# Track Analyzer
#########################
# process.ppTrack.qualityStrings = cms.untracked.vstring(['highPurity'])
# process.ppTrack.trackSrc = cms.InputTag("generalTracks")
# process.ppTrack.doPFMatching = False


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
                            process.jetSequences +
                            process.ggHiNtuplizer +
                            process.ggHiNtuplizerGED +
                            process.pfcandAnalyzer +
                            process.HiForest +
                            process.ppTrack
                            )

process.load('HeavyIonsAnalysis.JetAnalysis.EventSelection_cff')
process.phltJetHI = cms.Path( process.hltJetHI )
process.pcollisionEventSelection = cms.Path(process.collisionEventSelectionAOD)
process.load('CommonTools.RecoAlgos.HBHENoiseFilterResultProducer_cfi')
process.pHBHENoiseFilterResultProducer = cms.Path( process.HBHENoiseFilterResultProducer )

process.load("HeavyIonsAnalysis.VertexAnalysis.PAPileUpVertexFilter_cff")

process.pVertexFilterCutG = cms.Path(process.pileupVertexFilterCutG)
process.pVertexFilterCutGloose = cms.Path(process.pileupVertexFilterCutGloose)
process.pVertexFilterCutGtight = cms.Path(process.pileupVertexFilterCutGtight)
process.pVertexFilterCutGplus = cms.Path(process.pileupVertexFilterCutGplus)
process.pVertexFilterCutE = cms.Path(process.pileupVertexFilterCutE)
process.pVertexFilterCutEandG = cms.Path(process.pileupVertexFilterCutEandG)

process.pAna = cms.EndPath(process.skimanalysis)

# Customization
