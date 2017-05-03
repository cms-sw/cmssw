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
#"/store/user/dgulhan/Pyquen_Unquenched_AllQCDPhoton30_PhotonFilter20GeV_eta24_TuneZ2_PbPb_5020GeV_GEN_SIM_PU/Pyquen_Unquenched_AllQCDPhoton30_PhotonFilter20GeV_eta24_TuneZ2_PbPb_5020GeV_RECODEBUG/151021_150752/0000/step3_1.root"
"file:/afs/cern.ch/user/d/dgulhan/workDir/public/hltforest/REHLT_step3step2_10.root"
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

from Configuration.AlCa.GlobalTag_condDBv2 import GlobalTag as customiseGlobalTag

process.GlobalTag = customiseGlobalTag(process.GlobalTag, globaltag = '75X_mcRun2_HeavyIon_v7')
process.GlobalTag.connect   = 'frontier://FrontierProd/CMS_CONDITIONS'
process.GlobalTag.pfnPrefix = cms.untracked.string('frontier://FrontierProd/')
for pset in process.GlobalTag.toGet.value():
    pset.connect = pset.connect.value().replace('frontier://FrontierProd/', 'frontier://FrontierProd/')

process.GlobalTag.RefreshEachRun = cms.untracked.bool( False )
process.GlobalTag.ReconnectEachRun = cms.untracked.bool( False )

process.GlobalTag.toGet.extend([
 cms.PSet(record = cms.string("HeavyIonRcd"),
tag = cms.string("CentralityTable_HFtowers200_HydjetDrum5_v750x02_mc"),
connect = cms.string("frontier://FrontierProd/CMS_CONDITIONS"),
label = cms.untracked.string("HFtowersHydjetDrum5")
 ),
])
process.HiForest.GlobalTagLabel = process.GlobalTag.globaltag

process.load("CondCore.DBCommon.CondDBCommon_cfi")
from CondCore.DBCommon.CondDBSetup_cfi import *
process.jec = cms.ESSource("PoolDBESSource",
                               DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(0)
        ),
                               timetype = cms.string('runnumber'),
                               toGet = cms.VPSet(
        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PythiaCUETP8M1_5020GeV_753p1_v1_AK4Calo_hlt"),
                 label = cms.untracked.string("AK4Calo")
                 ),

        cms.PSet(record = cms.string("JetCorrectionsRecord"),
                 tag = cms.string("JetCorrectorParametersCollection_HI_PythiaCUETP8M1_5020GeV_753p1_v1_AK4Calo_hlt"),
                 label = cms.untracked.string("AK4Calo")
                 )
      ),
                           connect = cms.string("frontier://FrontierProd/CMS_CONDITIONS"),
)

process.es_prefer_jec = cms.ESPrefer('PoolDBESSource','jec')



process.load("RecoHI.HiCentralityAlgos.CentralityBin_cfi")
process.centralityBin.Centrality = cms.InputTag("hiCentrality")
process.centralityBin.centralityVariable = cms.string("HFtowers")
process.centralityBin.nonDefaultGlauberModel = cms.string("HydjetDrum5")

#####################################################################################
# Define tree output
#####################################################################################

process.TFileService = cms.Service("TFileService",
                                   fileName=cms.string("HiForest.root"))

#####################################################################################
# Additional Reconstruction and Analysis: Main Body
#####################################################################################


process.load('HeavyIonsAnalysis.JetAnalysis.jets.akPu3CaloJetSequence_PbPb_mc_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akVs3CaloJetSequence_PbPb_mc_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akVs3PFJetSequence_PbPb_mc_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akPu3PFJetSequence_PbPb_mc_cff')

process.load('HeavyIonsAnalysis.JetAnalysis.jets.akPu4CaloJetSequence_PbPb_mc_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akVs4CaloJetSequence_PbPb_mc_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akVs4PFJetSequence_PbPb_mc_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akPu4PFJetSequence_PbPb_mc_cff')

process.load('HeavyIonsAnalysis.JetAnalysis.jets.akPu5CaloJetSequence_PbPb_mc_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akVs5CaloJetSequence_PbPb_mc_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akVs5PFJetSequence_PbPb_mc_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akPu5PFJetSequence_PbPb_mc_cff')


process.load('HeavyIonsAnalysis.JetAnalysis.jets.hltPuAK4CaloJetsCorrectedIDPassed_PbPb_mix_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.hltPuAK4CaloJetsIDPassedCorrK_PbPb_mix_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.hltPuAK4CaloJetsIDPassedCorrL_PbPb_mix_cff')

process.hltPuAK4CaloJetsIDPassedCorrKJetAnalyzer.doSubEvent = cms.untracked.bool(True)
process.hltPuAK4CaloJetsIDPassedCorrKJetAnalyzer.doHiJetID = cms.untracked.bool(True)
process.hltPuAK4CaloJetsIDPassedCorrKJetAnalyzer.jetPtMin = cms.untracked.double(-9999)
process.hltPuAK4CaloJetsIDPassedCorrKJetAnalyzer.genPtMin = cms.untracked.double(-9999)



process.jetSequences = cms.Sequence(
#                                    process.akPu3CaloJetSequence +
#                                    process.akVs3CaloJetSequence +
#                                    process.akVs3PFJetSequence +
#                                    process.akPu3PFJetSequence +

#                                    process.akPu4CaloJetSequence +
#                                    process.akVs4CaloJetSequence +
#                                    process.akVs4PFJetSequence +
#                                    process.akPu4PFJetSequence +

                                    process.hltPuAK4CaloJetsCorrectedIDPassedJetSequence+
                                    process.hltPuAK4CaloJetsIDPassedCorrKJetSequence+
                                    process.hltPuAK4CaloJetsIDPassedCorrLJetSequence

                                    # process.akPu5CaloJetSequence +
                                    # process.akVs5CaloJetSequence +
                                    # process.akVs5PFJetSequence +
                                    # process.akPu5PFJetSequence

                                    )

process.load('HeavyIonsAnalysis.EventAnalysis.hievtanalyzer_mc_cfi')
process.hiEvtAnalyzer.doMC = cms.bool(False) #the gen info dataformat has changed in 73X, we need to update hiEvtAnalyzer code
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
process.pfcandAnalyzer.pfPtMin = -9999
process.rechitanalyzer.EBTreePtMin = -9999
process.rechitanalyzer.EETreePtMin = -9999



#####################################################################################

#########################
# Track Analyzer
#########################
process.anaTrack.qualityStrings = cms.untracked.vstring(['highPurity','tight','loose'])

process.pixelTrack.qualityStrings = cms.untracked.vstring('highPurity')
process.hiTracks.cut = cms.string('quality("highPurity")')

# set track collection to iterative tracking
process.anaTrack.trackSrc = cms.InputTag("hiGeneralTracks")

# clusters missing in recodebug - to be resolved
process.anaTrack.doPFMatching = False
process.pixelTrack.doPFMatching = False

process.anaTrack.doSimVertex = True
process.anaTrack.doSimTrack = True
# process.ppTrack.fillSimTrack = True

process.load("SimTracker.TrackAssociation.trackingParticleRecoTrackAsssociation_cff")
process.tpRecoAssocGeneralTracks = process.trackingParticleRecoTrackAsssociation.clone()
process.tpRecoAssocGeneralTracks.label_tr = cms.InputTag("hiGeneralTracks")
process.quickTrackAssociatorByHits.ComponentName = cms.string('quickTrackAssociatorByHits')


#####################
# Photons
#####################

process.load('RecoHI.HiEgammaAlgos.HiEgamma_cff')

process.photonCoreHlt = process.photonCore.clone()
process.photonCoreHlt.scHybridBarrelProducer = cms.InputTag("hltCleanedHiCorrectedIslandBarrelSuperClustersHI")
process.photonCoreHlt.scIslandEndcapProducer = cms.InputTag("hltHiCorrectedIslandEndcapSuperClustersHI")

process.photonsHlt = process.photons.clone()
process.photonsHlt.photonCoreProducer = cms.InputTag("photonCoreHlt")


process.load('HeavyIonsAnalysis.PhotonAnalysis.ggHiNtuplizer_cfi')
process.ggHiNtuplizer.genParticleSrc = cms.InputTag("genParticles")
process.ggHiNtuplizerGED = process.ggHiNtuplizer.clone(recoPhotonSrc = cms.InputTag('gedPhotonsTmp'),
                                                       recoPhotonHiIsolationMap = cms.InputTag('photonIsolationHIProducerGED')
                                                       )

process.hltGGHiNtuplizer = process.ggHiNtuplizer.clone()
process.hltGGHiNtuplizer.recoPhotonSrc = cms.InputTag("photonsHlt")
process.hltGGHiNtuplizer.useValMapIso = False
process.hltGGHiNtuplizer.doVsIso = False

process.load('HeavyIonsAnalysis.PhotonAnalysis.ggHiNtuplizer_cfi')
process.ggHiNtuplizer.genParticleSrc = cms.InputTag("genParticles")
process.ggHiNtuplizerGED = process.ggHiNtuplizer.clone(recoPhotonSrc = cms.InputTag('gedPhotonsTmp'),
                                                       recoPhotonHiIsolationMap = cms.InputTag('photonIsolationHIProducerGED')
                                                       )

#####################

# HYDJET RECO file didn't have ak2GenJets and ak6GenJets as input, so removed them
# and ran our own hiGenJets sequence
# from RecoHI.HiJetAlgos.HiGenJets_cff import ak3HiGenJets, ak4HiGenJets
# from RecoJets.Configuration.GenJetParticles_cff import genParticlesForJets
# genParticlesForJets.ignoreParticleIDs += cms.vuint32( 12,14,16)

# process.hiSelectGenJets = cms.Sequence(
#     genParticlesForJets +
#     ak3HiGenJets +
#     ak4HiGenJets
# )

process.anaTrack.doSimTrack = cms.untracked.bool(False)

process.HiGenParticleAna.genParticleSrc = cms.untracked.InputTag("genParticles")

process.load("GeneratorInterface.HiGenCommon.HeavyIon_cff")


process.ana_step = cms.Path(process.heavyIon*
                            process.hltanalysis *
#temp                            process.hltobject *
                            process.centralityBin *
                            process.hiEvtAnalyzer*
                            process.HiGenParticleAna*
                            #process.hiGenJetsCleaned*
#                            process.quickTrackAssociatorByHits*
                            #process.tpRecoAssocGeneralTracks + #used in HiPFJetAnalyzer
                            #process.hiSelectGenJets +
                            process.jetSequences +
                            process.photonCoreHlt +
                            process.photonsHlt +
                            process.hltGGHiNtuplizer +
                            process.ggHiNtuplizer +
                            process.ggHiNtuplizerGED +
                            process.pfcandAnalyzer +
                            process.rechitAna +
                            process.HiForest +
                            # process.cutsTPForFak +
                            # process.cutsTPForEff +
                            process.anaTrack
                            #process.pixelTrack
                            )

process.load('HeavyIonsAnalysis.JetAnalysis.EventSelection_cff')
process.phltJetHI = cms.Path( process.hltJetHI )
process.pcollisionEventSelection = cms.Path(process.collisionEventSelection)
process.load('CommonTools.RecoAlgos.HBHENoiseFilterResultProducer_cfi')
process.pHBHENoiseFilterResultProducer = cms.Path( process.HBHENoiseFilterResultProducer )
process.phfCoincFilter = cms.Path(process.hfCoincFilter )
process.phfCoincFilter3 = cms.Path(process.hfCoincFilter3 )
process.pprimaryVertexFilter = cms.Path(process.primaryVertexFilter )
process.phltPixelClusterShapeFilter = cms.Path(process.siPixelRecHits*process.hltPixelClusterShapeFilter )
process.phiEcalRecHitSpikeFilter = cms.Path(process.hiEcalRecHitSpikeFilter )

process.pAna = cms.EndPath(process.skimanalysis)

# Customization
