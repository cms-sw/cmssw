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
"file:/afs/cern.ch/work/i/istaslis/public/bjet5tev/forest/CMSSW_7_5_5_patch4/src/HeavyIonsAnalysis/JetAnalysis/test/step3_RAW2DIGI_L1Reco_RECO_9_1_GTa.root"
#"/store/user/mnguyen/Pythia8_Hydjet_bjet80_5020GeV_GEN-SIM_v6/Pythia8_Hydjet_bjet80_5020GeV_RECO_v6/step3_RAW2DIGI_L1Reco_RECO_9_1_GTa.root"
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

from Configuration.AlCa.GlobalTag_condDBv2 import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc_HIon', '')

process.GlobalTag.snapshotTime = cms.string("9999-12-31 23:59:59.000")
process.GlobalTag.toGet.extend([
   cms.PSet(record = cms.string("HeavyIonRcd"),
      tag =
cms.string("CentralityTable_HFtowers200_HydjetDrum5_v755x01_mc"),
      connect = cms.string("frontier://FrontierProd/CMS_CONDITIONS"),
      label = cms.untracked.string("HFtowers")
   ),
])

from HeavyIonsAnalysis.Configuration.CommonFunctions_cff import overrideJEC_PbPb5020
process = overrideJEC_PbPb5020(process)

process.load("RecoHI.HiCentralityAlgos.CentralityBin_cfi")

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

process.load("CondCore.DBCommon.CondDBCommon_cfi")

from CondCore.DBCommon.CondDBSetup_cfi import *
process.uetable = cms.ESSource("PoolDBESSource",
      DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(0)
        ),
      timetype = cms.string('runnumber'),
      toGet = cms.VPSet(
          cms.PSet(record = cms.string("JetCorrectionsRecord"),
                   tag = cms.string("UETableCompatibilityFormat_PF_HYDJET_5020GeV_754_38T_v00_mc"),
                   label = cms.untracked.string("UETable_PF")
          ),
          cms.PSet(record = cms.string("JetCorrectionsRecord"),
                   tag = cms.string("UETableCompatibilityFormat_Calo_HYDJET_5020GeV_754_38T_v00_mc"),
                   label = cms.untracked.string("UETable_Calo")
          )
      ), 
      connect = cms.string("frontier://FrontierProd/CMS_CONDITIONS")
)
process.es_prefer_uetable = cms.ESPrefer('PoolDBESSource','uetable')

process.load('HeavyIonsAnalysis.JetAnalysis.jets.HiReRecoJets_HI_cff')


# process.voronoiBackgroundPF.tableLabel = cms.string("UETable_PFMarta")
# process.voronoiBackgroundCalo.tableLabel = cms.string("UETable_CaloMarta")

##############
###Gen jets###
##############
process.load('RecoHI.HiJetAlgos.HiGenJets_cff')
process.load('RecoJets.Configuration.GenJetParticles_cff')

process.akHiGenJets = cms.Sequence(
                             process.genParticlesForJets +
                             process.ak1HiGenJets +
                             process.ak2HiGenJets +
                             process.ak3HiGenJets +
                             process.ak4HiGenJets +
                             process.ak5HiGenJets +
                             process.ak6HiGenJets)


							 
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akPu1CaloJetSequence_PbPb_mc_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akVs1CaloJetSequence_PbPb_mc_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akVs1PFJetSequence_PbPb_mc_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akPu1PFJetSequence_PbPb_mc_cff')

process.load('HeavyIonsAnalysis.JetAnalysis.jets.akPu2CaloJetSequence_PbPb_mc_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akVs2CaloJetSequence_PbPb_mc_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akVs2PFJetSequence_PbPb_mc_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akPu2PFJetSequence_PbPb_mc_cff')

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

process.load('HeavyIonsAnalysis.JetAnalysis.jets.akPu6CaloJetSequence_PbPb_mc_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akVs6CaloJetSequence_PbPb_mc_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akVs6PFJetSequence_PbPb_mc_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akPu6PFJetSequence_PbPb_mc_cff')

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

process.akVs1PFJetAnalyzer.doSubEvent = True
process.akVs1CaloJetAnalyzer.doSubEvent = True

process.akVs2PFJetAnalyzer.doSubEvent = True
process.akVs2CaloJetAnalyzer.doSubEvent = True

process.akVs3PFJetAnalyzer.doSubEvent = True
process.akVs3CaloJetAnalyzer.doSubEvent = True

process.akVs4PFJetAnalyzer.doSubEvent = True
process.akVs4CaloJetAnalyzer.doSubEvent = True

process.akVs5PFJetAnalyzer.doSubEvent = True
process.akVs5CaloJetAnalyzer.doSubEvent = True

process.akVs6PFJetAnalyzer.doSubEvent = True
process.akVs6CaloJetAnalyzer.doSubEvent = True

process.jetSequences = cms.Sequence(process.voronoiBackgroundPF+
                                    process.voronoiBackgroundCalo+
									
                                    process.akPu1CaloJetSequence +
                                    process.akVs1CaloJetSequence +
                                    process.akVs1PFJetSequence +
                                    process.akPu1PFJetSequence +
									
									process.akPu2CaloJetSequence +
                                    process.akVs2CaloJetSequence +
                                    process.akVs2PFJetSequence +
                                    process.akPu2PFJetSequence +
									
									process.akPu3CaloJetSequence +
                                    process.akVs3CaloJetSequence +
                                    process.akVs3PFJetSequence +
                                    process.akPu3PFJetSequence +

                                    process.akPu4CaloJetSequence +
                                    process.akVs4CaloJetSequence +
                                    process.akVs4PFJetSequence +
                                    process.akPu4PFJetSequence +
									
                                    process.akPu5CaloJetSequence +
                                    process.akVs5CaloJetSequence +
                                    process.akVs5PFJetSequence +
                                    process.akPu5PFJetSequence +

                                    process.akPu6CaloJetSequence +
                                    process.akVs6CaloJetSequence +
                                    process.akVs6PFJetSequence +
                                    process.akPu6PFJetSequence 
                                    )

process.load('HeavyIonsAnalysis.EventAnalysis.hievtanalyzer_mc_cfi')
process.hiEvtAnalyzer.doMC = cms.bool(False) #the gen info dataformat has changed in 73X, we need to update hiEvtAnalyzer code
process.load('HeavyIonsAnalysis.EventAnalysis.HiMixAnalyzerRECO_cff')

process.load('HeavyIonsAnalysis.EventAnalysis.hltanalysis_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.HiGenAnalyzer_cfi')

process.HiGenParticleAna.ptMin = -9999

#####################################################################################
# To be cleaned

process.load('HeavyIonsAnalysis.JetAnalysis.ExtraTrackReco_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.TrkAnalyzers_MC_cff')
process.load("HeavyIonsAnalysis.TrackAnalysis.METAnalyzer_cff")
process.load("HeavyIonsAnalysis.JetAnalysis.pfcandAnalyzer_cfi")
process.load('HeavyIonsAnalysis.JetAnalysis.rechitanalyzer_aod_cfi')

process.rechitAna = cms.Sequence(process.rechitanalyzer+process.pfTowers)
process.pfcandAnalyzer.skipCharged = False
process.pfcandAnalyzer.pfPtMin = 0

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
process.anaTrack.doPFMatching = True
process.pixelTrack.doPFMatching = False

process.anaTrack.doSimVertex = False
process.anaTrack.doSimTrack = False
# process.ppTrack.fillSimTrack = True

process.load("SimTracker.TrackAssociation.trackingParticleRecoTrackAsssociation_cff")
process.tpRecoAssocGeneralTracks = process.trackingParticleRecoTrackAsssociation.clone()
process.tpRecoAssocGeneralTracks.label_tr = cms.InputTag("hiGeneralTracks")
process.quickTrackAssociatorByHits.ComponentName = cms.string('quickTrackAssociatorByHits')


process.quickTrackAssociatorByHits.SimToRecoDenominator = cms.string('reco')
process.quickTrackAssociatorByHits.Cut_RecoToSim = cms.double(0.5)
process.quickTrackAssociatorByHits.Quality_SimToReco = cms.double(0.0)

#####################
# L1 Digis
#####################

process.load('EventFilter.L1TRawToDigi.caloStage1Digis_cfi')
process.caloStage1Digis.InputLabel = cms.InputTag("rawDataCollector")

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

#####################
# Photons
#####################

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

# process.anaTrack.doSimTrack = cms.untracked.bool(False)

process.HiGenParticleAna.genParticleSrc = cms.untracked.InputTag("genParticles")

process.load("GeneratorInterface.HiGenCommon.HeavyIon_cff")

# process.heavyIon.src = cms.InputTag("generator")
# process.heavyIon.ptCut = cms.double(0)
# process.heavyIon.doReco = cms.bool(True)
# process.heavyIon.doMC = cms.bool(False)

process.ana_step = cms.Path(
                            process.mixAnalyzer *
                            process.hltanalysis *
#temp                            process.hltobject *
                            process.centralityBin *
                            process.hiEvtAnalyzer*
                            process.HiGenParticleAna*
                            #process.hiGenJetsCleaned*
                            process.quickTrackAssociatorByHits*
#                            process.tpRecoAssocGeneralTracks + #used in HiPFJetAnalyzer
                            #process.hiSelectGenJets +
                            process.akHiGenJets +
                            process.hiReRecoCaloJets +
                            process.hiReRecoPFJets +
                            process.jetSequences +                            
                            process.ggHiNtuplizer +
                            process.ggHiNtuplizerGED +
                            process.pfcandAnalyzer +
                            process.rechitAna +
                            process.HiForest +
                            # process.cutsTPForFak +
                            # process.cutsTPForEff +
                            process.anaTrack 
                            #process.pixelTrack
			                      # process.L1Sequence +
                            # process.L1EmulatorUnpacker
                            )

process.load('HeavyIonsAnalysis.JetAnalysis.EventSelection_cff')
process.load('CommonTools.RecoAlgos.HBHENoiseFilterResultProducer_cfi')
process.load('HeavyIonsAnalysis.Configuration.hfCoincFilter_cff')
process.phfPosTowers=cms.Path(process.hfposTowers)
process.phfNegTowers=cms.Path(process.hfnegTowers)
process.phfPosFilter=cms.Path(process.hfposFilter)
process.phfNegFilter=cms.Path(process.hfnegFilter)
process.phfPosFilter2=cms.Path(process.hfposFilter2)
process.phfNegFilter2=cms.Path(process.hfnegFilter2)
process.phfPosFilter3=cms.Path(process.hfposFilter3)
process.phfNegFilter3=cms.Path(process.hfnegFilter3)
process.phfPosFilter4=cms.Path(process.hfposFilter4)
process.phfNegFilter4=cms.Path(process.hfnegFilter4)
process.phfPosFilter5=cms.Path(process.hfposFilter5)
process.phfNegFilter5=cms.Path(process.hfnegFilter5)


process.phfCoincFilter = cms.Path(process.hfCoincFilter )
process.phfCoincFilter2 = cms.Path(process.hfCoincFilter2 )
process.phfCoincFilter3 = cms.Path(process.hfCoincFilter3 )
process.phfCoincFilter4 = cms.Path(process.hfCoincFilter4 )
process.phfCoincFilter5 = cms.Path(process.hfCoincFilter5 )

process.phltJetHI = cms.Path( process.hltJetHI )
process.pcollisionEventSelection = cms.Path(process.collisionEventSelectionAOD)
process.pHBHENoiseFilterResultProducer = cms.Path( process.HBHENoiseFilterResultProducer )
process.pprimaryVertexFilter = cms.Path(process.primaryVertexFilter )
#process.phltPixelClusterShapeFilter = cms.Path(process.siPixelRecHits*process.hltPixelClusterShapeFilter )
#process.phiEcalRecHitSpikeFilter = cms.Path(process.hiEcalRecHitSpikeFilter )

process.pAna = cms.EndPath(process.skimanalysis)

# Customization
