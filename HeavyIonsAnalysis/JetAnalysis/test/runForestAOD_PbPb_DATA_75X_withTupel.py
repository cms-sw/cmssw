### HiForest Configuration
# Collisions: pp
# Type: Data
# Input: AOD

import FWCore.ParameterSet.Config as cms
process = cms.Process('HiForest')
process.options = cms.untracked.PSet()

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
                            fileNames = cms.untracked.vstring(
                                "file:/afs/cern.ch/user/a/azsigmon/public/ZMM_Skim_PbPb_PromptReco_44events.root"
                            )
)


# Number of events we want to process, -1 = all events
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1))


#####################################################################################
# Load Global Tag, Geometry, etc.
#####################################################################################

process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.Geometry.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')

from Configuration.AlCa.GlobalTag_condDBv2 import GlobalTag
#process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_data', '')
process.GlobalTag = GlobalTag(process.GlobalTag, '75X_dataRun2_v12', '')  #for now track GT manually, since centrality tables updated ex post facto
process.HiForest.GlobalTagLabel = process.GlobalTag.globaltag

from HeavyIonsAnalysis.Configuration.CommonFunctions_cff import overrideJEC_PbPb5020
process = overrideJEC_PbPb5020(process)

process.load("RecoHI.HiCentralityAlgos.CentralityBin_cfi")
process.centralityBin.Centrality = cms.InputTag("hiCentrality")
process.centralityBin.centralityVariable = cms.string("HFtowers")

#####################################################################################
# Define tree output
#####################################################################################

process.TFileService = cms.Service("TFileService",
                                   fileName=cms.string("HiForestAOD.root"))

#####################################################################################
# Additional Reconstruction and Analysis: Main Body
#####################################################################################

####################################################################################

#############################
# Jets
#############################
from Configuration.StandardSequences.ReconstructionHeavyIons_cff import voronoiBackgroundPF, voronoiBackgroundCalo

process.voronoiBackgroundPF = voronoiBackgroundPF
process.voronoiBackgroundCalo = voronoiBackgroundCalo
process.load('HeavyIonsAnalysis.JetAnalysis.jets.HiReRecoJets_HI_cff')

process.load('HeavyIonsAnalysis.JetAnalysis.jets.akPu2CaloJetSequence_PbPb_data_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akVs2CaloJetSequence_PbPb_data_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akVs2PFJetSequence_PbPb_data_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akPu2PFJetSequence_PbPb_data_cff')

process.load('HeavyIonsAnalysis.JetAnalysis.jets.akPu3CaloJetSequence_PbPb_data_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akVs3CaloJetSequence_PbPb_data_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akVs3PFJetSequence_PbPb_data_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akPu3PFJetSequence_PbPb_data_cff')

process.load('HeavyIonsAnalysis.JetAnalysis.jets.akPu4CaloJetSequence_PbPb_data_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akVs4CaloJetSequence_PbPb_data_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akVs4PFJetSequence_PbPb_data_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akPu4PFJetSequence_PbPb_data_cff')

process.load('HeavyIonsAnalysis.JetAnalysis.jets.akPu5CaloJetSequence_PbPb_data_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akVs5CaloJetSequence_PbPb_data_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akVs5PFJetSequence_PbPb_data_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.akPu5PFJetSequence_PbPb_data_cff')


process.highPurityTracks = cms.EDFilter("TrackSelector",
                                        src = cms.InputTag("hiGeneralTracks"),
                                        cut = cms.string('quality("highPurity")'))

process.load("RecoVertex.PrimaryVertexProducer.OfflinePrimaryVertices_cfi")
process.offlinePrimaryVertices.TrackLabel = 'highPurityTracks'

process.jetSequences = cms.Sequence(
    voronoiBackgroundPF+
    voronoiBackgroundCalo+

    process.akPu2CaloJets +
    process.akPu2PFJets +
    process.akVs2CaloJets +
    process.akVs2PFJets +

    #process.akPu3CaloJets +
    #process.akPu3PFJets +
    process.akVs3CaloJets +
    process.akVs3PFJets +

    #process.akPu4CaloJets +
    #process.akPu4PFJets +
    process.akVs4CaloJets +
    process.akVs4PFJets +

    process.akPu5CaloJets +
    process.akPu5PFJets +
    process.akVs5CaloJets +
    process.akVs5PFJets +

    process.highPurityTracks +
    process.offlinePrimaryVertices +

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
    process.akPu5PFJetSequence
    )

#####################################################################################

############################
# Event Analysis
############################
process.load('HeavyIonsAnalysis.EventAnalysis.hievtanalyzer_data_cfi')
process.load('HeavyIonsAnalysis.EventAnalysis.hltanalysis_cff')
process.load("HeavyIonsAnalysis.JetAnalysis.pfcandAnalyzer_cfi")
process.pfcandAnalyzer.skipCharged = False
process.pfcandAnalyzer.pfPtMin = 0
process.load("HeavyIonsAnalysis.JetAnalysis.hcalNoise_cff")

#####################################################################################

#########################
# Track Analyzer
#########################
process.load('HeavyIonsAnalysis.JetAnalysis.ExtraTrackReco_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.TrkAnalyzers_cff')
# process.load("HeavyIonsAnalysis.TrackAnalysis.METAnalyzer_cff")


####################################################################################

#####################
# Photons
#####################
process.load('HeavyIonsAnalysis.PhotonAnalysis.ggHiNtuplizer_cfi')
process.ggHiNtuplizer.doGenParticles = False
process.ggHiNtuplizerGED = process.ggHiNtuplizer.clone(recoPhotonSrc = cms.InputTag('gedPhotonsTmp'),
                                                       recoPhotonHiIsolationMap = cms.InputTag('photonIsolationHIProducerGED')
)

#####################################################################################

#########################
# Tupel and related PAT objects
#########################

process.load("MuonAnalysis.MuonAssociators.patMuonsWithTrigger_cff")
process.muonMatchHLTL2.maxDeltaR = 0.3
process.muonMatchHLTL3.maxDeltaR = 0.1
from MuonAnalysis.MuonAssociators.patMuonsWithTrigger_cff import *
process.patTriggerFull.l1GtReadoutRecordInputTag = cms.InputTag("gtDigis","","RECO")                 
process.patTrigger.collections.remove("hltL3MuonCandidates")
process.patTrigger.collections.append("hltHIL3MuonCandidates")
process.muonMatchHLTL3.matchedCuts = cms.string('coll("hltHIL3MuonCandidates")')
process.patMuonsWithoutTrigger.pvSrc = cms.InputTag("hiSelectedVertex")

process.load("PhysicsTools.PatAlgos.producersLayer1.photonProducer_cfi")
process.patPhotons.photonSource = cms.InputTag("gedPhotonsTmp")
process.patPhotons.electronSource = cms.InputTag("gedGsfElectronsTmp")
process.patPhotons.reducedBarrelRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEB")
process.patPhotons.reducedEndcapRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEE")
process.patPhotons.addPhotonID = cms.bool(False)
process.patPhotonSequence = cms.Sequence(process.patPhotons)

process.load("PhysicsTools.PatAlgos.producersLayer1.electronProducer_cfi")
process.patElectrons.electronSource = cms.InputTag("gedGsfElectronsTmp")
process.patElectrons.reducedBarrelRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEB")
process.patElectrons.reducedEndcapRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEE")
process.patElectrons.pvSrc = cms.InputTag("hiSelectedVertex")
process.patElectrons.addElectronID = cms.bool(False)
process.patElectronSequence = cms.Sequence(process.patElectrons)

process.tupel = cms.EDAnalyzer("Tupel",
#  trigger      = cms.InputTag( "patTrigger" ),
#  triggerEvent = cms.InputTag( "patTriggerEvent" ),
#  triggerSummaryLabel = cms.InputTag("hltTriggerSummaryAOD","","HLT"),
  photonSrc    = cms.untracked.InputTag("patPhotons"),
  vtxSrc       = cms.untracked.InputTag("offlinePrimaryVertices"),
  electronSrc  = cms.untracked.InputTag("patElectrons"),
  muonSrc      = cms.untracked.InputTag("patMuonsWithTrigger"),
#  tauSrc      = cms.untracked.InputTag("slimmedPatTaus"),
  jetSrc       = cms.untracked.InputTag("akPu4PFpatJetsWithBtagging"),
  metSrc       = cms.untracked.InputTag("patMETsPF"),
  genSrc       = cms.untracked.InputTag("genParticles"),
  gjetSrc      = cms.untracked.InputTag('ak4GenJets'),
  muonMatch    = cms.string( 'muonTriggerMatchHLTMuons' ),
  muonMatch2   = cms.string( 'muonTriggerMatchHLTMuons2' ),
  elecMatch    = cms.string( 'elecTriggerMatchHLTElecs' ),
#  mSrcRho      = cms.untracked.InputTag('fixedGridRhoFastjetAll'),
  CalojetLabel = cms.untracked.InputTag('ak4CalopatJets'),
  metSource    = cms.VInputTag("slimmedMETs","slimmedMETs","slimmedMETs","slimmedMETs"),
  lheSource    = cms.untracked.InputTag('source')
)

####################################################################################

#########################
# Main analysis list
#########################

process.ana_step = cms.Path(process.hltanalysis *
                            process.centralityBin *
                            process.hiEvtAnalyzer*
                            process.patMuonsWithTriggerSequence *
                            process.jetSequences *
                            process.patPhotonSequence *
                            process.patElectronSequence *
                            process.tupel +
                            process.ggHiNtuplizer +
                            process.ggHiNtuplizerGED +
                            process.pfcandAnalyzer +
                            process.HiForest +
                            process.trackSequencesPbPb +
                            process.hcalNoise
                            )

#####################################################################################

#########################
# Event Selection
#########################

process.load('HeavyIonsAnalysis.JetAnalysis.EventSelection_cff')
process.pcollisionEventSelection = cms.Path(process.collisionEventSelectionAOD)
process.load('CommonTools.RecoAlgos.HBHENoiseFilterResultProducer_cfi')
process.pHBHENoiseFilterResultProducer = cms.Path( process.HBHENoiseFilterResultProducer )
process.pprimaryVertexFilter = cms.Path(process.primaryVertexFilter )

process.load('HeavyIonsAnalysis.Configuration.hfCoincFilter_cff')
process.phfCoincFilter1 = cms.Path(process.hfCoincFilter)
process.phfCoincFilter2 = cms.Path(process.hfCoincFilter2)
process.phfCoincFilter3 = cms.Path(process.hfCoincFilter3)
process.phfCoincFilter4 = cms.Path(process.hfCoincFilter4)
process.phfCoincFilter5 = cms.Path(process.hfCoincFilter5)

process.pclusterCompatibilityFilter = cms.Path(process.clusterCompatibilityFilter)

process.pAna = cms.EndPath(process.skimanalysis)

# Customization
