### HiForest Configuration
# Collisions: PbPb
# Type: MonteCarlo
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
                            duplicateCheckMode = cms.untracked.string("noDuplicateCheck"),
                            fileNames = cms.untracked.vstring(
                                "file:/afs/cern.ch/user/a/azsigmon/public/Z30mumuJet_HYDJET_5020GeV_RECODEBUG_1.root"
                            )
)

# Number of events we want to process, -1 = all events
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)


#####################################################################################
# Load Global Tag, Geometry, etc.
#####################################################################################

process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.Geometry.GeometryDB_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')

from Configuration.AlCa.GlobalTag_condDBv2 import GlobalTag
#process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc_HIon', '')
process.GlobalTag = GlobalTag(process.GlobalTag, '75X_mcRun2_HeavyIon_v11', '') #for now track GT manually, since centrality tables updated ex post facto
process.HiForest.GlobalTagLabel = process.GlobalTag.globaltag

from HeavyIonsAnalysis.Configuration.CommonFunctions_cff import overrideJEC_PbPb5020
process = overrideJEC_PbPb5020(process)

process.load("RecoHI.HiCentralityAlgos.CentralityBin_cfi")

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
process.load('HeavyIonsAnalysis.JetAnalysis.FullJetSequence_nominalPbPb')

# Use this one for JEC:
#process.load('HeavyIonsAnalysis.JetAnalysis.FullJetSequence_JECPbPb')

####################################################################################

#############################
# Gen Analyzer
#############################
process.load('HeavyIonsAnalysis.EventAnalysis.HiMixAnalyzerRECO_cff')
process.load('GeneratorInterface.HiGenCommon.HeavyIon_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.HiGenAnalyzer_cfi')
process.load('HeavyIonsAnalysis.EventAnalysis.runanalyzer_cff')
process.HiGenParticleAna.genParticleSrc = cms.untracked.InputTag("genParticles")
# Temporary disactivation - until we have DIGI & RECO in CMSSW_7_5_7_patch4
process.HiGenParticleAna.doHI = False


#####################################################################################

############################
# Event Analysis
############################
process.load('HeavyIonsAnalysis.EventAnalysis.hievtanalyzer_mc_cfi')
process.hiEvtAnalyzer.doMC = cms.bool(False) #the gen info dataformat has changed in 73X, we need to update hiEvtAnalyzer code
process.load('HeavyIonsAnalysis.EventAnalysis.hltanalysis_cff')
process.load("HeavyIonsAnalysis.JetAnalysis.pfcandAnalyzer_cfi")
process.pfcandAnalyzer.skipCharged = False
process.pfcandAnalyzer.pfPtMin = 0

#####################################################################################

#########################
# Track Analyzer
#########################
process.load('HeavyIonsAnalysis.JetAnalysis.ExtraTrackReco_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.TrkAnalyzers_cff')

# Use this instead for track corrections
## process.load('HeavyIonsAnalysis.JetAnalysis.TrkAnalyzers_Corr_cff')

#####################################################################################

#####################
# Photons
#####################

process.load('HeavyIonsAnalysis.PhotonAnalysis.ggHiNtuplizer_cfi')
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

process.load("PhysicsTools.PatAlgos.mcMatchLayer0.photonMatch_cfi")
process.photonMatch.src = cms.InputTag("gedPhotonsTmp")
process.load("PhysicsTools.PatAlgos.producersLayer1.photonProducer_cfi")
process.patPhotons.photonSource = cms.InputTag("gedPhotonsTmp")
process.patPhotons.electronSource = cms.InputTag("gedGsfElectronsTmp")
process.patPhotons.reducedBarrelRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEB")
process.patPhotons.reducedEndcapRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEE")
process.patPhotons.addPhotonID = cms.bool(False)
process.patPhotonSequence = cms.Sequence(process.photonMatch * process.patPhotons)

process.load("PhysicsTools.PatAlgos.mcMatchLayer0.electronMatch_cfi")
process.electronMatch.src = cms.InputTag("gedGsfElectronsTmp")
process.load("PhysicsTools.PatAlgos.producersLayer1.electronProducer_cfi")
process.patElectrons.electronSource = cms.InputTag("gedGsfElectronsTmp")
process.patElectrons.reducedBarrelRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEB")
process.patElectrons.reducedEndcapRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEE")
process.patElectrons.pvSrc = cms.InputTag("hiSelectedVertex")
process.patElectrons.addElectronID = cms.bool(False)
process.patElectronSequence = cms.Sequence(process.electronMatch * process.patElectrons)

process.tupel = cms.EDAnalyzer("Tupel",
  trigger      = cms.InputTag( "patTrigger" ),
#  triggerEvent = cms.InputTag( "patTriggerEvent" ),
#  triggerSummaryLabel = cms.InputTag("hltTriggerSummaryAOD","","HLT"),
  photonSrc    = cms.untracked.InputTag("patPhotons"),
  vtxSrc       = cms.untracked.InputTag("hiSelectedVertex"),
  electronSrc  = cms.untracked.InputTag("patElectrons"),
  muonSrc      = cms.untracked.InputTag("patMuonsWithTrigger"),
#  tauSrc      = cms.untracked.InputTag("slimmedPatTaus"),
  jetSrc       = cms.untracked.InputTag("akPu4PFpatJetsWithBtagging"),
  metSrc       = cms.untracked.InputTag("patMETsPF"),
  genSrc       = cms.untracked.InputTag("genParticles"),
  gjetSrc      = cms.untracked.InputTag('ak4HiGenJets'),
  muonMatch    = cms.string( 'muonTriggerMatchHLTMuons' ),
  muonMatch2   = cms.string( 'muonTriggerMatchHLTMuons2' ),
  elecMatch    = cms.string( 'elecTriggerMatchHLTElecs' ),
  mSrcRho      = cms.untracked.InputTag('fixedGridRhoFastjetAll'),
  CalojetLabel = cms.untracked.InputTag('ak4CalopatJets'),
  metSource    = cms.VInputTag("slimmedMETs","slimmedMETs","slimmedMETs","slimmedMETs"),
  lheSource    = cms.untracked.InputTag('source')
)

####################################################################################

#########################
# Main analysis list
#########################

process.ana_step = cms.Path(
# Temporary disactivation - until we have DIGI & RECO in CMSSW_7_5_7_patch4
# process.mixAnalyzer *
                            process.runAnalyzer *
                            process.hltanalysis *
                            process.centralityBin *
                            process.hiEvtAnalyzer*
                            process.HiGenParticleAna*
                            process.patMuonsWithTriggerSequence *
                            process.jetSequences *
                            process.patPhotonSequence *
                            process.patElectronSequence *
                            process.tupel +
                            process.ggHiNtuplizer +
                            process.ggHiNtuplizerGED +
                            process.pfcandAnalyzer +
                            process.HiForest +
                            process.trackSequencesPbPb
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
