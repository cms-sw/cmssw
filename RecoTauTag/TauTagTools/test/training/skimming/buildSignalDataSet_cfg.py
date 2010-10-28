import FWCore.ParameterSet.Config as cms
import RecoTauTag.TauTagTools.RecoTauCommonJetSelections_cfi as common

'''

buildSignalDataSet_cfg

Build a reduced dataset containing gen-level infomration and reco::PFJets
that are matched to "common" hadronic tau decay modes.  Used to build the signal
training sample for the TaNC.

Author: Evan K. Friis (UC Davis)

'''

process = cms.Process("TANC")

maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring()
process.source = cms.Source("PoolSource", fileNames = readFiles,
                            secondaryFileNames = secFiles)
readFiles.extend(
    [ '/store/relval/CMSSW_3_8_4/RelValZTT/GEN-SIM-RECO/START38_V12-v1/0024/D65A6981-86C2-DF11-8FAE-00304867C16A.root',
     '/store/relval/CMSSW_3_8_4/RelValZTT/GEN-SIM-RECO/START38_V12-v1/0024/CC691F0C-96C2-DF11-8C28-00261894396F.root',
     '/store/relval/CMSSW_3_8_4/RelValZTT/GEN-SIM-RECO/START38_V12-v1/0024/AC15C30F-81C2-DF11-940E-002618943924.root',
     '/store/relval/CMSSW_3_8_4/RelValZTT/GEN-SIM-RECO/START38_V12-v1/0024/8E51890C-82C2-DF11-B7ED-002618943949.root',
     '/store/relval/CMSSW_3_8_4/RelValZTT/GEN-SIM-RECO/START38_V12-v1/0024/46893A0C-82C2-DF11-B8A3-003048678BAE.root'
    ]
)

# Load standard services
process.load("Configuration.StandardSequences.Services_cff")
process.TFileService = cms.Service(
    "TFileService",
    fileName = cms.string("signal_skim_plots.root")
)

#################################################################
# Rebuild the PF event content.
#################################################################

# We need to rerun particle flow so this process owns it :/
# Adapted from RecoParticleFlow/Configuration/test/RecoToDisplay_cfg.py
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
from Configuration.PyReleaseValidation.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['mc']

# Local re-reco: Produce tracker rechits, pf rechits and pf clusters
process.localReReco = cms.Sequence(process.siPixelRecHits+
                                   process.siStripMatchedRecHits+
                                   process.particleFlowCluster)

# Track re-reco
process.globalReReco =  cms.Sequence(process.offlineBeamSpot+
                                     process.recopixelvertexing+
                                     process.ckftracks+
                                     process.caloTowersRec+
                                     process.vertexreco+
                                     process.recoJets+
                                     process.muonrecoComplete+
                                     process.electronGsfTracking+
                                     process.metreco)

# Particle Flow re-processing
process.pfReReco = cms.Sequence(process.particleFlowReco+
                                process.ak5PFJets)

process.rereco = cms.Sequence(
    process.localReReco*
    process.globalReReco*
    process.pfReReco)

#################################################################
# Build and select true taus
#################################################################

# Load tau truth builders
process.load("RecoTauTag.TauTagTools.TauTruthProduction_cfi")

process.selectedTrueHadronicTaus = cms.EDFilter(
    "GenJetSelector",
    # We only care about hadronic decay modes we will use later
    src = cms.InputTag("trueCommonHadronicTaus"),
    cut = cms.string('pt > 5.0 & abs(eta) < 2.5'),
    # Don't keep events that have no good taus
    filter = cms.bool(True),
)

# Require a minimum PT cut on our jets
process.selectedRecoJets = cms.EDFilter(
    "CandViewRefSelector",
    src = cms.InputTag("ak5PFJets"),
    cut = common.kinematic_selection,
    filter = cms.bool(True)
)

# Match our jets to the true taus
process.recoJetsTruthMatching = cms.EDProducer(
    "GenJetMatcher",
    src = cms.InputTag("selectedRecoJets"),
    matched = cms.InputTag("selectedTrueHadronicTaus"),
    mcPdgId     = cms.vint32(),                      # n/a
    mcStatus    = cms.vint32(),                      # n/a
    checkCharge = cms.bool(False),
    maxDeltaR   = cms.double(0.15),
    maxDPtRel   = cms.double(3.0),
    # Forbid two RECO objects to match to the same GEN object
    resolveAmbiguities    = cms.bool(True),
    resolveByMatchQuality = cms.bool(True),
)

# Select only those jets matched to true taus
process.signalJetsMatched = cms.EDFilter(
    "CandViewGenJetMatchRefSelector",
    src = cms.InputTag("ak5PFJets"),
    matching = cms.InputTag("recoJetsTruthMatching"),
    # Don't keep events with no matches
    filter = cms.bool(True)
)

# Make the signal jets proper PFJet refs
process.signalJets = cms.EDProducer(
    "PFJetRefsCastFromCandView",
    src = cms.InputTag("signalJetsMatched"),
)

# Plot signal jets
process.plotSignalJets = cms.EDAnalyzer(
    "CandViewHistoAnalyzer",
    src = cms.InputTag("signalJets"),
    histograms = common.jet_histograms
)

# DO SIGNAL PRESELECTION
process.signalJetsLeadObject = cms.EDFilter(
    "JetViewRefSelector",
    src = cms.InputTag("signalJets"),
    cut = common.lead_object_jet_selection,
    filter = cms.bool(True),
)

process.plotSignalJetsLeadObject = cms.EDAnalyzer(
    "CandViewHistoAnalyzer",
    src = cms.InputTag("signalJetsLeadObject"),
    histograms = common.jet_histograms
)

# Turn off for now.
process.signalJetsCollimationCut = cms.EDFilter(
    "JetViewRefSelector",
    src = cms.InputTag("signalJetsLeadObject"),
    cut = cms.string("pt()*sqrt(etaetaMoment()) > -1"),
    filter = cms.bool(True),
)

process.preselectedSignalJets = cms.EDProducer(
    "PFJetRefsCastFromCandView",
    src = cms.InputTag("signalJetsLeadObject"),
)

process.plotPreselectedSignalJets = cms.EDAnalyzer(
    "CandViewHistoAnalyzer",
    src = cms.InputTag("preselectedSignalJets"),
    histograms = common.jet_histograms
)

# Produce and save PiZeros for the signal jets
import RecoTauTag.RecoTau.RecoTauPiZeroProducer_cfi as PiZeroProd
process.signalJetsRecoTauPiZeros = PiZeroProd.ak5PFJetsRecoTauPiZeros.clone(
    jetSrc = cms.InputTag("preselectedSignalJets")
)

# To prevent the "MismatchedInputFiles" exception during training, we must make
# the contents of the signal file a superset of the background file.

process.backgroundJetRefs = cms.EDFilter(
    "CandViewRefSelector",
    src = cms.InputTag("signalJetsMatched"),
    cut = cms.string("pt < 0"), # always fails
    filter = cms.bool(False)
)

process.preselectedBackgroundJets = cms.EDProducer(
    "PFJetRefsCastFromCandView",
    src = cms.InputTag("backgroundJetRefs"),
)

process.backgroundJetsRecoTauPiZeros = process.signalJetsRecoTauPiZeros.clone(
    src = cms.InputTag("preselectedBackgroundJets"),
)

process.addFakeBackground = cms.Sequence(
    process.backgroundJetRefs *
    process.preselectedBackgroundJets *
    process.backgroundJetsRecoTauPiZeros
)

process.selectSignal = cms.Path(
    process.tauGenJets *
    process.trueCommonHadronicTaus *
    process.selectedTrueHadronicTaus *
    process.rereco *
    process.selectedRecoJets *
    process.recoJetsTruthMatching *
    process.signalJetsMatched *
    process.signalJets *
    process.plotSignalJets *
    process.signalJetsLeadObject *
    process.plotSignalJetsLeadObject *
    #process.signalJetsCollimationCut *
    process.preselectedSignalJets *
    process.plotPreselectedSignalJets *
    process.signalJetsRecoTauPiZeros *
    process.addFakeBackground
)

# Keep only a subset of data
poolOutputCommands = cms.untracked.vstring(
    'drop *',
    'keep *_ak5PFJets_*_TANC',
    'keep *_offlinePrimaryVertices_*_TANC',
    'keep *_particleFlow_*_TANC',
    'keep recoTracks_generalTracks_*_TANC',
    'keep recoTracks_electronGsfTracks_*_TANC',
    'keep *_genParticles_*_*', # this product is okay, since we dont' need it in bkg
    'keep *_selectedTrueHadronicTaus_*_*',
    'keep *_preselectedSignalJets_*_*',
    'keep *_signalJetsRecoTauPiZeros_*_*',
    # These two products are needed to make signal content a superset
    'keep *_preselectedBackgroundJets_*_*',
    'keep *_backgroundJetsRecoTauPiZeros_*_*',
)

# Write output
process.write = cms.OutputModule(
    "PoolOutputModule",
    fileName = cms.untracked.string("signal_training.root"),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring("selectSignal"),
    ),
    outputCommands = poolOutputCommands
)
process.out = cms.EndPath(process.write)

# Print out trigger information
process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )
