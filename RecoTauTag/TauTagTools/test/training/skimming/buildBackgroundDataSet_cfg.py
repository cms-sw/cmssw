import FWCore.ParameterSet.Config as cms
import RecoTauTag.TauTagTools.RecoTauCommonJetSelections_cfi as common

'''

buildBackgroundDataSet_cfg

Build a reduced dataset containing selected reco::PFJets (from real data)
selected to remove trigger bias.  Used to build the background training sample
for the TaNC.

Author: Evan K. Friis (UC Davis)

'''

process = cms.Process("TANC")

maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring()
process.source = cms.Source("PoolSource", fileNames = readFiles,
                            secondaryFileNames = secFiles)
readFiles.extend([
    '/store/relval/CMSSW_3_8_4/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_38Y_V12-v1/0025/E6E7A0FC-9AC2-DF11-A9C1-003048679076.root',
    '/store/relval/CMSSW_3_8_4/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_38Y_V12-v1/0024/2EB50F82-7DC2-DF11-9826-0026189437E8.root',
    '/store/relval/CMSSW_3_8_4/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_38Y_V12-v1/0023/CA871CF9-76C2-DF11-A962-003048678FE4.root',
    '/store/relval/CMSSW_3_8_4/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_38Y_V12-v1/0023/B8E56D7E-77C2-DF11-8E09-001A928116B8.root',
    '/store/relval/CMSSW_3_8_4/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_38Y_V12-v1/0023/AC27459D-78C2-DF11-9FBF-002618943974.root',
    '/store/relval/CMSSW_3_8_4/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_38Y_V12-v1/0023/9882FA2D-78C2-DF11-9198-002618943934.root',
    '/store/relval/CMSSW_3_8_4/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_38Y_V12-v1/0023/60808F24-7CC2-DF11-8396-002618943864.root',
    '/store/relval/CMSSW_3_8_4/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_38Y_V12-v1/0023/462A277B-76C2-DF11-9F3F-00304867C0F6.root'
])

# Load standard services
process.load("Configuration.StandardSequences.Services_cff")
process.TFileService = cms.Service(
    "TFileService",
    fileName = cms.string("background_skim_plots.root")
)

_HLT_PATH = 'HLT_Jet15U'

# The basic HLT requirement
from HLTrigger.HLTfilters.hltHighLevel_cfi import hltHighLevel
process.trigger = hltHighLevel.clone()
process.trigger.HLTPaths = cms.vstring(_HLT_PATH)

#################################################################
# Select good data events - based on TauCommissinon sequence
# https://twiki.cern.ch/twiki/bin/view/CMS/Collisions2010Recipes
#################################################################

# Require "Physics declared"
process.load('HLTrigger.special.hltPhysicsDeclared_cfi')
process.hltPhysicsDeclared.L1GtReadoutRecordTag = 'gtDigis'

# Filter beam scraping events
process.noscraping = cms.EDFilter(
    "FilterOutScraping",
    applyfilter = cms.untracked.bool(True),
    debugOn = cms.untracked.bool(True),
    numtrack = cms.untracked.uint32(10),
    thresh = cms.untracked.double(0.25)
)

# Require all subdetectors to be on (reported by DCS)
process.load("DPGAnalysis.Skims.DetStatus_cfi")
process.dcsstatus.DetectorType = cms.vstring(
    'EBp', 'EBm', 'EEp', 'EEm',
    ##'ESp', 'ESm',
    'HBHEa', 'HBHEb', 'HBHEc',
    ##'HF', 'HO',
    'DT0', 'DTp', 'DTm', 'CSCp', 'CSCm',
    'TIBTID', 'TOB', 'TECp', 'TECm',
    'BPIX', 'FPIX'
)
process.dcsstatus.ApplyFilter = cms.bool(True)
process.dcsstatus.DebugOn = cms.untracked.bool(False)
process.dcsstatus.AndOr = cms.bool(True)

# Select only 'good' primvary vertices
process.primaryVertexFilter = cms.EDFilter(
    "GoodVertexFilter",
    vertexCollection = cms.InputTag('offlinePrimaryVertices'),
    minimumNDOF = cms.uint32(4),
    maxAbsZ = cms.double(15),
    maxd0 = cms.double(2)
)

# veto events with significant RBX/HPD noise activity
# ( see https://twiki.cern.ch/twiki/bin/view/CMS/HcalDPGAnomalousSignals )
process.load("CommonTools.RecoAlgos.HBHENoiseFilter_cfi")

process.dataQualityFilters = cms.Sequence(
    process.hltPhysicsDeclared *
    process.noscraping *
    process.dcsstatus *
    process.primaryVertexFilter *
    process.HBHENoiseFilter
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
process.GlobalTag.globaltag = autoCond['com10']

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
# Select our background tau candidates
#################################################################

# Basic selection on our
process.selectedRecoJets = cms.EDFilter(
    "CandViewRefSelector",
    src = common.jet_collection,
    cut = common.kinematic_selection,
    filter = cms.bool(True)
)

# We need at least two jets (to match to the trigger)
process.atLeastTwoRecoJets = cms.EDFilter(
    'CandViewCountFilter',
    src = cms.InputTag("selectedRecoJets"),
    minNumber = cms.uint32(2)
)

# Match our selected jets to the trigger
process.hltMatchedJets = cms.EDProducer(
    'trgMatchedCandidateProducer',
    InputProducer = cms.InputTag('selectedRecoJets'),
    hltTag = cms.untracked.InputTag(_HLT_PATH, "", "HLT"),
    triggerEventTag = cms.untracked.InputTag("hltTriggerSummaryAOD","","HLT")
)

# We need at least one of our selected jets to match HLT
process.atLeastOneHLTJet = cms.EDFilter(
    'CandViewCountFilter',
    src = cms.InputTag('hltMatchedJets'),
    minNumber = cms.uint32(1)
)

# Get all jets that are NOT matched to HLT trigger jets
process.nonHLTMatchedJets = cms.EDProducer(
    'CandViewCleaner',
    srcCands = cms.InputTag('selectedRecoJets'),
    srcObjects = cms.VInputTag(
        cms.InputTag('hltMatchedJets')
    ),
    moduleLabel = cms.string(''),
    deltaRMin = cms.double(0.1),
)

process.selectAndMatchJets = cms.Sequence(
    process.selectedRecoJets *
    process.atLeastTwoRecoJets *
    process.hltMatchedJets *
    process.atLeastOneHLTJet *
    process.nonHLTMatchedJets
)

#################################################################
# Remove trigger-biased jets
#################################################################

# Remove trigger bias from hlt matched jets
process.nonBiasedTriggerJets = cms.EDProducer(
    "CandViewRefTriggerBiasRemover",
    triggered = cms.InputTag("hltMatchedJets"),
)

process.evtcontent = cms.EDAnalyzer("EventContentAnalyzer")

# Combine the non-biased trigger jets with the (unbiased) untriggered jets.
# This a collection of Refs to what is our final output collection.
process.backgroundJetsCandRefs = cms.EDProducer(
    "CandRefMerger",
    src = cms.VInputTag(
        cms.InputTag('nonHLTMatchedJets'),
        cms.InputTag('nonBiasedTriggerJets')
    )
)

# Create a list of PFJetRefs (instead of CandidateRefs)
process.backgroundJets = cms.EDProducer(
    "PFJetRefsCastFromCandView",
    src = cms.InputTag("backgroundJetsCandRefs"),
)

# Create a clone of actual PFJets - not refs
#process.backgroundJets = cms.EDProducer(
    #"PFJetCopyProducer",
    #src = cms.InputTag("backgroundJetsPFJetRefs")
#)

process.atLeastOneBackgroundJet = cms.EDFilter(
    "CandViewCountFilter",
    src = cms.InputTag("backgroundJets"),
    minNumber = cms.uint32(1)
)

process.removeBiasedJets = cms.Sequence(
    process.nonBiasedTriggerJets *
    process.backgroundJetsCandRefs *
    process.backgroundJets*# <--- final output collection
    process.atLeastOneBackgroundJet
)


# Plot discriminants
process.plotBackgroundJets = cms.EDAnalyzer(
    "CandViewHistoAnalyzer",
    src = cms.InputTag("backgroundJets"),
    histograms = common.jet_histograms
)

# Require that at least one object in the jet can pass the lead track
# requirement
process.backgroundJetsLeadObject = cms.EDFilter(
    "JetViewRefSelector",
    src = cms.InputTag("backgroundJets"),
    cut = common.lead_object_jet_selection,
    filter = cms.bool(True),
)

process.plotBackgroundJetsLeadObject = cms.EDAnalyzer(
    "CandViewHistoAnalyzer",
    src = cms.InputTag("backgroundJetsLeadObject"),
    histograms = common.jet_histograms
)

# Turn off for now.
process.backgroundJetsCollimationCut = cms.EDFilter(
    "JetViewRefSelector",
    src = cms.InputTag("backgroundJetsLeadObject"),
    cut = cms.string("pt()*sqrt(etaetaMoment()) > -1"),
    filter = cms.bool(True),
)

process.preselectedBackgroundJets = cms.EDProducer(
    "PFJetRefsCastFromCandView",
    src = cms.InputTag("backgroundJetsLeadObject"),
)

# Build pizeros for our final jet collection
import RecoTauTag.RecoTau.RecoTauPiZeroProducer_cfi as PiZeroProd
process.backgroundJetsRecoTauPiZeros = PiZeroProd.ak5PFJetsRecoTauPiZeros.clone(
    jetSrc = cms.InputTag("preselectedBackgroundJets")
)

process.plotPreselectedBackgroundJets = cms.EDAnalyzer(
    "CandViewHistoAnalyzer",
    src = cms.InputTag("preselectedBackgroundJets"),
    histograms = common.jet_histograms
)

# Build the final preselection
process.preselectBackgroundJets = cms.Sequence(
    process.plotBackgroundJets *
    process.backgroundJetsLeadObject *
    process.plotBackgroundJetsLeadObject *
    #process.backgroundJetsCollimationCut *
    process.preselectedBackgroundJets *
    process.plotPreselectedBackgroundJets
)

# Final path
process.selectBackground = cms.Path(
    process.trigger *
    process.dataQualityFilters *
    process.rereco *
    process.selectAndMatchJets *
    process.removeBiasedJets *
    process.preselectBackgroundJets *
    process.backgroundJetsRecoTauPiZeros
)


# Keep only a subset of data
poolOutputCommands = cms.untracked.vstring(
    'drop *',
    'keep *_ak5PFJets_*_TANC',
    'keep *_offlinePrimaryVertices_*_TANC',
    'keep recoTracks_generalTracks_*_TANC',
    'keep recoTracks_electronGsfTracks_*_TANC',
    'keep recoPFCandidates_particleFlow_*_TANC',
    'keep *_preselectedBackgroundJets_*_TANC',
    'keep *_backgroundJetsRecoTauPiZeros_*_TANC',
)

# Write output
process.write = cms.OutputModule(
    "PoolOutputModule",
    fileName = cms.untracked.string("background_training.root"),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring("selectBackground"),
    ),
    outputCommands = poolOutputCommands
)
process.out = cms.EndPath(process.write)

# Print out trigger information
process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )
