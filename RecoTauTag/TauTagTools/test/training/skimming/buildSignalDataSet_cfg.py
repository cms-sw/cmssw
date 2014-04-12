import FWCore.ParameterSet.Config as cms
import RecoTauTag.TauTagTools.RecoTauCommonJetSelections_cfi as common
import sys

'''

buildSignalDataSet_cfg

Build a reduced dataset containing gen-level infomration and reco::PFJets
that are matched to "common" hadronic tau decay modes.  Used to build the signal
training sample for the TaNC.

Author: Evan K. Friis (UC Davis)

'''

# We have to put working defaults here since multicrab doesn't respect the fact
# that we require good argv to be passed
sampleId = -999
sampleName = "ErrorParsingCLI"

if not hasattr(sys, "argv"):
    #raise ValueError, "Can't extract CLI arguments!"
    print "ERROR: Can't extract CLI arguments!"
else:
    print sys.argv
    argOffset = 0
    if sys.argv[0] != 'cmsRun':
        argOffset = 1
    args = sys.argv[2 - argOffset]
    sampleId = int(args.split(',')[1])
    sampleName = args.split(',')[0]
    print "Found %i for sample id" % sampleId
    print "Running on sample: %s" % sampleName

process = cms.Process("TANC")

maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring()
process.source = cms.Source("PoolSource", fileNames = readFiles,
                            secondaryFileNames = secFiles)

#dbs search --noheader --query="find file where primds=RelValZTT and release=$CMSSW_VERSION and tier=GEN-SIM-RECO"  | sed "s|.*|\"&\",|"
readFiles.extend([
"/store/relval/CMSSW_3_9_9/RelValZTT/GEN-SIM-RECO/START39_V9-v1/0001/22F9F1AF-9B3D-E011-BCE8-001A928116D0.root",
"/store/relval/CMSSW_3_9_9/RelValZTT/GEN-SIM-RECO/START39_V9-v1/0000/AE9602D6-593D-E011-A106-0030486790FE.root",
"/store/relval/CMSSW_3_9_9/RelValZTT/GEN-SIM-RECO/START39_V9-v1/0000/74B5926B-543D-E011-A7B9-0026189438AC.root",
"/store/relval/CMSSW_3_9_9/RelValZTT/GEN-SIM-RECO/START39_V9-v1/0000/223E536A-543D-E011-BED7-003048678B3C.root",
"/store/relval/CMSSW_3_9_9/RelValZTT/GEN-SIM-RECO/START39_V9-v1/0000/00E71A68-533D-E011-97EE-002618943934.root",
])

# Load standard services
process.load("Configuration.StandardSequences.Services_cff")
process.TFileService = cms.Service(
    "TFileService",
    fileName = cms.string("signal_skim_plots_%s.root" % sampleName)
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
                                process.ak5PFJets+
                                process.kt6PFJets)

process.kt6PFJets.doRhoFastjet = True
process.kt6PFJets.Rho_EtaMax = cms.double( 4.4)

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

process.preselectedSignalJetRefs = cms.EDProducer(
    "PFJetRefsCastFromCandView",
    src = cms.InputTag("signalJetsLeadObject"),
)
process.preselectedSignalJets = cms.EDProducer(
    "PFJetCopyProducer",
    src = cms.InputTag("preselectedSignalJetRefs")
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

process.preselectedBackgroundJetRefs = cms.EDProducer(
    "PFJetRefsCastFromCandView",
    src = cms.InputTag("backgroundJetRefs"),
)

process.preselectedBackgroundJets = cms.EDProducer(
    "PFJetCopyProducer",
    src = cms.InputTag("preselectedBackgroundJetRefs"),
)

process.backgroundJetsRecoTauPiZeros = process.signalJetsRecoTauPiZeros.clone(
    src = cms.InputTag("preselectedBackgroundJets"),
)

process.addFakeBackground = cms.Sequence(
    process.backgroundJetRefs *
    process.preselectedBackgroundJetRefs*
    process.preselectedBackgroundJets
)

# Add a flag to the event to keep track of event type
process.eventSampleFlag = cms.EDProducer(
    "RecoTauEventFlagProducer",
    flag = cms.int32(sampleId),
)

################################################################################
#  Build PF taus to be used in training
################################################################################

import PhysicsTools.PatAlgos.tools.helpers as configtools
# Build tau collections
process.load("RecoTauTag.Configuration.RecoPFTauTag_cff")
# The pileup vertex selection is common between both algorithms
process.buildTaus = cms.Sequence(process.recoTauPileUpVertices)

# Remove all the extra discriminants from the HPS tanc tau sequence
process.recoTauHPSTancSequence.remove(process.hpsTancTauDiscriminantSequence)
process.recoTauHPSTancSequence.remove(process.recoTauPileUpVertices)
process.recoTauHPSTancSequence.remove(process.pfRecoTauTagInfoProducer)
process.recoTauHPSTancSequence.remove(process.ak5PFJetTracksAssociatorAtVertex)

nModifiers = len(process.combinatoricRecoTaus.modifiers)
for iMod in range(nModifiers):
    if process.combinatoricRecoTaus.modifiers[iMod].name.value() == "TTIworkaround":
        del process.combinatoricRecoTaus.modifiers[iMod]
        break
process.hpsTancTaus.src = "combinatoricRecoTaus"

# Select taus that pass the decay mode finding
process.hpsTancTausPassingDecayMode = cms.EDFilter(
    "RecoTauDiscriminatorRefSelector",
    src = cms.InputTag("hpsTancTaus"),
    discriminator = cms.InputTag(
        "hpsTancTausDiscriminationByDecayModeSelection"),
    cut = cms.double(0.5),
    filter = cms.bool(False)
)
process.recoTauHPSTancSequence += process.hpsTancTausPassingDecayMode

# Add selectors for the different decay modes
for decayMode in [0, 1, 2, 10]:
    selectorName = "selectedHpsTancTrainTausDecayMode%i" % decayMode
    setattr(process, selectorName, cms.EDFilter(
        "PFTauViewRefSelector",
        src = cms.InputTag("hpsTancTausPassingDecayMode"),
        cut = cms.string("decayMode = %i" % decayMode),
        filter = cms.bool(False)
    ))
    process.recoTauHPSTancSequence += getattr(process, selectorName)

# Make copies of the signal and background tau production sequences
configtools.cloneProcessingSnippet(
    process, process.recoTauHPSTancSequence, "Signal")
configtools.massSearchReplaceAnyInputTag(
    process.recoTauHPSTancSequenceSignal,
    cms.InputTag("ak5PFJets"), cms.InputTag("preselectedSignalJets")
)
configtools.cloneProcessingSnippet(
    process, process.recoTauHPSTancSequence, "Background")
configtools.massSearchReplaceAnyInputTag(
    process.recoTauHPSTancSequenceBackground,
    cms.InputTag("ak5PFJets"), cms.InputTag("preselectedBackgroundJets")
)
process.buildTaus += process.recoTauHPSTancSequenceSignal
process.buildTaus += process.recoTauHPSTancSequenceBackground

################################################################################
#  Define signal path
################################################################################

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
    process.preselectedSignalJetRefs *
    process.preselectedSignalJets *
    process.plotPreselectedSignalJets *
    #process.signalJetsRecoTauPiZeros *
    process.addFakeBackground *
    process.eventSampleFlag *
    process.buildTaus
)

# Store the trigger stuff in the event
from PhysicsTools.PatAlgos.tools.trigTools import switchOnTrigger
switchOnTrigger(process, sequence="selectSignal", outputModule='')

# Keep only a subset of data
poolOutputCommands = cms.untracked.vstring(
    'drop *',
    'keep patTriggerObjects_*_*_TANC',
    'keep patTriggerFilters_*_*_TANC',
    'keep patTriggerPaths_*_*_TANC',
    'keep patTriggerEvent_*_*_TANC',
    'keep PileupSummaryInfo_*_*_*',
    'keep *_ak5PFJets_*_TANC',
    'keep *_kt6PFJets_*_TANC', # for PU subtraction
    'keep *_offlinePrimaryVertices_*_TANC',
    'keep *_offlineBeamSpot_*_TANC',
    'keep *_particleFlow_*_TANC',
    'keep recoTracks_generalTracks_*_TANC',
    'keep recoTracks_electronGsfTracks_*_TANC',
    'keep *_genParticles_*_*', # this product is okay, since we dont' need it in bkg
    'keep *_selectedTrueHadronicTaus_*_*',
    'keep *_preselectedSignalJets_*_*',
    #'keep *_signalJetsRecoTauPiZeros_*_*',
    # These two products are needed to make signal content a superset
    'keep *_preselectedBackgroundJets_*_*',
    'keep *_eventSampleFlag_*_*'
    #'keep *_backgroundJetsRecoTauPiZeros_*_*',
)

# Make a list of all the tau products (skipping combinatoric stuff to save
# space)
tau_products = [m.label() for m in configtools.listModules(process.buildTaus)
                if 'combinatoric' not in m.label()]
# Add all our tau products
for product in tau_products:
    poolOutputCommands.append('keep *_%s_*_TANC' % product)

# Write output
process.write = cms.OutputModule(
    "PoolOutputModule",
    fileName = cms.untracked.string("signal_training_%s.root" % sampleName),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring("selectSignal"),
    ),
    outputCommands = poolOutputCommands
)
process.out = cms.EndPath(process.write)

# Print out trigger information
process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )
