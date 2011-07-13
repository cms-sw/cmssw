import FWCore.ParameterSet.Config as cms
import RecoTauTag.TauTagTools.RecoTauCommonJetSelections_cfi as common
import sys

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
filterType = None
sampleId = None
conditions=None

if not hasattr(sys, "argv"):
    raise ValueError, "Can't extract CLI arguments!"
else:
    argOffset = 0
    if sys.argv[0] != 'cmsRun':
        argOffset = 1
    # Stupid crab
    rawOptions = sys.argv[2-argOffset]
    filterType = rawOptions.split(',')[0]
    print "Using %s filter type!" % filterType
    sampleId = int(rawOptions.split(',')[1])
    print "Found %i for sample id" % sampleId
    conditions = rawOptions.split(',')[2]

print "Loading filter type"
process.load(filterType)

# Load standard services
process.load("Configuration.StandardSequences.Services_cff")
process.TFileService = cms.Service(
    "TFileService",
    fileName = cms.string(
        "background_skim_plots_%s.root" % process.filterConfig.name.value())
)

# Check if we need to modify triggers for MC running
if 'GR' not in conditions:
    print "Using MC mode"
    process.filterConfig.hltPaths = cms.vstring(
        process.filterConfig.hltPaths[0])

# Get the files specified for this filter
readFiles.extend(process.filterConfig.testFiles)

# The basic HLT requirement
from HLTrigger.HLTfilters.hltHighLevel_cfi import hltHighLevel
process.trigger = hltHighLevel.clone()
process.trigger.HLTPaths = process.filterConfig.hltPaths # <-- defined in filterType.py
process.trigger.throw = False
print process.trigger.HLTPaths

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
process.GlobalTag.globaltag = '%s::All' % conditions

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

# Match our selected jets to the trigger
process.hltMatchedJets = cms.EDProducer(
    'trgMatchedCandidateProducer',
    InputProducer = cms.InputTag('selectedRecoJets'),
    hltTags = cms.VInputTag(
        [cms.InputTag(path, "", "HLT")
         for path in process.filterConfig.hltPaths]),
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
    srcObject = cms.InputTag('selectedRecoJets'),
    srcObjectsToRemove = cms.VInputTag(
        cms.InputTag('hltMatchedJets')
    ),
    moduleLabel = cms.string(''),
    deltaRMin = cms.double(0.1),
)

process.selectAndMatchJets = cms.Sequence(
    process.selectedRecoJets *
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
# Depending on the filter type, we may or may not want to keep non-biased
# trigger matched jets.
process.backgroundJetsCandRefs = cms.EDProducer(
    "CandRefMerger",
    src = cms.VInputTag(
        cms.InputTag('nonHLTMatchedJets'),
    )
)
if process.filterConfig.useUnbiasedHLTMatchedJets:
    # <-- defined in filterType.py
    process.backgroundJetsCandRefs.src.append(
        cms.InputTag('nonBiasedTriggerJets'))

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

process.preselectedBackgroundJetRefs = cms.EDProducer(
    "PFJetRefsCastFromCandView",
    src = cms.InputTag("backgroundJetsLeadObject"),
)

process.preselectedBackgroundJets = cms.EDProducer(
    "PFJetCopyProducer",
    src = cms.InputTag("preselectedBackgroundJetRefs"),
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
    process.preselectedBackgroundJetRefs *
    process.preselectedBackgroundJets *
    process.plotPreselectedBackgroundJets
)

# Add a flag to the event to keep track of the event type
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

configtools.cloneProcessingSnippet(
    process, process.recoTauHPSTancSequence, "Background")
configtools.massSearchReplaceAnyInputTag(
    process.recoTauHPSTancSequenceBackground,
    cms.InputTag("ak5PFJets"), cms.InputTag("preselectedBackgroundJets")
)
process.buildTaus += process.recoTauHPSTancSequenceBackground

# Final path
process.selectBackground = cms.Path(
    process.trigger *
    process.dataQualityFilters *
    process.selectEnrichedEvents * # <-- defined in filterType.py
    process.rereco *
    process.selectAndMatchJets *
    process.removeBiasedJets *
    process.preselectBackgroundJets*
    process.eventSampleFlag*
    process.buildTaus
    #process.backgroundJetsRecoTauPiZeros
)

# Store the trigger stuff in the event
from PhysicsTools.PatAlgos.tools.trigTools import switchOnTrigger
switchOnTrigger(process, sequence="selectBackground", outputModule='')

# Keep only a subset of data
poolOutputCommands = cms.untracked.vstring(
    'drop *',
    'keep patTriggerObjects_*_*_TANC',
    'keep patTriggerFilters_*_*_TANC',
    'keep patTriggerPaths_*_*_TANC',
    'keep patTriggerEvent_*_*_TANC',
    'keep PileupSummaryInfo_*_*_*',
    'keep *_ak5PFJets_*_TANC',
    'keep *_offlinePrimaryVertices_*_TANC',
    'keep recoTracks_generalTracks_*_TANC',
    'keep recoTracks_electronGsfTracks_*_TANC',
    'keep recoPFCandidates_particleFlow_*_TANC',
    'keep *_preselectedBackgroundJets_*_TANC',
    'keep *_eventSampleFlag_*_*'
    #'keep *_backgroundJetsRecoTauPiZeros_*_TANC',
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
    fileName = cms.untracked.string(
        "background_training_%s.root" % process.filterConfig.name.value()),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring("selectBackground"),
    ),
    outputCommands = poolOutputCommands
)
process.out = cms.EndPath(process.write)

# Print out trigger information
process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )
