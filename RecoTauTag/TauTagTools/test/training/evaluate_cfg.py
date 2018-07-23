#!/usr/bin/env cmsRun
'''

evaluate_cfg

Compute the MVA training's performance on the validation sample and compare it
to the other default algorithms.

Author: Evan K. Friis (UC Davis)

'''
from __future__ import print_function

import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing
import sys
import os
import RecoTauTag.TauTagTools.RecoTauCommonJetSelections_cfi as common

options = VarParsing.VarParsing ('analysis')

options.register(
    'db', '',
    VarParsing.VarParsing.multiplicity.singleton,
    VarParsing.VarParsing.varType.string,
    "Database with trained MVA"
)

options.register(
    'signal', 0,
    VarParsing.VarParsing.multiplicity.singleton,
    VarParsing.VarParsing.varType.int,
    "If signal=1, only jets matched to gen-level taus will be used"
)

options.register(
    'transform', '',
    VarParsing.VarParsing.multiplicity.singleton,
    VarParsing.VarParsing.varType.string,
    "Python file containing TaNC transform"
)

options.parseArguments()

_SIGNAL = False
if options.signal == 1:
    _SIGNAL = True

process = cms.Process("Eval")
#process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(20000) )
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 2000
#process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(5000) )

# DQM store, PDT sources etc
process.load("TrackingTools.TransientTrack.TransientTrackBuilder_cfi")
process.load("Configuration.StandardSequences.Services_cff")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
from Configuration.PyReleaseValidation.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['mc']

# Setup output file for the plots
process.TFileService = cms.Service(
    "TFileService",
    fileName = cms.string(options.outputFile)
)

# Input files
process.source = cms.Source(
    "PoolSource",
    fileNames = cms.untracked.vstring(options.inputFiles),
    # Don't use any of the same events used to evaluate the modes
    #skipEvents = cms.untracked.uint32(20000)
)

_KIN_CUT = "pt > 5 & abs(eta) < 2.5"

# For signal, select jets that match a hadronic decaymode
process.kinematicSignalJets = cms.EDFilter(
    "CandViewRefSelector",
    src = cms.InputTag("preselectedSignalJets"),
    cut = cms.string(_KIN_CUT),
)
process.signalSpecific = cms.Sequence(process.kinematicSignalJets)

process.kinematicTrueTaus = cms.EDFilter(
    "GenJetSelector",
    src = cms.InputTag("selectedTrueHadronicTaus"),
    cut = cms.string(_KIN_CUT),
)
process.signalSpecific += process.kinematicTrueTaus

# Reselect our signal jets, using only those matched to true taus
process.signalJetsDMTruthMatching = cms.EDProducer(
    "GenJetMatcher",
    src = cms.InputTag("kinematicSignalJets"),
    matched = cms.InputTag("kinematicTrueTaus"),
    mcPdgId     = cms.vint32(),                      # n/a
    mcStatus    = cms.vint32(),                      # n/a
    checkCharge = cms.bool(False),
    maxDeltaR   = cms.double(0.15),
    maxDPtRel   = cms.double(3.0),
    # Forbid two RECO objects to match to the same GEN object
    resolveAmbiguities    = cms.bool(True),
    resolveByMatchQuality = cms.bool(True),
)
process.signalSpecific += process.signalJetsDMTruthMatching

# Overwrite the ak4PFJets with our special collection.  The regular tau
# sequences will use this as input.
process.ak4PFJets = cms.EDFilter(
    "CandViewGenJetMatchRefSelector",
    src = cms.InputTag("kinematicSignalJets"),
    matching = cms.InputTag("signalJetsDMTruthMatching"),
    # Don't keep events with no matches
    filter = cms.bool(True)
)

process.kinematicBackgroundJets = cms.EDFilter(
    "CandViewRefSelector",
    src = cms.InputTag("preselectedBackgroundJets"),
    cut = cms.string(_KIN_CUT)
)
process.backgroundSpecific = cms.Sequence(process.kinematicBackgroundJets)

# Load the tau sequence
process.load("RecoTauTag.Configuration.RecoPFTauTag_cff")
# Remove the lepton discrimination
lepton_discriminants = [name for name in process.PFTau.moduleNames()
                        if 'Muon' in name or 'Electron' in name]
print("Removing lepton discriminants: %s" % " ".join(lepton_discriminants))
for lepton_discriminant in lepton_discriminants:
    module = getattr(process, lepton_discriminant)
    process.PFTau.remove(module)

# RecoTau modifier that takes a PFJet -> tau GenJet matching and embed the true
# four vector and decay mode in unused PFTau variables
jetMatchModifier = cms.PSet(
        jetTruthMatch = cms.InputTag("signalJetsDMTruthMatching"),
        name = cms.string('embed'),
        plugin = cms.string('RecoTauTruthEmbedder')
)
# Figure out how to deal with our events - are they signal or background?
if _SIGNAL:
    process.specific = process.signalSpecific
    # Embed the truth into our taus
    process.shrinkingConePFTauProducer.modifiers.append(jetMatchModifier)
    process.combinatoricRecoTaus.modifiers.append(jetMatchModifier)
else:
    process.specific = process.backgroundSpecific
    process.ak4PFJets = cms.EDFilter(
        "CandViewRefSelector",
        src = cms.InputTag("kinematicBackgroundJets"),
        cut = cms.string(_KIN_CUT),
        filter = cms.bool(False)
    )

# For signal, make some plots of the matching information
process.mediumShrinkingTaus = cms.EDFilter(
    "RecoTauDiscriminatorRefSelector",
    src = cms.InputTag("shrinkingConePFTauProducer"),
    discriminator = cms.InputTag("shrinkingConePFTauDiscriminationByTaNCfrHalfPercent"),
    cut = cms.double(0.5),
    filter = cms.bool(False)
)
process.plotShrinkingRes = cms.EDAnalyzer(
    "CandViewHistoAnalyzer",
    src = cms.InputTag("mediumShrinkingTaus"),
    histograms = common.tau_histograms
)

process.mediumHPSTaus = cms.EDFilter(
    "RecoTauDiscriminatorRefSelector",
    src = cms.InputTag("hpsPFTauProducer"),
    discriminator = cms.InputTag("hpsPFTauDiscriminationByLooseIsolation"),
    cut = cms.double(0.5),
    filter = cms.bool(False)
)
process.plotHPSRes = process.plotShrinkingRes.clone(
    src = cms.InputTag("mediumHPSTaus")
)

process.mediumHPSTancTaus = cms.EDFilter(
    "RecoTauDiscriminatorRefSelector",
    src = cms.InputTag("hpsTancTaus"),
    discriminator = cms.InputTag("hpsTancTausDiscriminationByTancMedium"),
    cut = cms.double(0.5),
    filter = cms.bool(False)
)
process.plotHPSTancRes = process.plotShrinkingRes.clone(
    src = cms.InputTag("mediumHPSTancTaus")
)

process.main = cms.Sequence(process.specific*process.ak4PFJets*process.PFTau)
#process.main = cms.Sequence(process.specific*process.ak4PFJets*process.ak4PFJetsRecoTauPiZeros*process.combinatoricRecoTaus*process.hpsTancTauSequence)

if _SIGNAL:
    process.main += process.mediumShrinkingTaus
    process.main += process.plotShrinkingRes
    process.main += process.mediumHPSTaus
    process.main += process.plotHPSRes
    process.main += process.mediumHPSTancTaus
    process.main += process.plotHPSTancRes

################################################################################
##         Rekey flight path discriminators from skimming                    ###
################################################################################
# We don't save the vertex, beamspot stuff, so we need to rekey our
# discriminators for the flight path significance from the ones we made in the
# skim
process.load("RecoTauTag.TauTagTools.PFTauMatching_cfi")
process.matchForRekeying = process.pfTauMatcher.clone(
    src = cms.InputTag("hpsTancTaus"),
    matched = cms.InputTag(
        _SIGNAL and "hpsTancTausSignal" or "hpsTancTausBackground"),
    resolveAmbiguities = cms.bool(True),
    resolveByMatchQuality = cms.bool(True),
)

from RecoTauTag.RecoTau.TauDiscriminatorTools import noPrediscriminants

process.hpsTancTausDiscriminationByFlightPathRekey = cms.EDProducer(
    "RecoTauRekeyDiscriminator",
    PFTauProducer = cms.InputTag("hpsTancTaus"),
    Prediscriminants = noPrediscriminants,
    otherDiscriminator = cms.InputTag(
        _SIGNAL and "hpsTancTausDiscriminationByFlightPathSignal"
        or "hpsTancTausDiscriminationByFlightPathBackground"),
    matching = cms.InputTag("matchForRekeying"),
    verbose = cms.untracked.bool(False)
)
process.rekeyFlight = cms.Sequence(
    process.matchForRekeying*process.hpsTancTausDiscriminationByFlightPathRekey)

process.PFTau.replace(
    process.hpsTancTausDiscriminationByFlightPath,
    process.rekeyFlight
)
process.hpsTancTausDiscriminationByTancRaw.discriminantOptions.\
        FlightPathSignificance.discSrc = cms.InputTag(
            "hpsTancTausDiscriminationByFlightPathRekey"
        )
process.hpsTancTausDiscriminationByTancRaw.remapOutput = False

# Plot the input jets to use in weighting the transformation
process.plotAK5PFJets = cms.EDAnalyzer(
    "CandViewHistoAnalyzer",
    src = cms.InputTag("ak4PFJets"),
    histograms = common.jet_histograms
)
process.main += process.plotAK5PFJets

################################################################################
##         Prepare new TaNC discriminator                                    ###
################################################################################

# Setup local database access
process.load("RecoTauTag.TauTagTools.TancConditions_cff")
process.TauTagMVAComputerRecord.connect = cms.string(
    'sqlite:%s' % options.db)
process.TauTagMVAComputerRecord.toGet[0].tag = "Tanc"
# Don't conflict with GlobalTag (which provided shrinking cone tanc)
process.TauTagMVAComputerRecord.appendToDataLabel = cms.string("train_test")
process.combinatoricRecoTausDiscriminationByTanc.dbLabel = "train_test"
process.hpsTancTausDiscriminationByTancRaw.dbLabel = "train_test"

# Load our custom transform
transform_dir = os.path.dirname(options.transform)
sys.path.append(transform_dir)
import transforms as custom_transform
# Set the TaNC transformed to use the input transform
process.load("RecoTauTag.Configuration.HPSTancTaus_cfi")
process.hpsTancTausDiscriminationByTanc.transforms = custom_transform.transforms
process.combinatoricRecoTausTancTransform.transforms = \
        custom_transform.transforms
process.hpsTancTausDiscriminationByTancLoose.Prediscriminants.tancCut.cut = \
        custom_transform.cuts.looseCut
process.hpsTancTausDiscriminationByTancMedium.Prediscriminants.tancCut.cut = \
        custom_transform.cuts.mediumCut
process.hpsTancTausDiscriminationByTancTight.Prediscriminants.tancCut.cut = \
        custom_transform.cuts.tightCut

################################################################################
##         Plot the output of each tau algorithm                             ###
################################################################################

discriminators = {}
discriminators['hpsPFTauProducer'] = [
    'hpsPFTauDiscriminationByDecayModeFinding',
    'hpsPFTauDiscriminationByLooseIsolation',
    'hpsPFTauDiscriminationByMediumIsolation',
    'hpsPFTauDiscriminationByTightIsolation',
]

discriminators['shrinkingConePFTauProducer'] = [
    #'shrinkingConePFTauDiscriminationByLeadingPionPtCut',
    #'shrinkingConePFTauDiscriminationByIsolation',
    #'shrinkingConePFTauDiscriminationByTrackIsolation',
    #'shrinkingConePFTauDiscriminationByECALIsolation',
    'shrinkingConePFTauDiscriminationByTaNC',
    #'shrinkingConePFTauDiscriminationByTaNCfrOnePercent',
    'shrinkingConePFTauDiscriminationByTaNCfrHalfPercent',
    #'shrinkingConePFTauDiscriminationByTaNCfrQuarterPercent',
    #'shrinkingConePFTauDiscriminationByTaNCfrTenthPercent'
]

discriminators['hpsTancTaus'] = [
    'hpsTancTausDiscriminationByTanc',
    'hpsTancTausDiscriminationByTancLoose',
    'hpsTancTausDiscriminationByTancMedium',
    'hpsTancTausDiscriminationByTancTight',
    'hpsTancTausDiscriminationByTancRaw'
]

pileup_cut = 'pt > 15 & jetRef().pt > 20'
if _SIGNAL:
    pileup_cut = 'pt > 15 & alternatLorentzVect().pt > 15'
process.plothpsTancTaus = cms.EDAnalyzer(
    "RecoTauPlotDiscriminator",
    src = cms.InputTag("hpsTancTaus"),
    plotPU = cms.bool(True),
    pileupInfo = cms.InputTag("addPileupInfo"),
    pileupVertices = cms.InputTag("recoTauPileUpVertices"),
    pileupTauPtCut = cms.double(15),
    pileupPlotCut = cms.string(pileup_cut),
    discriminators = cms.VInputTag(
        discriminators['hpsTancTaus']
    ),
    nbins = cms.uint32(580),
    min = cms.double(-1.2),
    max = cms.double(1.2),
)

process.plotshrinkingConePFTauProducer = process.plothpsTancTaus.clone(
    src = cms.InputTag("shrinkingConePFTauProducer"),
    discriminators = cms.VInputTag(
        discriminators['shrinkingConePFTauProducer']
    )
)

process.plothpsPFTauProducer = process.plothpsTancTaus.clone(
    src = cms.InputTag("hpsPFTauProducer"),
    discriminators = cms.VInputTag(
        discriminators['hpsPFTauProducer']
    )
)

process.plots = cms.Sequence(
    process.mediumHPSTaus *
    process.plothpsTancTaus *
    process.plotshrinkingConePFTauProducer *
    process.plothpsPFTauProducer
)

process.main += process.plots

process.path = cms.Path(process.main)
process.schedule = cms.Schedule(
    process.path
)
process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )
