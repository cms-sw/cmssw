#!/usr/bin/env cmsRun
'''

evaluate_cfg

Compute the MVA training's performance on the validation sample and compare it
to the other default algorithms.

Author: Evan K. Friis (UC Davis)

'''

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

# We need to turn off the track based quality cuts, since we don't save them
# in the skim.
#from RecoTauTag.RecoTau.PFRecoTauQualityCuts_cfi import PFTauQualityCuts
#PFTauQualityCuts.signalQualityCuts = cms.PSet(
    #minTrackPt = PFTauQualityCuts.signalQualityCuts.minTrackPt,
    #minGammaEt = PFTauQualityCuts.signalQualityCuts.minGammaEt,
#)
#PFTauQualityCuts.isolationQualityCuts = cms.PSet(
    #minTrackPt = PFTauQualityCuts.isolationQualityCuts.minTrackPt,
    #minGammaEt = PFTauQualityCuts.isolationQualityCuts.minGammaEt,
#)

process = cms.Process("Eval")
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(5000) )

# The tanc needs the global tag
from Configuration.PyReleaseValidation.autoCond import autoCond
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = autoCond[_SIGNAL and 'mc' or 'com10']


# Setup output file for the plots
process.TFileService = cms.Service(
    "TFileService",
    fileName = cms.string(options.outputFile)
)

# Input files
process.source = cms.Source(
    "PoolSource",
    fileNames = cms.untracked.vstring(options.inputFiles)
)

_KIN_CUT = "pt > 20 & abs(eta) < 2.5"

# For signal, select jets that match a hadronic decaymode
process.kinematicSignalJets = cms.EDFilter(
    "CandViewRefSelector",
    src = cms.InputTag("ak5PFJets"),
    cut = cms.string(_KIN_CUT),
)
process.signalSpecific = cms.Sequence(process.kinematicSignalJets)

# Reselect our signal jets, using only those matched to true taus
process.signalJetsDMTruthMatching = cms.EDProducer(
    "GenJetMatcher",
    src = cms.InputTag("kinematicSignalJets"),
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
process.signalSpecific += process.signalJetsDMTruthMatching

# Overwrite the ak5PFJets with our special collection.  The regular tau
# sequences will use this as input.
process.ak5PFJets = cms.EDFilter(
    "CandViewGenJetMatchRefSelector",
    src = cms.InputTag("kinematicSignalJets"),
    matching = cms.InputTag("signalJetsDMTruthMatching"),
    # Don't keep events with no matches
    filter = cms.bool(True)
)

process.kinematicBackgroundJets = cms.EDFilter(
    "CandViewRefSelector",
    src = cms.InputTag("ak5PFJets"),
    cut = cms.string(_KIN_CUT)
)
process.backgroundSpecific = cms.Sequence(process.kinematicBackgroundJets)

# Load the tau sequence
process.load("RecoTauTag.Configuration.RecoPFTauTag_cff")
# Remove the lepton discrimination
print "Removing lepton discriminants"
print process.PFTau.remove(process.shrinkingConePFTauDiscriminationAgainstElectron)
print process.PFTau.remove(process.shrinkingConePFTauDiscriminationAgainstMuon)
print process.PFTau.remove(process.hpsPFTauDiscriminationAgainstElectron)
print process.PFTau.remove(process.hpsPFTauDiscriminationAgainstMuon)

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
    process.ak5PFJets = cms.EDFilter(
        "CandViewRefSelector",
        src = cms.InputTag("kinematicBackgroundJets"),
        cut = cms.string(_KIN_CUT), # take everything
        filter = cms.bool(False)
    )

# For signal, make some plots of the matching information
process.plotShrinkingRes = cms.EDAnalyzer(
    "CandViewHistoAnalyzer",
    src = cms.InputTag("shrinkingConePFTauProducer"),
    histograms = common.tau_histograms
)

process.plotHPSRes = process.plotShrinkingRes.clone(
    src = cms.InputTag("hpsPFTauProducer")
)

process.plotHPSTancRes = process.plotShrinkingRes.clone(
    src = cms.InputTag("hpsTancTaus")
)

process.main = cms.Sequence(process.specific*process.ak5PFJets*process.PFTau)
#process.main = cms.Sequence(process.specific*process.ak5PFJets*process.ak5PFJetsRecoTauPiZeros*process.combinatoricRecoTaus*process.hpsTancTauSequence)

if _SIGNAL:
    process.main += process.plotShrinkingRes
    process.main += process.plotHPSRes
    process.main += process.plotHPSTancRes

# Plot the input jets to use in weighting the transformation
process.plotAK5PFJets = cms.EDAnalyzer(
    "CandViewHistoAnalyzer",
    src = cms.InputTag("ak5PFJets"),
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
    'shrinkingConePFTauDiscriminationByLeadingPionPtCut',
    'shrinkingConePFTauDiscriminationByIsolation',
    'shrinkingConePFTauDiscriminationByTrackIsolation',
    'shrinkingConePFTauDiscriminationByECALIsolation',
    'shrinkingConePFTauDiscriminationByTaNC',
    'shrinkingConePFTauDiscriminationByTaNCfrOnePercent',
    'shrinkingConePFTauDiscriminationByTaNCfrHalfPercent',
    'shrinkingConePFTauDiscriminationByTaNCfrQuarterPercent',
    'shrinkingConePFTauDiscriminationByTaNCfrTenthPercent'
]

discriminators['hpsTancTaus'] = [
    'hpsTancTausDiscriminationByTanc'
    #'hpsTancTausDiscriminationByTancRaw'
]

process.plothpsTancTaus = cms.EDAnalyzer(
    "RecoTauPlotDiscriminator",
    src = cms.InputTag("hpsTancTaus"),
    discriminators = cms.VInputTag(
        cms.InputTag("hpsTancTausDiscriminationByTanc"),
        cms.InputTag("hpsTancTausDiscriminationByTancRaw"),
    ),
    nbins = cms.uint32(300),
    min = cms.double(-1),
    max = cms.double(2),
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
    process.plothpsTancTaus *
    process.plotshrinkingConePFTauProducer *
    process.plothpsPFTauProducer
)

process.main += process.plots

process.path = cms.Path(process.main)
