#!/usr/bin/env cmsRun
'''

evaluateMode_cfg

Run the combinatoric tau production & TaNC cleaner selection for a given tau
decay mode on the validation sample.  Produces output histograms containing the
TaNC output for *clean* signal and background taus.

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
    'tracks', -1,
    VarParsing.VarParsing.multiplicity.singleton,
    VarParsing.VarParsing.varType.int,
    "Number of tracks in decay mode")

options.register(
    'pizeros', -1,
    VarParsing.VarParsing.multiplicity.singleton,
    VarParsing.VarParsing.varType.int,
    "Number of pi zeros in decay mode")

options.register(
    'signal', 0,
    VarParsing.VarParsing.multiplicity.singleton,
    VarParsing.VarParsing.varType.int,
    "If signal=1, only jets matched to gen-level taus will be used"
)

options.parseArguments()

_SIGNAL = False
if options.signal == 1:
    _SIGNAL = True

if options.tracks < 0 or options.pizeros < 0:
    print "You must specify the [tracks] and [pizeros] arguments."
    sys.exit(1)
# Make a nice tuple of the decay mode
_decay_mode = (options.tracks, options.pizeros)

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

# Build a combinatoric tau producer that only builds the desired decay modes
from RecoTauTag.RecoTau.RecoTauCombinatoricProducer_cfi \
        import _combinatoricTauConfig

selectedDecayModes = cms.VPSet()
for decayMode in _combinatoricTauConfig.decayModes:
    decay_mode_tuple = (decayMode.nCharged.value(), decayMode.nPiZeros.value())
    if decay_mode_tuple == _decay_mode:
        print "Decay mode:", decay_mode_tuple
        print " tau prod. max tracks:", decayMode.maxTracks
        print " tau prod. max pizeros:", decayMode.maxPiZeros
        selectedDecayModes.append(decayMode)
        break

# Only build the selected decay modes
_combinatoricTauConfig.decayModes = selectedDecayModes

process = cms.Process("Eval")
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(5000) )

# DQM store, PDT sources etc
process.load("TrackingTools.TransientTrack.TransientTrackBuilder_cfi")
process.load("Configuration.StandardSequences.Services_cff")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
# Shit, is this okay?
from Configuration.PyReleaseValidation.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['mc']


# Setup output file
process.TFileService = cms.Service(
    "TFileService",
    fileName = cms.string(options.outputFile)
)

# Input files
process.source = cms.Source(
    "PoolSource",
    fileNames = cms.untracked.vstring(options.inputFiles)
)

# Load PiZero algorithm
process.load("RecoTauTag.RecoTau.RecoTauPiZeroProducer_cfi")

decay_mode_translator = {
    (1, 0) : 'oneProng0Pi0',
    (1, 1) : 'oneProng1Pi0',
    (1, 2) : 'oneProng2Pi0',
    (3, 0) : 'threeProng0Pi0',
    (3, 1) : 'threeProng1Pi0',
}

_KIN_CUT = "pt > 10 & abs(eta) < 2.5"

# For signal, select jets that match a hadronic decaymode
process.kinematicSignalJets = cms.EDFilter(
    "CandViewRefSelector",
    src = cms.InputTag("preselectedSignalJets"),
    cut = cms.string(_KIN_CUT),
)
process.signalSpecific = cms.Sequence(process.kinematicSignalJets)

# Reselect our signal jets, using only those matched to this decay mode
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

process.inputJets = cms.EDFilter(
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

# Figure out how to deal with our events - are they signal or background?
if _SIGNAL:
    process.specific = process.signalSpecific
else:
    process.specific = process.backgroundSpecific
    process.inputJets = cms.EDFilter(
        "CandViewRefSelector",
        src = cms.InputTag("kinematicBackgroundJets"),
        cut = cms.string("pt > 0"), # take everything
        filter = cms.bool(False)
    )

# Rebuild PiZeros for signal FIXME: remove this once skimming okay
process.inputPiZeros = process.ak5PFJetsRecoTauPiZeros.clone(
    jetSrc = cms.InputTag("inputJets")
)
process.main = cms.Sequence(process.specific*process.inputJets*
                            process.inputPiZeros)

# Plot the input jets to use in weighting the transformation
process.plotInputJets = cms.EDAnalyzer(
    "CandViewHistoAnalyzer",
    src = cms.InputTag("inputJets"),
    histograms = common.jet_histograms
)
process.main += process.plotInputJets

# Build our base combinatoric taus
process.rawTaus = cms.EDProducer(
    "RecoTauProducer",
    jetSrc = cms.InputTag("inputJets"),
    piZeroSrc = cms.InputTag("inputPiZeros"),
    builders = cms.VPSet(_combinatoricTauConfig),
    modifiers = cms.VPSet()
)
process.main += process.rawTaus

# Apply lead pion requirement to taus
process.rawTausLeadPionPt = cms.EDFilter(
    "PFTauViewRefSelector",
    src = cms.InputTag("rawTaus"),
    cut = cms.string('leadPFCand().pt() > 5.0'),
    # We can skip events where no taus pass this requirement
    filter = cms.bool(True),
)
process.main += process.rawTausLeadPionPt

################################################################################
##         Prepare new TaNC discriminator                                    ###
################################################################################

# Setup database
process.load("RecoTauTag.TauTagTools.TancConditions_cff")
process.TauTagMVAComputerRecord.connect = cms.string(
    'sqlite:%s' % options.db)

process.es_prefer_tanc = cms.ESPrefer("PoolDBESSource",
                                      "TauTagMVAComputerRecord")

process.TauTagMVAComputerRecord.toGet[0].tag = cms.string('Train')
mvaLabel = os.path.splitext(os.path.basename(options.db))[0]

from RecoTauTag.RecoTau.TauDiscriminatorTools import noPrediscriminants
process.discriminate = cms.EDProducer(
    "RecoTauMVADiscriminator",
    PFTauProducer = cms.InputTag("rawTaus"),
    Prediscriminants = noPrediscriminants,
    dbLabel = cms.string(""),
    remapOutput = cms.bool(True),
    mvas = cms.VPSet(
        cms.PSet(
            nCharged = cms.uint32(options.tracks),
            nPiZeros = cms.uint32(options.pizeros),
            mvaLabel = cms.string(mvaLabel),
            #mvaLabel = cms.string(
            #    "%iprong%ipi0" % (options.tracks, options.pizeros))
        )
    ),
    prefailValue = cms.double(-2.0),
)
process.main += process.discriminate

process.rawTausLeadPionPt = cms.EDFilter(
    "PFTauViewRefSelector",
    src = cms.InputTag("rawTaus"),
    cut = cms.string('leadPFCand().pt() > 5.0'),
    # We can skip events where no taus pass this requirement
    filter = cms.bool(True),
)
process.main += process.rawTausLeadPionPt

# Select the final collection of taus.  For each PFJet, only the "best"
# tau according TaNC is selected.
process.taus = cms.EDProducer(
    "RecoTauCleaner",
    src = cms.InputTag("rawTausLeadPionPt"),
    cleaners = cms.VPSet(
        cms.PSet(
            name = cms.string("TaNCCleaner"),
            # FIXME
            plugin = cms.string("RecoTauDiscriminantCleanerPlugin"),
            src = cms.InputTag("discriminate"),
        ),
    )
)
process.main += process.taus

# We need to re-run the TaNC to re key it :(
process.discriminationByTaNC = process.discriminate.clone(
    PFTauProducer = cms.InputTag("taus")
)
process.main += process.discriminationByTaNC

################################################################################
##         Plot the output of the "best" Taus                                ###
################################################################################

process.cleanTauPlots = cms.EDAnalyzer(
    "RecoTauPlotDiscriminator",
    src = cms.InputTag("taus"),
    discriminators = cms.VInputTag(
        cms.InputTag("discriminationByTaNC")
    ),
    nbins = cms.uint32(900),
    min = cms.double(-1),
    max = cms.double(2),
)

process.main += process.cleanTauPlots

process.dirtyTauPlots = process.cleanTauPlots.clone(
    src = cms.InputTag("rawTausLeadPionPt"),
    discriminators = cms.VInputTag(cms.InputTag("discriminate"))
)
process.main += process.dirtyTauPlots

process.path = cms.Path(process.main)
