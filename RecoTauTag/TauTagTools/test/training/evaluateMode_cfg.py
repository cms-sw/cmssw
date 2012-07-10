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
#_combinatoricTauConfig.decayModes = selectedDecayModes

process = cms.Process("Eval")
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10000) )

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
process.load("RecoTauTag.Configuration.RecoPFTauTag_cff")

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

process.inputJetRefs = cms.EDFilter(
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
    process.inputJetRefs = cms.EDFilter(
        "CandViewRefSelector",
        src = cms.InputTag("kinematicBackgroundJets"),
        cut = cms.string("pt > 0"), # take everything
        filter = cms.bool(False)
    )

# Convert to PFJetRefs, instead of candidate
process.castedInputJets = cms.EDProducer(
    "PFJetRefsCastFromCandView",
    src = cms.InputTag("inputJetRefs"),
)

# Cast the jet refs to real PFJets
process.inputJets = cms.EDProducer(
    "PFJetCopyProducer",
    src = cms.InputTag("castedInputJets"),
)

process.main = cms.Sequence(
    process.specific*process.inputJetRefs*
    process.castedInputJets*process.inputJets)

# Plot the input jets to use in weighting the transformation
process.plotInputJets = cms.EDAnalyzer(
    "CandViewHistoAnalyzer",
    src = cms.InputTag("inputJets"),
    histograms = common.jet_histograms
)
process.main += process.plotInputJets

# Build our taus
process.load("RecoTauTag.Configuration.RecoPFTauTag_cff")
process.recoTauAK5PFJets08Region.src = cms.InputTag("inputJets")
process.ak5PFJetsRecoTauPiZeros.jetSrc = "inputJets"
process.combinatoricRecoTaus.jetSrc = "inputJets"
process.combinatoricRecoTaus.piZeroSrc = cms.InputTag("ak5PFJetsRecoTauPiZeros")
process.combinatoricRecoTaus.jetRegionSrc = cms.InputTag("recoTauAK5PFJets08Region")
process.combinatoricRecoTaus.outputSelection = cms.string(
        'mass() < 3 & isolationPFChargedHadrCandsPtSum() < 7')

#process.combinatoricRecoTaus.builders = cms.VPSet(_combinatoricTauConfig)
# Add PFTauTagInfo workaround
process.main += process.ak5PFJetsRecoTauPiZeros
process.main += process.recoTauCommonSequence
process.main += process.combinatoricRecoTaus
process.main += process.hpsTancTauSequence
print process.hpsTancTauSequence.remove(
    process.hpsTancTausDiscriminationAgainstElectron)
print process.hpsTancTauSequence.remove(
    process.hpsTancTausDiscriminationAgainstMuon)

################################################################################
##         Prepare new TaNC discriminator                                    ###
################################################################################

# Setup database
process.load("RecoTauTag.TauTagTools.TancConditions_cff")
process.TauTagMVAComputerRecord.connect = cms.string(
    'sqlite:%s' % options.db)

#process.es_prefer_tanc = cms.ESPrefer("PoolDBESSource",
#                                      "TauTagMVAComputerRecord")
#

process.TauTagMVAComputerRecord.toGet[0].tag = cms.string('Tanc')
process.TauTagMVAComputerRecord.appendToDataLabel = cms.string("hpstanc")
#mvaLabel = os.path.splitext(os.path.basename(options.db))[0]


################################################################################
##         Plot the output of the "best" Taus                                ###
################################################################################

# Only plot those that match this decay mode
process.matchingDecayMode = cms.EDFilter(
    "PFTauViewRefSelector",
    src = cms.InputTag("hpsTancTaus"),
    cut = cms.string("signalPFChargedHadrCands().size() = %i &"
                     "signalPiZeroCandidates().size() = %i" %
                     (options.tracks, options.pizeros)),
    filter = cms.bool(False),
)
process.main += process.matchingDecayMode

process.cleanTauPlots = cms.EDAnalyzer(
    "RecoTauPlotDiscriminator",
    src = cms.InputTag("matchingDecayMode"),
    discriminators = cms.VInputTag(
        cms.InputTag("hpsTancTausDiscriminationByTancRaw")
    ),
    nbins = cms.uint32(900),
    min = cms.double(-1),
    max = cms.double(2),
)

process.main += process.cleanTauPlots

process.dirtyTauPlots = process.cleanTauPlots.clone(
    src = cms.InputTag("combinatoricRecoTaus"),
    discriminators = cms.VInputTag(
        cms.InputTag("combinatoricRecoTausDiscriminationByTanc"))
)
process.main += process.dirtyTauPlots

process.path = cms.Path(process.main)
