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


if options.tracks < 0 or options.pizeros < 0:
    print "You must specify the [tracks] and [pizeros] arguments."
    sys.exit(1)
# Make a nice tuple of the decay mode
_decay_mode = (options.tracks, options.pizeros)
decay_mode_map = {
    (1, 0): 0,
    (1, 1): 1,
    (1, 2): 2,
    (3, 0): 10
}
_decay_mode_name = decay_mode_map[_decay_mode]

process = cms.Process("Eval")
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(60000) )

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 2000

# DQM store, PDT sources etc
process.load("Configuration.StandardSequences.Services_cff")

# Setup output file
process.TFileService = cms.Service(
    "TFileService",
    fileName = cms.string(options.outputFile)
)

print "WARNING: input branch workaround!"
# Input files
process.source = cms.Source(
    "PoolSource",
    fileNames = cms.untracked.vstring(
        file for file in options.inputFiles if 'Multi' not in file),
)

# Load PiZero algorithm
process.load("RecoTauTag.Configuration.RecoPFTauTag_cff")

_KIN_CUT = "pt > 15 & abs(eta) < 2.5"
_TYPE_LABEL = options.signal==1 and "Signal" or "Background"

process.selectedBaseTaus = cms.EDFilter(
    "PFTauViewRefSelector",
    src = cms.InputTag("hpsTancTaus" + _TYPE_LABEL),
    cut = cms.string(_KIN_CUT)
)

# Select those passing the decay mode cut
process.selectedBaseDecayModeTaus = cms.EDFilter(
    "RecoTauDiscriminatorSelector",
    src = cms.InputTag("selectedBaseTaus"),
    discriminator = cms.InputTag(
        "hpsTancTausDiscriminationByDecayModeSelection" + _TYPE_LABEL),
    filter = cms.bool(False),
    cut = cms.double(0.5),
)

process.selectedDecayModeTaus = cms.EDFilter(
    "PFTauViewRefSelector",
    src = cms.InputTag(
        ("selectedHpsTancTrainTausDecayMode%i" % _decay_mode_name) + _TYPE_LABEL),
    cut = cms.string(_KIN_CUT)
)

process.main = cms.Sequence(
    process.selectedBaseTaus*
    process.selectedBaseDecayModeTaus*
    process.selectedDecayModeTaus)

# Plot the input jets to use in weighting the transformation
process.plotInputJets = cms.EDAnalyzer(
    "CandViewHistoAnalyzer",
    src = cms.InputTag("selectedBaseDecayModeTaus"),
    histograms = common.tau_histograms
)
process.main += process.plotInputJets

# Build our taus
process.load("RecoTauTag.Configuration.RecoPFTauTag_cff")

# Dont compute MVAs for ones that don't matter.  We make a discriminator that
# selects taus taht should have an MVA value computed for them.
from RecoTauTag.RecoTau.TauDiscriminatorTools import noPrediscriminants
process.selectedTausDiscriminator = cms.EDProducer(
    "PFRecoTauDiscriminationByStringCut",
    PFTauProducer = cms.InputTag("hpsTancTaus" + _TYPE_LABEL),
    cut = cms.string(_KIN_CUT + " && decayMode == %i" % _decay_mode_name),
    Prediscriminants = noPrediscriminants,
)

process.hpsTancTausDiscriminationByTancRaw.PFTauProducer = cms.InputTag(
    "hpsTancTaus" + _TYPE_LABEL)
# Make sure we first require the correct decay mode
process.hpsTancTausDiscriminationByTancRaw.Prediscriminants = cms.PSet(
    BooleanOperator = cms.string("and"),
    relevanceCut = cms.PSet(
        cut = cms.double(0.5),
        Producer = cms.InputTag("selectedTausDiscriminator")
    )
)
process.hpsTancTausDiscriminationByTancRaw.remapOutput = False


# Setup the flight path discriminators.  In the training case, these come from
# two different places
process.hpsTancTausDiscriminationByTancRaw.discriminantOptions.\
        FlightPathSignificance.discSrc = cms.VInputTag(
    "hpsTancTausDiscriminationByFlightPathSignal",
    "hpsTancTausDiscriminationByFlightPathBackground",
)

process.main += process.selectedTausDiscriminator
process.main += process.hpsTancTausDiscriminationByTancRaw

################################################################################
##         Prepare new TaNC discriminator                                    ###
################################################################################

# Setup database
process.load("RecoTauTag.TauTagTools.TancConditions_cff")
process.TauTagMVAComputerRecord.connect = cms.string(
    'sqlite:%s' % options.db)

process.TauTagMVAComputerRecord.toGet[0].tag = cms.string('Tanc')
process.TauTagMVAComputerRecord.appendToDataLabel = cms.string("hpstanc")
#mvaLabel = os.path.splitext(os.path.basename(options.db))[0]


################################################################################
##         Plot the output of the "best" Taus                                ###
################################################################################

process.cleanTauPlots = cms.EDAnalyzer(
    "RecoTauPlotDiscriminator",
    src = cms.InputTag("selectedDecayModeTaus"),
    plotPU = cms.bool(True),
    pileupInfo = cms.InputTag("addPileupInfo"),
    pileupTauPtCut = cms.double(15),
    pileupVertices = cms.InputTag("recoTauPileUpVertices"),
    discriminators = cms.VInputTag(
        cms.InputTag("hpsTancTausDiscriminationByTancRaw")
    ),
    nbins = cms.uint32(900),
    min = cms.double(-1.5),
    max = cms.double(1.5),
)

process.main += process.cleanTauPlots

process.path = cms.Path(process.main)

process.options = cms.untracked.PSet(
    IgnoreCompletely = cms.untracked.vstring("MismatchedInputFIles"),
)
