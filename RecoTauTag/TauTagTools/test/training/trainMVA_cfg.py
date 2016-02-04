'''

TaNC MVA trainer

Author: Evan K. Friis (UC Davis)

'''

import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing
import os
import sys

options = VarParsing.VarParsing ('analysis')

# Register options
options.register(
    'xml', '',
    VarParsing.VarParsing.multiplicity.singleton,
    VarParsing.VarParsing.varType.string,
    "XML file with MVA configuration")

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

options.parseArguments()

if options.tracks < 0 or options.pizeros < 0:
    print "You must specify the [tracks] and [pizeros] arguments."
    sys.exit(1)

# Make a nice tuple of the decay mode
_decay_mode = (options.tracks, options.pizeros)
# Map the XML file name to a nice computer name
_computer_name = os.path.basename(os.path.splitext(options.xml)[0])
print _computer_name

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

_KIN_CUT = 'jetRef.pt > 10 & abs(eta) < 2.5'

process = cms.Process("TrainTaNC")
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

# Input files
process.source = cms.Source(
    "PoolSource",
    fileNames = cms.untracked.vstring(options.inputFiles),
    skipBadFiles = cms.untracked.bool(True),
)

# DQM store, PDT sources etc
process.load("TrackingTools.TransientTrack.TransientTrackBuilder_cfi")
process.load("Configuration.StandardSequences.Services_cff")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
# Shit, is this okay?
from Configuration.PyReleaseValidation.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['mc']

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr = cms.untracked.PSet(placeholder = cms.untracked.bool(True))
process.MessageLogger.cout = cms.untracked.PSet(INFO = cms.untracked.PSet(
    reportEvery = cms.untracked.int32(500), # every 100th only
    limit = cms.untracked.int32(-1)       # or limit to 10 printouts...
))
process.MessageLogger.statistics.append('cout')

# Load tau algorithms
process.load("RecoTauTag.Configuration.RecoPFTauTag_cff")

#######################################################
# Database BS
#######################################################

process.PoolDBOutputService = cms.Service(
    "PoolDBOutputService",
    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
    DBParameters = cms.PSet( messageLevel = cms.untracked.int32(0) ),
    timetype = cms.untracked.string('runnumber'),
    connect = cms.string('sqlite_file:%s' % options.outputFile.replace('.root','')),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('TauTagMVAComputerRcd'),
        tag = cms.string('Train')
    ))
)

process.MVATrainerSave = cms.EDAnalyzer(
    "TauMVATrainerSave",
    toPut = cms.vstring(_computer_name),
    toCopy = cms.vstring()
)

setattr(process.MVATrainerSave, _computer_name,
        cms.string(_computer_name + ".mva"))

process.looper = cms.Looper(
    "TauMVATrainerLooper",
    trainers = cms.VPSet(cms.PSet(
        calibrationRecord = cms.string(_computer_name),
        saveState = cms.untracked.bool(True),
        trainDescription = cms.untracked.string(options.xml),
        loadState = cms.untracked.bool(False),
        doMonitoring = cms.bool(True),
    ))
)

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

###############################################################################
# Define signal and background paths Each path only gets run if the appropriate
# colleciton is present.  This allows separation of signal and background
# events.
###############################################################################

# Define selectors which detect sig/bkg events
process.signalSelectEvents= cms.EDFilter(
    "CandCollectionExistFilter",
    src = cms.InputTag("preselectedSignalJets"),
)

process.signalSequence = cms.Sequence(process.signalSelectEvents)
process.signalSequence += process.recoTauCommonSequence

decay_mode_translator = {
    (1, 0) : 'oneProng0Pi0',
    (1, 1) : 'oneProng1Pi0',
    (1, 2) : 'oneProng2Pi0',
    (3, 0) : 'threeProng0Pi0',
    (3, 1) : 'threeProng1Pi0',
}

# For signal, select true hadronic tau decays that match the desired decay mode
from RecoTauTag.TauTagTools.TauTruthProduction_cfi import trueHadronicTaus
process.selectedTrueHadronicTausMatchingDM = trueHadronicTaus.clone(
    src = cms.InputTag('selectedTrueHadronicTaus'),
    select = cms.vstring(decay_mode_translator[_decay_mode]),
    filter = cms.bool(True)
)
process.signalSequence += process.selectedTrueHadronicTausMatchingDM

# Reselect our signal jets, using only those matched to this decay mode
process.signalJetsDMTruthMatching = cms.EDProducer(
    "GenJetMatcher",
    src = cms.InputTag("preselectedSignalJets"),
    matched = cms.InputTag("selectedTrueHadronicTausMatchingDM"),
    mcPdgId     = cms.vint32(),                      # n/a
    mcStatus    = cms.vint32(),                      # n/a
    checkCharge = cms.bool(False),
    maxDeltaR   = cms.double(0.15),
    maxDPtRel   = cms.double(3.0),
    # Forbid two RECO objects to match to the same GEN object
    resolveAmbiguities    = cms.bool(True),
    resolveByMatchQuality = cms.bool(True),
)
process.signalSequence += process.signalJetsDMTruthMatching

process.signalJetsDMMatched = cms.EDFilter(
    "CandViewGenJetMatchRefSelector",
    src = cms.InputTag("preselectedSignalJets"),
    matching = cms.InputTag("signalJetsDMTruthMatching"),
    # Don't keep events with no matches
    filter = cms.bool(True)
)
process.signalSequence += process.signalJetsDMMatched

# Rebuild PiZeros for signal FIXME: remove this once skimming okay
process.signalPiZeros = process.ak5PFJetsRecoTauPiZeros.clone(
    jetSrc = cms.InputTag("signalJetsDMMatched")
)
process.signalSequence += process.signalPiZeros

# Tau production step
process.signalRawTaus = process.combinatoricRecoTaus.clone(
    jetSrc = cms.InputTag("signalJetsDMMatched"),
    piZeroSrc = cms.InputTag("signalPiZeros"),
    buildNullTaus = cms.bool(False),
    builders = cms.VPSet(_combinatoricTauConfig),
    modifiers = cms.VPSet(
        cms.PSet(
            name = cms.string("sipt"),
            plugin = cms.string("RecoTauImpactParameterSignificancePlugin"),
            qualityCuts = PFTauQualityCuts,
        )
    ),
)
process.signalSequence += process.signalRawTaus

# Apply lead pion requirement to taus
process.signalRawTausLeadPionPt = cms.EDFilter(
    "PFTauViewRefSelector",
    src = cms.InputTag("signalRawTaus"),
    cut = cms.string(
        'leadPFChargedHadrCand().muonRef().isNull() &leadPFCand().pt() > 5.0'),
    # We can skip events where no taus pass this requirement
    filter = cms.bool(True),
)
process.signalSequence += process.signalRawTausLeadPionPt

# Match these taus to the desired truth objects
process.signalTausDMTruthMatching = process.signalJetsDMTruthMatching.clone(
    src = cms.InputTag("signalRawTausLeadPionPt"),
    resolveAmbiguities = cms.bool(False),
)
process.signalSequence += process.signalTausDMTruthMatching

# Take only those with pt > 20
process.signalRawTausKinematicCut = cms.EDFilter(
    "PFTauViewRefSelector",
    src = cms.InputTag("signalRawTausLeadPionPt"),
    cut = cms.string(_KIN_CUT),
    filter = cms.bool(True),
)
process.signalSequence += process.signalRawTausKinematicCut

# Select the final collection of taus passed to the trainer
process.signalTaus = cms.EDProducer(
    "RecoTauCleaner",
    src = cms.InputTag("signalRawTausKinematicCut"),
    cleaners = cms.VPSet(
        cms.PSet(
            name = cms.string("TruthPtMatch"),
            plugin = cms.string("RecoTauDistanceFromTruthPlugin"),
            matching = cms.InputTag("signalTausDMTruthMatching"),
        ),
    )
)
process.signalSequence += process.signalTaus

process.signalPath = cms.Path(process.signalSequence)


###############################################################################
# Define background path
###############################################################################

# We don't put the background collection into signal events.  Just check if it
# exists, then we know we have a bkg event.
process.backgroundSelectEvents = cms.EDFilter(
    "CandViewCountFilter",
    src = cms.InputTag("preselectedBackgroundJets"),
    minNumber = cms.uint32(1)
)

# Rebuild PiZeros for signal FIXME: remove this once skimming okay
process.backgroundPiZeros = process.ak5PFJetsRecoTauPiZeros.clone(
    jetSrc = cms.InputTag("preselectedBackgroundJets")
)

process.backgroundRawTaus = process.signalRawTaus.clone(
    jetSrc = cms.InputTag("preselectedBackgroundJets"),
    piZeroSrc = cms.InputTag("backgroundPiZeros"),
)
process.backgroundRawTausLeadPionPt = process.signalRawTausLeadPionPt.clone(
    src = cms.InputTag("backgroundRawTaus")
)

# Take only those with pt > 20
process.backgroundRawTausKinematicCut = cms.EDFilter(
    "PFTauViewRefSelector",
    src = cms.InputTag("backgroundRawTausLeadPionPt"),
    cut = cms.string(_KIN_CUT),
    filter = cms.bool(True),
)

# Select (randomly) only one tau for each jet
#process.backgroundTaus = cms.EDProducer(
    #"RecoTauCleaner",
    #src = cms.InputTag("backgroundRawTausLeadPionPt"),
    #cleaners = cms.VPSet(
        #cms.PSet(
            #name = cms.string("RandomSelection"),
            #plugin = cms.string("RecoTauRandomCleanerPlugin"),
        #),
    #)
#)

process.backgroundSequence = cms.Sequence(
    process.backgroundSelectEvents *
    process.recoTauCommonSequence *
    process.backgroundPiZeros *
    process.backgroundRawTaus *
    process.backgroundRawTausLeadPionPt *
    process.backgroundRawTausKinematicCut
    #* process.backgroundTaus
)
process.backgroundPath = cms.Path(process.backgroundSequence)

# Finally, pass our selected sig/bkg taus to the MVA trainer
from RecoTauTag.RecoTau.RecoTauDiscriminantConfiguration import \
        discriminantConfiguration
process.trainer = cms.EDAnalyzer(
    "RecoTauMVATrainer",
    signalSrc = cms.InputTag("signalTaus"),
    backgroundSrc = cms.InputTag("backgroundRawTausKinematicCut"),
    computerName = cms.string(_computer_name),
    dbLabel = cms.string("trainer"),
    discriminantOptions = discriminantConfiguration
)

process.trainPath = cms.Path(
    process.recoTauCommonSequence*
    process.trainer)

process.outpath = cms.EndPath(process.MVATrainerSave)

process.schedule = cms.Schedule(
    process.signalPath,
    process.backgroundPath,
    process.trainPath,
    process.outpath
)

process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )
