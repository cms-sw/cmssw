# Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>

# Runs the C++ preset examples (TruthGraphPresetExamples.cc) over a file that holds the
# logical graph:
#
#   cmsRun presetExamples_cfg.py step3.root --preset top
#   cmsRun presetExamples_cfg.py step3.root --preset all -n 5
#
# A production file holds the graph with no selection preset, so the signal level is
# empty there. --rebuild builds the graph again from the GEN and SIM record in the file,
# with the preset's selection, which is what every example except full starts from. A
# gun takes its species from the generator fragment, so give --fragment for a gun sample:
#
#   cmsRun presetExamples_cfg.py step3.root --preset gun --rebuild --fragment TenTau_E_15_500

import FWCore.ParameterSet.Config as cms
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("inputFile", nargs="?", default="step3.root")
parser.add_argument("--preset", default="all",
                    help="a preset name of truthGraphSelections.py, or all")
parser.add_argument("-n", "--maxEvents", type=int, default=3)
parser.add_argument("--src", default="truthLogicalGraphProducer", help="label of the logical graph")
parser.add_argument("--rebuild", action="store_true",
                    help="build the graph from the file's GEN and SIM record with the preset's selection")
parser.add_argument("--fragment", default=None,
                    help="with --rebuild, resolve the selection from this generator fragment name instead of "
                         "the bare preset, which a gun needs for its species")
args = parser.parse_args()
if "/" not in args.inputFile and ":" not in args.inputFile:
    args.inputFile = "file:" + args.inputFile

process = cms.Process("EXAMPLES")
process.load("FWCore.MessageService.MessageLogger_cfi")
process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(args.maxEvents))
process.source = cms.Source("PoolSource", fileNames=cms.untracked.vstring(args.inputFile))
process.options = cms.untracked.PSet(wantSummary=cms.untracked.bool(False))

process.examples = cms.EDAnalyzer("TruthGraphPresetExamples", src=cms.InputTag(args.src))
if args.preset != "all":
    process.examples.presets = cms.vstring(args.preset)
process.p = cms.Path(process.examples)

if args.rebuild:
    from PhysicsTools.TruthInfo.truthGraphSelections import postProcessingPSet
    from Validation.Configuration.truthPrevalidation_cff import truthLogicalGraphProducer as _logical
    process.truthGraphProducer = cms.EDProducer(
        "TruthGraphProducer",
        genEventHepMC3=cms.InputTag("generatorSmeared"),
        genEventHepMC=cms.InputTag("generatorSmeared"),
        simTracks=cms.InputTag("g4SimHits"),
        simVertices=cms.InputTag("g4SimHits"),
        addGenToSimEdges=cms.bool(True),
    )
    if args.fragment is not None:
        selection = postProcessingPSet(name=args.fragment)
    else:
        selection = postProcessingPSet(template="full" if args.preset == "all" else args.preset)
    process.truthLogicalGraphProducer = _logical.clone(
        src=cms.InputTag("truthGraphProducer"),
        postProcessing=selection,
    )
    process.examples.src = cms.InputTag("truthLogicalGraphProducer")
    process.p.insert(0, process.truthGraphProducer + process.truthLogicalGraphProducer)

process.MessageLogger.cerr.threshold = "WARNING"
