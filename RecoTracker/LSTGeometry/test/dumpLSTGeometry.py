import FWCore.ParameterSet.Config as cms
from Configuration.AlCa.GlobalTag import GlobalTag
import Configuration.Geometry.defaultPhase2ConditionsEra_cff as _settings
import FWCore.ParameterSet.VarParsing as VarParsing
from RecoTracker.LSTGeometry.lstGeometryESProducer_cfi import lstGeometryESProducer as _lstGeom

trackingLSTCommon = cms.Modifier()
trackingLST = cms.ModifierChain(trackingLSTCommon)

###################################################################
# Set default phase-2 settings
###################################################################
_PH2_GLOBAL_TAG, _PH2_ERA = _settings.get_era_and_conditions(_settings.DEFAULT_VERSION)

# No era in Fireworks/Geom reco dumper
process = cms.Process("DUMP", _PH2_ERA, trackingLST)

# import of standard configurations
process.load("Configuration.StandardSequences.Services_cff")
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("Configuration.Geometry.GeometryExtendedRun4DefaultReco_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

process.GlobalTag = GlobalTag(process.GlobalTag, _PH2_GLOBAL_TAG, "")

process.MessageLogger.cerr.threshold = "INFO"
process.MessageLogger.cerr.LSTGeometryESProducer = dict(limit=-1)

process.source = cms.Source("EmptySource")
process.maxEvents.input = 1

options = VarParsing.VarParsing()
options.register(
    "outputDirectory",
    "data/",
    VarParsing.VarParsing.multiplicity.singleton,
    VarParsing.VarParsing.varType.string,
    "Output directory for LST geometry files",
)
options.register(
    "ptCut",
    0.8,
    VarParsing.VarParsing.multiplicity.singleton,
    VarParsing.VarParsing.varType.float,
    "pT cut for LST module maps",
)
options.register(
    "binaryOutput",
    True,
    VarParsing.VarParsing.multiplicity.singleton,
    VarParsing.VarParsing.varType.bool,
    "Dump LST geometry as binary files",
)
options.parseArguments()

process.dump = cms.EDAnalyzer(
    "DumpLSTGeometry",
    outputDirectory = cms.untracked.string(options.outputDirectory),
    ptCut = cms.double(options.ptCut),
    outputAsBinary = cms.untracked.bool(options.binaryOutput),
)

process.lstGeometryESProducer = _lstGeom.clone(ptCut = cms.double(options.ptCut))
process.dTask = cms.Task(process.lstGeometryESProducer)
process.dSeq = cms.Sequence(process.dump,process.dTask)
process.p = cms.Path(process.dSeq)

print(f"Requesting LST geometry dump into directory: {options.outputDirectory}\n")
