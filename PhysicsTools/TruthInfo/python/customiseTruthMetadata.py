# Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>

"""Write truth::GraphMetadata, once per run, in a job that builds the truth graph.

The metadata says how the graph was built: the build mode, the selection and pruning, the
level bits, the hit-index scope and energy units, the generator and the release. The
producer reads the configuration of the graph modules and of the generator from the job
itself, and from the jobs before it, so it describes the modules that ran.

The generator program name, its version and the fragment name are not in any parameter
set. They are filled here when this job runs the generator, and are empty otherwise.

From cmsDriver, after the customise or the step that adds the graph producers:

    --customise PhysicsTools/TruthInfo/customiseTruthMetadata.customiseTruthMetadata
"""

import os
import re

import FWCore.ParameterSet.Config as cms

# The scram tool of each generator program, by the prefix of the generator module type.
_GENERATOR_TOOLS = (
    ("Pythia8", "pythia8"),
    ("Herwig7", "herwig7"),
    ("Sherpa", "sherpa"),
)

# The labels of the graph modules: the GEN-only job of customiseTruthGenOnly, and the DIGI
# step of the enableTruth chain.
_GEN_ONLY_LABELS = dict(graphProducer="truthGenOnlyLogicalGraphProducer",
                        rawGraphProducer="truthGenOnlyGraphProducer",
                        hitIndexProducer="")
_DIGI_LABELS = dict(graphProducer="truthLogicalGraphProducer",
                    rawGraphProducer="mix",
                    hitIndexProducer="truthLogicalGraphHitIndexProducer")

# The output module types that write EDM files and so can hold the truth products.
_EDM_OUTPUT_TYPES = ("PoolOutputModule", "TimeoutPoolOutputModule", "RNTupleTempOutputModule")

truthGraphMetadataEventContent = cms.untracked.vstring("keep *_truthGraphMetadata_*_*")


def _toolVersion(tool):
    """The version of a scram tool of this release, from its toolbox file, or "" when the
    file is not found."""
    arch = os.environ.get("SCRAM_ARCH", "")
    for base in (os.environ.get("CMSSW_BASE"), os.environ.get("CMSSW_RELEASE_BASE")):
        if not base:
            continue
        path = os.path.join(base, "config", "toolbox", arch, "tools", "selected", tool + ".xml")
        try:
            with open(path) as toolFile:
                match = re.search(r'<tool\s[^>]*\bversion="([^"]*)"', toolFile.read())
        except OSError:
            continue
        if match:
            return match.group(1)
    return ""


def truthOutputModules(process):
    """The output modules that write EDM files of events: PoolOutputModule-type modules,
    without the LHE stream, which also writes the events the GEN step rejects."""
    modules = []
    for out in process.outputModules_().values():
        if out.type_() not in _EDM_OUTPUT_TYPES:
            continue
        if hasattr(out, "dataset") and hasattr(out.dataset, "dataTier") and out.dataset.dataTier.value() == "LHE":
            continue
        modules.append(out)
    return modules


def generatorInfo(process, label="generator"):
    """The generator program name, its version and the fragment name, for a job that runs
    the generator. Empty strings for a job that does not."""
    if not hasattr(process, label):
        return dict(generatorName="", generatorVersion="", generatorFragment="")
    moduleType = getattr(process, label).type_()
    name, version = moduleType, ""
    for prefix, tool in _GENERATOR_TOOLS:
        if moduleType.startswith(prefix):
            name, version = tool, _toolVersion(tool)
            break
    fragment = ""
    if hasattr(process, "configurationMetadata") and hasattr(process.configurationMetadata, "annotation"):
        # cmsDriver writes "<fragment> nevts:<n>".
        fragment = process.configurationMetadata.annotation.value().split(" ")[0]
    return dict(generatorName=name, generatorVersion=version, generatorFragment=fragment)


def addTruthMetadata(process, graphProducer, rawGraphProducer, hitIndexProducer=""):
    """Add the metadata producer for the named graph modules, and keep its product in every
    EDM output module."""
    if hasattr(process, "truthGraphMetadata"):
        raise RuntimeError("addTruthMetadata: the job already has truthGraphMetadata")
    info = generatorInfo(process)
    process.truthGraphMetadata = cms.EDProducer(
        "TruthGraphMetadataProducer",
        graphProducer=cms.string(graphProducer),
        rawGraphProducer=cms.string(rawGraphProducer),
        hitIndexProducer=cms.string(hitIndexProducer),
        generatorLabel=cms.string("generator"),
        **{key: cms.string(value) for key, value in info.items()},
    )
    process.truthGraphMetadataPath = cms.Path(process.truthGraphMetadata)
    if process.schedule is not None:
        process.schedule.insert(0, process.truthGraphMetadataPath)
    for out in truthOutputModules(process):
        out.outputCommands.extend(truthGraphMetadataEventContent)
    return process


def customiseTruthMetadata(process):
    """Add the metadata producer for the graph modules this job has: the GEN-only ones of
    customiseTruthGenOnly, or the DIGI ones of the enableTruth chain. A job with both, or
    with neither, needs addTruthMetadata with explicit labels."""
    hasGenOnly = hasattr(process, _GEN_ONLY_LABELS["graphProducer"])
    hasDigi = hasattr(process, _DIGI_LABELS["graphProducer"])
    if hasGenOnly == hasDigi:
        raise RuntimeError("customiseTruthMetadata: the job has %s graph producer; "
                           "call addTruthMetadata with the labels of the graph to describe"
                           % ("a GEN-only and a DIGI" if hasGenOnly else "no"))
    return addTruthMetadata(process, **(_GEN_ONLY_LABELS if hasGenOnly else _DIGI_LABELS))
