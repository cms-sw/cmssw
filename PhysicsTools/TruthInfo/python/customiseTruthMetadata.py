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
import subprocess

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

truthGraphMetadataEventContent = cms.untracked.vstring("keep *_truthGraphMetadata_*_*")


def _toolVersion(tool):
    """The version of a scram tool of this release, or "" when scram does not answer. scram
    answers only inside the release area."""
    try:
        info = subprocess.run(["scram", "tool", "info", tool], capture_output=True, text=True, timeout=60,
                              cwd=os.environ.get("CMSSW_BASE")).stdout
    except (OSError, subprocess.SubprocessError):
        return ""
    match = re.search(r"^Version\s*:\s*(\S+)", info, re.MULTILINE)
    return match.group(1) if match else ""


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
    output module."""
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
    for out in process.outputModules_().values():
        out.outputCommands.extend(truthGraphMetadataEventContent)
    return process


def customiseTruthMetadata(process):
    """Add the metadata producer for the graph modules this job has: the GEN-only ones of
    customiseTruthGenOnly, else the DIGI ones of the enableTruth chain."""
    labels = _GEN_ONLY_LABELS if hasattr(process, _GEN_ONLY_LABELS["graphProducer"]) else _DIGI_LABELS
    return addTruthMetadata(process, **labels)
