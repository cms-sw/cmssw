# Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>

"""Build the truth graph from the GEN record alone, in a job without SIM.

The raw graph holds every GEN particle and vertex of the record, with the full decay
chain and the parton shower. The logical graph is built from it as at DIGI, except that
no particle is pruned for having no hit. Neither holds a SIM node, a hit or a GEN to SIM
link. The levels are filled, except caloBoundary, which needs the SIM and stays empty.

From cmsDriver, on a job that runs the GEN step:

    cmsDriver.py ... -s GEN \\
        --customise PhysicsTools/TruthInfo/customiseTruthGenOnly.customiseTruthGenOnly

The producers run on demand when an output module keeps their products. Every EDM
output module of the job keeps them, except the LHE stream. customiseTruthGenOnlyWithMetadata also writes
truth::GraphMetadata once per run (see customiseTruthMetadata).
"""

import FWCore.ParameterSet.Config as cms

from Validation.Configuration.truthPrevalidation_cff import (
    truthGraphProducer as _truthGraphProducer,
    truthLogicalGraphProducer as _truthLogicalGraphProducer,
)
from PhysicsTools.TruthInfo.truthGraphMixedDigi_cff import reconstructablePdgIds
from PhysicsTools.TruthInfo.customiseTruthMetadata import customiseTruthMetadata, truthOutputModules

truthGenOnlyGraphProducer = _truthGraphProducer.clone(
    simTracks=cms.InputTag(""),
    simVertices=cms.InputTag(""),
    addGenToSimEdges=cms.bool(False),
)

truthGenOnlyLogicalGraphProducer = _truthLogicalGraphProducer.clone(
    src=cms.InputTag("truthGenOnlyGraphProducer"),
    simTracks=cms.InputTag(""),
    simVertices=cms.InputTag(""),
    simHitCollections=cms.VInputTag(),
    trackerSimHitCollections=cms.VInputTag(),
    muonSimHitCollections=cms.VInputTag(),
)
truthGenOnlyLogicalGraphProducer.postProcessing.dropHitlessSimSubgraphs = cms.bool(False)
truthGenOnlyLogicalGraphProducer.postProcessing.reconstructablePdgIds = cms.vint32(*reconstructablePdgIds)

truthGenOnlyEventContent = cms.untracked.vstring(
    "keep *_truthGenOnlyLogicalGraphProducer_*_*",
    "keep *_truthGenOnlyGraphProducer_*_*",
)


def customiseTruthGenOnly(process, writeMetadata=False):
    """Add the GEN-only graph to the GEN step and keep it in every output module. With
    writeMetadata, also write truth::GraphMetadata once per run."""
    if not hasattr(process, "generation_step"):
        raise RuntimeError("customiseTruthGenOnly needs a job with the GEN step (process.generation_step)")
    process.truthGenOnlyGraphProducer = truthGenOnlyGraphProducer.clone()
    process.truthGenOnlyLogicalGraphProducer = truthGenOnlyLogicalGraphProducer.clone()
    process.truthGenOnlyGraphTask = cms.Task(process.truthGenOnlyGraphProducer,
                                             process.truthGenOnlyLogicalGraphProducer)
    process.generation_step.associate(process.truthGenOnlyGraphTask)
    for out in truthOutputModules(process):
        out.outputCommands.extend(truthGenOnlyEventContent)
    if writeMetadata:
        customiseTruthMetadata(process)
    return process


def customiseTruthGenOnlyWithMetadata(process):
    """customiseTruthGenOnly with the metadata, for cmsDriver."""
    return customiseTruthGenOnly(process, writeMetadata=True)
