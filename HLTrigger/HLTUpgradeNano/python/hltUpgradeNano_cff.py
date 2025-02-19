import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.common_cff import *
from PhysicsTools.NanoAOD.nano_cff import nanoMetadata

hltUpgradeNanoTask = cms.Task(nanoMetadata)
hltUpgradeNanoSequence = cms.Sequence(hltUpgradeNanoTask)


### Tracksters
trackstersTable = cms.EDProducer(
    "TracksterCollectionTableProducer",
    src=cms.InputTag("hltTiclTrackstersMerge"),
    cut=cms.string(""),
    name=cms.string("tracksters"),
    doc=cms.string("HLT Merged Tracksters"),
    singleton=cms.bool(False),  # the number of entries is variable
    variables=cms.PSet(
        raw_energy=Var("raw_energy", "float", doc="Raw Energy of the trackster"),
    ),
    collectionVariables = cms.PSet(
        tracksterVertices = cms.PSet(
            name = cms.string("vertices"),
            doc = cms.string("Vertex properties"),
            useCount = cms.bool(True),
            useOffset = cms.bool(True),
            variables = cms.PSet(
                vertices = Var("vertices", "uint", doc="Layer clusters indices."),
                vertex_mult = Var("vertex_multiplicity", "float", doc="Fraction of Layer cluster energy used by the Trackster.")
            )
        )
    )
)


hltUpgradeNanoTask.add(trackstersTable)


def addAssociators(process):

    return process
