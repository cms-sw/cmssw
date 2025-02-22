import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.common_cff import *
from PhysicsTools.NanoAOD.nano_cff import nanoMetadata

hltUpgradeNanoTask = cms.Task(nanoMetadata)


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
    collectionVariables=cms.PSet(
        tracksterVertices=cms.PSet(
            name=cms.string("vertices"),
            doc=cms.string("Vertex properties"),
            useCount=cms.bool(True),
            useOffset=cms.bool(True),
            variables=cms.PSet(
                vertices=Var("vertices", "uint", doc="Layer clusters indices."),
                vertex_mult=Var(
                    "vertex_multiplicity",
                    "float",
                    doc="Fraction of Layer cluster energy used by the Trackster.",
                ),
            ),
        )
    ),
)

### Tracksters Associators
trackstersAssociationOneToManyTable = cms.EDProducer(
    "TracksterAssociationOneToManyCollectionTableProducer",
    src=cms.InputTag(
        "allTrackstersToSimTrackstersAssociationsByHits:ticlSimTrackstersToticlTrackstersMerge"
    ),
    name=cms.string("SimTS2TSMergeByHits"),
    doc=cms.string("Association betwewn SimTracksters and tracksterMerge, by hits."),
    singleton=cms.bool(True),  # the number of entries is variable
    collectionVariables=cms.PSet(
        links=cms.PSet(
            name=cms.string("SimTS2TSMergeByHitsLinks"),
            doc=cms.string("Association links."),
            useCount=cms.bool(True),
            useOffset=cms.bool(False),
            variables=cms.PSet(
                index=Var("index", "uint", doc="Layer clusters indices."),
                score=Var("score", "float", doc="Layer clusters score."),
            ),
        )
    ),
)

hltUpgradeNanoTask.add(trackstersTable, trackstersAssociationOneToManyTable)

hltUpgradeNanoSequence = cms.Sequence(hltUpgradeNanoTask)


def addAssociators(process):
    return process
