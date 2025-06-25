import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.common_cff import *
from PhysicsTools.NanoAOD.nano_cff import nanoMetadata

hltUpgradeNanoTask = cms.Task(nanoMetadata)


### Tracksters
hltTrackstersTable = cms.EDProducer(
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
hltTrackstersAssociationOneToManyTable = cms.EDProducer(
    "TracksterTracksterEnergyScoreFlatTableProducer",
    src=cms.InputTag(
        "hltAllTrackstersToSimTrackstersAssociationsByHits:hltTiclSimTrackstersTohltTiclTrackstersMerge"
    ),
    name=cms.string("SimTS2TSMergeByHits"),
    doc=cms.string("Association between SimTracksters and tracksterMerge, by hits."),
    collectionVariables=cms.PSet(
        links=cms.PSet(
            name=cms.string("SimTS2TSMergeByHitsLinks"),
            doc=cms.string("Association links."),
            useCount=cms.bool(True),
            useOffset=cms.bool(False),
            variables=cms.PSet(
                index=Var("index", "uint", doc="Index of the associated Trackster."),
                sharedEnergy=Var(
                    "sharedEnergy",
                    "float",
                    doc="Shared energy with associated Trackster.",
                ),
                score=Var("score", "float", doc="Association score."),
            ),
        )
    ),
)

### Tracksters Associators
hltSimCl2CPOneToOneFlatTable = cms.EDProducer(
    "SimClusterCaloParticleFractionFlatTableProducer",
    src=cms.InputTag("SimClusterToCaloParticleAssociation:simClusterToCaloParticleMap"),
    name=cms.string("SimCl2CPWithFraction"),
    doc=cms.string("Association between SimClusters and CaloParticles."),
    variables=cms.PSet(
        index=Var("index", "int", doc="Index of linked CaloParticle."),
        fraction=Var("fraction", "float", doc="Fraction of linked CaloParticle."),
    ),
)
