import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.common_cff import *
from PhysicsTools.NanoAOD.nano_cff import nanoMetadata
from RecoHGCal.TICL.iterativeTICL_cff import ticlIterLabelsPSet

hgcalUpgradeNanoTask = cms.Task(nanoMetadata)
hgcalSimTrackstersLabels = [
    'ticlSimTracksters', 'ticlSimTrackstersfromCPs']


def createTracksterTables(ticlIterLabels, simTrackstersLabels, collectionPrefix=""):
    """
    Args:
        ticlIterLabels: List of TICL iteration labels
        simTrackstersLabels: List of sim trackster labels
        collectionPrefix: Prefix for InputTag collections ("" for offline, "hlt" for HLT)

    Returns:
        Dictionary mapping module names to EDProducer objects
    """
    producers = {}

    tracksterVars = cms.PSet(
        raw_energy=Var("raw_energy", "float", doc="Raw Energy of the trackster [GeV]"),
        raw_em_energy=Var("raw_em_energy", "float", doc="EM raw Energy of the trackster [GeV]"),
        raw_pt=Var("raw_pt", "float", doc="Trackster raw pT, computed from trackster raw energy and direction [GeV]"),
        regressed_energy=Var("regressed_energy", "float", doc="Regressed Energy of the trackster, for the SimTrackster it corresponds to the GEN-energy"),
        barycenter_x=Var("barycenter.x", "float", doc="Trackster barycenter x [cm]"),
        barycenter_y=Var("barycenter.y", "float", doc="Trackster barycenter y [cm]"),
        barycenter_z=Var("barycenter.z", "float", doc="Trackster barycenter z [cm]"),
        barycenter_eta=Var("barycenter.eta", "float", doc="Trackster barycenter pseudorapidity"),
        barycenter_phi=Var("barycenter.phi", "float", doc="Trackster barycenter phi"),
        EV1=Var("eigenvalues()[0]", "float", doc="Trackster PCA eigenvalues 0"),
        EV2=Var("eigenvalues()[1]", "float", doc="Trackster PCA eigenvalues 1"),
        EV3=Var("eigenvalues()[2]", "float", doc="Trackster PCA eigenvalues 2"),
        eVector0_x=Var("eigenvectors()[0].x", "float", doc="Trackster PCA principal axis, x component"),
        eVector0_y=Var("eigenvectors()[0].z", "float", doc="Trackster PCA principal axis, y component"),
        eVector0_z=Var("eigenvectors()[0].y", "float", doc="Trackster PCA principal axis, z component"),
        time=Var("time", "float", doc="Trackster HGCAL time"),
        timeError=Var("timeError", "float", doc="Trackster HGCAL time error")
    )

    # Determine the association prefix based on collection, not really elegant
    # hltAllTrackstersToSimTrackstersAssociationsByHits or allTrackstersToSimTrackstersAssociationsByHits
    assocPrefix = collectionPrefix if collectionPrefix else ""
    if assocPrefix:
        assocPrefix += "All"
    else:
        assocPrefix = "all"

    # Create trackster tables for each label 
    for iterLabel in ticlIterLabels:
        table = cms.EDProducer(
            "TracksterCollectionTableProducer",
            skipNonExistingSrc=cms.bool(True),
            src=cms.InputTag(iterLabel),
            cut=cms.string(""),
            name=cms.string(iterLabel),
            doc=cms.string(iterLabel),
            singleton=cms.bool(False),
            variables=tracksterVars,
            collectionVariables=cms.PSet(
                tracksterVertices=cms.PSet(
                    name=cms.string(f"{iterLabel}vertices"),
                    doc=cms.string("Vertex properties"),
                    useCount=cms.bool(True),
                    useOffset=cms.bool(True),
                    variables=cms.PSet(
                        vertices=Var("vertices", "uint", doc="Layer clusters indices."),
                        vertex_mult=Var("vertex_multiplicity", "float", doc="Fraction of Layer cluster energy used by the Trackster."),
                    ),
                )
            ),
        )
        producers[f"{collectionPrefix}{iterLabel}TableProducer"] = table

        # Create association tables
        for simLabel in simTrackstersLabels:
            CP_SC_label = "CP" if "CP" in simLabel else "SC"
            # Sim2Reco association
            assocTable = cms.EDProducer(
                "TracksterTracksterEnergyScoreFlatTableProducer",
                src=cms.InputTag(f"{assocPrefix}TrackstersToSimTrackstersAssociationsByHits:{simLabel}To{iterLabel}"),
                skipNonExistingSrc=cms.bool(True),
                name=cms.string(f"Sim{CP_SC_label}2{iterLabel}ByHits"),
                doc=cms.string(f"Association between SimTracksters and {iterLabel}, by hits."),
                collectionVariables=cms.PSet(
                    links=cms.PSet(
                        name=cms.string(f"Sim{CP_SC_label}2{iterLabel}ByHitsLinks"),
                        doc=cms.string("Association links."),
                        useCount=cms.bool(True),
                        useOffset=cms.bool(False),
                        variables=cms.PSet(
                            index=Var("index", "uint", doc="Index of the associated Trackster."),
                            sharedEnergy=Var("sharedEnergy", "float", doc="Shared energy with associated Trackster."),
                            score=Var("score", "float", doc="Sim2Reco Association score."),
                        ),
                    ),
                ),
            )
            producers[f"{collectionPrefix}{simLabel}To{iterLabel}AssociationTableProducer"] = assocTable

            # Reco2Sim association
            recoToSimAssocTable = cms.EDProducer(
                "TracksterTracksterEnergyScoreFlatTableProducer",
                src=cms.InputTag(f"{assocPrefix}TrackstersToSimTrackstersAssociationsByHits:{iterLabel}To{simLabel}"),
                skipNonExistingSrc=cms.bool(True),
                name=cms.string(f"Reco{iterLabel}2Sim{CP_SC_label}ByHits"),
                doc=cms.string(f"Association between {iterLabel} and SimTracksters, by hits."),
                collectionVariables=cms.PSet(
                    links=cms.PSet(
                        name=cms.string(f"Reco{iterLabel}2Sim{CP_SC_label}ByHitsLinks"),
                        doc=cms.string("Association links."),
                        useCount=cms.bool(True),
                        useOffset=cms.bool(False),
                        variables=cms.PSet(
                            index=Var("index", "uint", doc="Index of the associated SimTrackster."),
                            sharedEnergy=Var("sharedEnergy", "float", doc="Shared energy with associated SimTrackster."),
                            score=Var("score", "float", doc="Reco2Sim Association score."),
                        ),
                    ),
                ),
            )
            producers[f"{collectionPrefix}{iterLabel}To{simLabel}AssociationTableProducer"] = recoToSimAssocTable

    # Create sim trackster tables
    simCollectionName = f"{collectionPrefix}ticlSimTracksters" if collectionPrefix else "ticlSimTracksters"
    for simLabel in simTrackstersLabels:
        objName = ""
        if "CP" in simLabel:
            splitPattern = f"{collectionPrefix}TiclSimTracksters" if collectionPrefix else "ticlSimTracksters"
            _, objName = simLabel.split(splitPattern)

        simTable = cms.EDProducer(
            "TracksterCollectionTableProducer",
            skipNonExistingSrc=cms.bool(True),
            src=cms.InputTag(simCollectionName, objName),
            cut=cms.string(""),
            name=cms.string(simLabel),
            doc=cms.string(simLabel),
            singleton=cms.bool(False),
            variables=tracksterVars,
            collectionVariables=cms.PSet(
                tracksterVertices=cms.PSet(
                    name=cms.string(f"{simLabel}vertices"),
                    doc=cms.string("Vertex properties"),
                    useCount=cms.bool(True),
                    useOffset=cms.bool(True),
                    variables=cms.PSet(
                        vertices=Var("vertices", "uint", doc="Layer clusters indices."),
                        vertex_mult=Var("vertex_multiplicity", "float", doc="Fraction of Layer cluster energy used by the Trackster."),
                    ),
                )
            ),
        )
        producers[f"{collectionPrefix}{simLabel}TableProducer"] = simTable

        # Sim trackster extra table
        extraTable = cms.EDProducer(
            "SimTracksterTableProducer",
            tableName=cms.string(simLabel),
            skipNonExistingSrc=cms.bool(True),
            simTracksters=cms.InputTag(simCollectionName, objName),
            caloParticles=cms.InputTag("mix", "MergedCaloTruth"),
            simClusters=cms.InputTag("mix", "MergedCaloTruth"),
            caloParticleToSimClustersMap=cms.InputTag(simCollectionName),
            precision=cms.int32(7),
        )
        producers[f"{collectionPrefix}{simLabel}TableExtraProducer"] = extraTable

    return producers


# Create offline trackster tables
_offlineProducers = createTracksterTables(ticlIterLabelsPSet.labels, hgcalSimTrackstersLabels, collectionPrefix="")

for name, producer in _offlineProducers.items():
    globals()[name] = producer.clone()

# SimCluster to CaloParticle association 
SimCl2CPOneToOneFlatTable = cms.EDProducer(
    "SimClusterCaloParticleFractionFlatTableProducer",
    src=cms.InputTag("SimClusterToCaloParticleAssociation:simClusterToCaloParticleMap"),
    name=cms.string("SimCl2CPWithFraction"),
    doc=cms.string("Association between SimClusters and CaloParticles."),
    variables=cms.PSet(
        index=Var("index", "int", doc="Index of linked CaloParticle."),
        fraction=Var("fraction", "float", doc="Fraction of linked CaloParticle."),
    ),
)

tracksterTableProducers = []
hgcalTrackstersAssociationOneToManyTableProducers = []
simTracksterTableProducers = []

for name, producer in _offlineProducers.items():
    if "AssociationTableProducer" in name and "SimCl2CP" not in name:
        hgcalTrackstersAssociationOneToManyTableProducers.append(globals()[name])
    elif "SimTrackster" in name or "fromCPs" in name:
        simTracksterTableProducers.append(globals()[name])
    elif "TableProducer" in name and "Association" not in name and "SimCl2CP" not in name:
        tracksterTableProducers.append(globals()[name])

# Create sequences
hgcalTrackstersTableSequence = cms.Sequence(sum(tracksterTableProducers, cms.Sequence()))
hgcalTiclAssociationsTableSequence = cms.Sequence(sum(hgcalTrackstersAssociationOneToManyTableProducers, cms.Sequence()))
hgcalSimTracksterSequence = cms.Sequence(sum(simTracksterTableProducers, cms.Sequence()))
hgcalTiclAssociationsTableSequence += SimCl2CPOneToOneFlatTable
