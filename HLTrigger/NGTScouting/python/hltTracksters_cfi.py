import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.common_cff import *
from PhysicsTools.NanoAOD.nano_cff import nanoMetadata
from Validation.HGCalValidation.HLT_TICLIterLabels_cff import hltTiclIterLabels

hltUpgradeNanoTask = cms.Task(nanoMetadata)
hltSimTrackstersLabels = [
    'hltTiclSimTracksters', 'hltTiclSimTrackstersfromCPs']
# Tracksters
hltTrackstersTable = []
hltSimTrackstersTable = []
hltTrackstersAssociationOneToManyTableProducers = []
tracksterTableProducers = []
for iterLabel in hltTiclIterLabels:
    tracksterTable = cms.EDProducer(
        "TracksterCollectionTableProducer",
        skipNonExistingSrc=cms.bool(True),
        src=cms.InputTag(iterLabel),
        cut=cms.string(""),
        name=cms.string(iterLabel),
        doc=cms.string(iterLabel),
        singleton=cms.bool(False),  # the number of entries is variable
        variables=cms.PSet(
            raw_energy=Var("raw_energy", "float",
                           doc="Raw Energy of the trackster [GeV]"),
            raw_em_energy=Var("raw_em_energy", "float",
                              doc="EM raw Energy of the trackster [GeV]"),
            raw_pt=Var(
                "raw_pt", "float", doc="Trackster raw pT, computed from trackster raw energy and direction [GeV]"),
            regressed_energy=Var("regressed_energy", "float",
                                 doc="Regressed Energy of the trackster, for the SimTrackster it corresponds to the GEN-energy"),
            barycenter_x=Var("barycenter.x", "float",
                             doc="Trackster barycenter x [cm]"),
            barycenter_y=Var("barycenter.y", "float",
                             doc="Trackster barycenter y [cm]"),
            barycenter_z=Var("barycenter.z", "float",
                             doc="Trackster barycenter z [cm]"),
            barycenter_eta=Var("barycenter.eta", "float",
                               doc="Trackster barycenter pseudorapidity"),
            barycenter_phi=Var("barycenter.phi", "float",
                               doc="Trackster barycenter phi"),
            EV1=Var("eigenvalues()[0]", "float",
                    doc="Trackster PCA eigenvalues 0"),
            EV2=Var("eigenvalues()[1]", "float",
                    doc="Trackster PCA eigenvalues 1"),
            EV3=Var("eigenvalues()[2]", "float",
                    doc="Trackster PCA eigenvalues 2"),
            eVector0_x=Var(
                "eigenvectors()[0].x", "float", doc="Trackster PCA principal axis, x component"),
            eVector0_y=Var(
                "eigenvectors()[0].z", "float", doc="Trackster PCA principal axis, y component"),
            eVector0_z=Var(
                "eigenvectors()[0].y", "float", doc="Trackster PCA principal axis, z component"),
            time=Var("time", "float", doc="Trackster HGCAL time"),
            timeError=Var("timeError", "float",
                          doc="Trackster HGCAL time error")
        ),
        collectionVariables=cms.PSet(
            tracksterVertices=cms.PSet(
                name=cms.string(f"{iterLabel}vertices"),
                doc=cms.string("Vertex properties"),
                useCount=cms.bool(True),
                useOffset=cms.bool(True),
                variables=cms.PSet(
                    vertices=Var("vertices", "uint",
                                 doc="Layer clusters indices."),
                    vertex_mult=Var(
                        "vertex_multiplicity",
                        "float",
                        doc="Fraction of Layer cluster energy used by the Trackster.",
                    ),
                ),
            )
        ),
    )
    label = f"{iterLabel}TableProducer"
    globals()[label] = tracksterTable.clone()
    tracksterTableProducers.append(globals()[label])
    for iterLabelSim in hltSimTrackstersLabels:
        CP_SC_label = "CP" if "CP" in iterLabelSim else "SC"
        trackstersAssociationOneToManyS2RTable = cms.EDProducer(
            "TracksterTracksterEnergyScoreFlatTableProducer",
            src=cms.InputTag(
                f"hltAllTrackstersToSimTrackstersAssociationsByHits:{iterLabelSim}To{iterLabel}"
            ),
            name=cms.string(f"Sim{CP_SC_label}2{iterLabel}ByHits"),
            doc=cms.string(
                f"Association between SimTracksters and {iterLabel}, by hits."),
            collectionVariables=cms.PSet(
                links=cms.PSet(
                    name=cms.string(
                        f"Sim{CP_SC_label}2{iterLabel}ByHitsLinks"),
                    doc=cms.string("Association links."),
                    useCount=cms.bool(True),
                    useOffset=cms.bool(True),
                    variables=cms.PSet(
                        index=Var("index", "uint",
                                  doc="Index of the associated Trackster."),
                        sharedEnergy=Var(
                            "sharedEnergy",
                            "float",
                            doc="Shared energy with associated Trackster.",
                        ),
                        score=Var("score", "float", doc="Sim2Reco Association score."),
                    ),
                )
            ),
        )
        labelAssociation = f"{iterLabelSim}To{iterLabel}AssociationTableProducer"
        globals()[labelAssociation] = trackstersAssociationOneToManyS2RTable.clone()
        hltTrackstersAssociationOneToManyTableProducers.append(
            globals()[labelAssociation])

        trackstersAssociationOneToManyR2STable = cms.EDProducer(
            "TracksterTracksterEnergyScoreFlatTableProducer",
            src=cms.InputTag(
                f"hltAllTrackstersToSimTrackstersAssociationsByHits:{iterLabel}To{iterLabelSim}"
            ),
            name=cms.string(f"Reco{iterLabel}2Sim{CP_SC_label}ByHits"),
            doc=cms.string(
                f"Association between {iterLabel} and SimTracksters, by hits."),
            collectionVariables=cms.PSet(
                links=cms.PSet(
                    name=cms.string(f"Reco{iterLabel}2Sim{CP_SC_label}ByHitsLinks"),
                    doc=cms.string("Association links."),
                    useCount=cms.bool(True),
                    useOffset=cms.bool(False),
                    variables=cms.PSet(
                        index=Var("index", "uint",
                                  doc="Index of the associated SimTrackster."),
                        sharedEnergy=Var(
                            "sharedEnergy",
                            "float",
                            doc="Shared energy with associated SimTrackster.",
                        ),
                        score=Var("score", "float", doc="Reco2Sim Association score."),
                    ),
                )
            ),
        )
        labelAssociation = f"{iterLabel}To{iterLabelSim}AssociationTableProducer"
        globals()[labelAssociation] = trackstersAssociationOneToManyR2STable.clone()
        hltTrackstersAssociationOneToManyTableProducers.append(
            globals()[labelAssociation])

hltTrackstersTableSequence = cms.Sequence(
    sum(tracksterTableProducers, cms.Sequence()))
hltTiclAssociationsTableSequence = cms.Sequence(
    sum(hltTrackstersAssociationOneToManyTableProducers, cms.Sequence()))
simTracksterTableProducers = []
for iterLabel in hltSimTrackstersLabels:
    label = iterLabel
    objName = ""
    if ("CP" in iterLabel):
        label, objName = iterLabel.split("hltTiclSimTracksters")
    hltSimTracksterTable = cms.EDProducer(
        "TracksterCollectionTableProducer",
        skipNonExistingSrc=cms.bool(True),
        src=cms.InputTag(f"hltTiclSimTracksters", objName),
        cut=cms.string(""),
        name=cms.string(f"{iterLabel}"),
        doc=cms.string(f"{iterLabel}"),
        singleton=cms.bool(False),  # the number of entries is variable
        variables=cms.PSet(
            raw_energy=Var("raw_energy", "float",
                           doc="Raw Energy of the trackster [GeV]"),
            raw_em_energy=Var("raw_em_energy", "float",
                              doc="EM raw Energy of the trackster [GeV]"),
            raw_pt=Var(
                "raw_pt", "float", doc="Trackster raw pT, computed from trackster raw energy and direction [GeV]"),
            regressed_energy=Var("regressed_energy", "float",
                                 doc="Regressed Energy of the trackster, for the SimTrackster it corresponds to the GEN-energy"),
            barycenter_x=Var("barycenter.x", "float",
                             doc="Trackster barycenter x [cm]"),
            barycenter_y=Var("barycenter.y", "float",
                             doc="Trackster barycenter y [cm]"),
            barycenter_z=Var("barycenter.z", "float",
                             doc="Trackster barycenter z [cm]"),
            barycenter_eta=Var("barycenter.eta", "float",
                               doc="Trackster barycenter pseudorapidity"),
            barycenter_phi=Var("barycenter.phi", "float",
                               doc="Trackster barycenter phi"),
            EV1=Var("eigenvalues()[0]", "float",
                    doc="Trackster PCA eigenvalues 0"),
            EV2=Var("eigenvalues()[1]", "float",
                    doc="Trackster PCA eigenvalues 1"),
            EV3=Var("eigenvalues()[2]", "float",
                    doc="Trackster PCA eigenvalues 2"),
            eVector0_x=Var(
                "eigenvectors()[0].x", "float", doc="Trackster PCA principal axis, x component"),
            eVector0_y=Var(
                "eigenvectors()[0].z", "float", doc="Trackster PCA principal axis, y component"),
            eVector0_z=Var(
                "eigenvectors()[0].y", "float", doc="Trackster PCA principal axis, z component"),
            time=Var("time", "float", doc="Trackster HGCAL time"),
            timeError=Var("timeError", "float",
                          doc="Trackster HGCAL time error")
        ),
        collectionVariables=cms.PSet(
            tracksterVertices=cms.PSet(
                name=cms.string(f"{iterLabel}vertices"),
                doc=cms.string("Vertex properties"),
                useCount=cms.bool(True),
                useOffset=cms.bool(True),
                variables=cms.PSet(
                    vertices=Var("vertices", "uint",
                                 doc="Layer clusters indices."),
                    vertex_mult=Var(
                        "vertex_multiplicity",
                        "float",
                        doc="Fraction of Layer cluster energy used by the Trackster.",
                    ),
                ),
            )
        ),
    )
    label = f"{iterLabel}TableProducer"
    globals()[label] = hltSimTracksterTable.clone()
    simTracksterTableProducers.append(globals()[label])

    hltTiclSimTrackstersExtraTable = cms.EDProducer("SimTracksterTableProducer",
                                                    tableName=cms.string(
                                                        f"{iterLabel}"),
                                                    skipNonExistingSrc=cms.bool(
                                                        True),
                                                    simTracksters=cms.InputTag(
                                                        "hltTiclSimTracksters", objName),
                                                    caloParticles=cms.InputTag(
                                                        "mix", "MergedCaloTruth"),
                                                    simClusters=cms.InputTag(
                                                        "mix", "MergedCaloTruth"),
                                                    caloParticleToSimClustersMap=cms.InputTag(
                                                        "hltTiclSimTracksters"),
                                                    precision=cms.int32(7),
                                                    )
    labelExtra = f"{iterLabel}TableExtraProducer"
    globals()[labelExtra] = hltTiclSimTrackstersExtraTable.clone()
    simTracksterTableProducers.append(globals()[labelExtra])

hltSimTracksterSequence = cms.Sequence(
    sum(simTracksterTableProducers, cms.Sequence()))
# Tracksters Associators
hltSimCl2CPOneToOneFlatTable = cms.EDProducer(
    "SimClusterCaloParticleFractionFlatTableProducer",
    src=cms.InputTag(
        "SimClusterToCaloParticleAssociation:simClusterToCaloParticleMap"),
    name=cms.string("SimCl2CPWithFraction"),
    doc=cms.string("Association between SimClusters and CaloParticles."),
    variables=cms.PSet(
        index=Var("index", "int", doc="Index of linked CaloParticle."),
        fraction=Var("fraction", "float",
                     doc="Fraction of linked CaloParticle."),
    ),
)
hltTiclAssociationsTableSequence += hltSimCl2CPOneToOneFlatTable
