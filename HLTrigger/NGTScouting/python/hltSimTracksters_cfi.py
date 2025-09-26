import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.common_cff import *
from PhysicsTools.NanoAOD.nano_cff import nanoMetadata
from Validation.HGCalValidation.HLT_TICLIterLabels_cff import hltTiclIterLabels

hltUpgradeNanoTask = cms.Task(nanoMetadata)

hltSimTracksterTable = cms.EDProducer(
    "TracksterCollectionTableProducer",
    skipNonExistingSrc=cms.bool(True),
    src=cms.InputTag("hltTiclSimTracksters"),
    cut=cms.string(""),
    name=cms.string("hltTiclSimTracksters"),
    doc=cms.string("hltTiclSimTracksters"),
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
            name=cms.string(f"hltTiclSimTrackstersvertices"),
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

hltSimTracksterFromCPsTable = cms.EDProducer(
    "TracksterCollectionTableProducer",
    skipNonExistingSrc=cms.bool(True),
    src=cms.InputTag("hltTiclSimTracksters", "fromCPs"),
    cut=cms.string(""),
    name=cms.string("hltTiclSimTrackstersFromCPs"),
    doc=cms.string("hltTiclSimTrackstersFromCPs"),
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
            name=cms.string(f"hltTiclSimTrackstersFromCPsvertices"),
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

# Extra tables for SimTracksters using information from SimClusters and CaloParticles 
# Might be replaced in case we save CaloParticles and SimClusters, in that case we just need adding the corresponding SimObject indices in the SimTrackster tables
hltTiclSimTrackstersExtraTable = cms.EDProducer("SimTracksterTableProducer",
                                tableName = cms.string("hltTiclSimTracksters"),
                                skipNonExistingSrc = cms.bool(True),
                                simTracksters = cms.InputTag( "hltTiclSimTracksters" ),
                                caloParticles = cms.InputTag( "mix", "MergedCaloTruth" ),
                                simClusters = cms.InputTag( "mix", "MergedCaloTruth" ),
                                caloParticleToSimClustersMap = cms.InputTag("hltTiclSimTracksters"),
                                precision = cms.int32(7),
                                )

hltTiclSimTrackstersFromCPsExtraTable = cms.EDProducer("SimTracksterTableProducer",
                                tableName = cms.string("hltTiclSimTrackstersFromCPs"),
                                skipNonExistingSrc = cms.bool(True),
                                simTracksters = cms.InputTag( "hltTiclSimTracksters", "fromCPs"),
                                caloParticles = cms.InputTag( "mix", "MergedCaloTruth" ),
                                simClusters = cms.InputTag( "mix", "MergedCaloTruth" ),
                                caloParticleToSimClustersMap = cms.InputTag("hltTiclSimTracksters"),
                                precision = cms.int32(7),
                                )
