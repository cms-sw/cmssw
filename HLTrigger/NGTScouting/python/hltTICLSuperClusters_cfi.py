import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.common_cff import *
from PhysicsTools.NanoAOD.nano_cff import nanoMetadata
from Validation.HGCalValidation.HLT_TICLIterLabels_cff import hltTiclIterLabels

hltUpgradeNanoTask = cms.Task(nanoMetadata)

# SuperClusters 
hltTiclSuperClustersTable = cms.EDProducer(
    "TICLSuperClustersTableProducer",
    skipNonExistingSrc=cms.bool(True),
    src=cms.InputTag("hltTiclEGammaSuperClusterProducerUnseeded"),
    cut=cms.string(""),
    name=cms.string("hltTICLSuperClusters"),
    doc=cms.string("HLT TICL SuperClusters"),
    singleton=cms.bool(False),  # the number of entries is variable
    variables=cms.PSet(
        raw_energy=Var("rawEnergy", "float",
                       doc="Raw Energy of the SuperCluster [GeV]"),
        energy=Var("energy", "float",
                          doc="Regressed Energy of SuperCluster [GeV]"),
        corrected_energy=Var(
            "correctedEnergy", "float", doc="Corrected energy of the SuperCluster [GeV]"),
        position_x=Var("position.x", "float",
                         doc="SuperCluster position x [cm]"),
        position_y=Var("position.y", "float",
                         doc="SuperCluster position y [cm]"),
        position_z=Var("position.z", "float",
                         doc="SuperCluster barycenter z [cm]"),
        position_eta=Var("position.eta", "float",
                           doc="SuperCluster position pseudorapidity"),
        position_phi=Var("position.phi", "float",
                           doc="SuperCluster position phi"),
    ),
)
