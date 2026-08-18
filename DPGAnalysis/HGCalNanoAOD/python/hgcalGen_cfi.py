import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.common_cff import *
from PhysicsTools.NanoAOD.nano_cff import nanoMetadata

# GenParticles table 
hgcalGenPartTable = cms.EDProducer(
    "SimpleGenParticleFlatTableProducer",
    skipNonExistingSrc=cms.bool(True),
    src=cms.InputTag("genParticles"),
    cut=cms.string("status==1 && pt > 0.5"),  # stable gen particles with pT > 0.5
    name=cms.string("HGCalGenPart"),
    doc=cms.string("Gen particles for HGCAL studies"),
    singleton=cms.bool(False),
    variables=cms.PSet(
        pt=Var("pt", "float", doc="gen particle pT"),
        eta=Var("eta", "float", doc="gen particle eta"),
        phi=Var("phi", "float", doc="gen particle phi"),
        mass=Var("mass", "float", doc="gen particle mass"),
        pdgId=Var("pdgId", "int", doc="gen particle PDG ID"),
        status=Var("status", "int", doc="gen particle status"),
        energy=Var("energy", "float", doc="gen particle energy"),
        vx=Var("vx", "float", doc="gen particle vx"),
        vy=Var("vy", "float", doc="gen particle vy"),
        vz=Var("vz", "float", doc="gen particle vz"),
    ),
)

hgcalGenSequence = cms.Sequence(hgcalGenPartTable)
