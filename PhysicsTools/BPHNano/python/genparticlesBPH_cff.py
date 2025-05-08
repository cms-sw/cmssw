import FWCore.ParameterSet.Config as cms
from  PhysicsTools.NanoAOD.genparticles_cff import *
from PhysicsTools.NanoAOD.simpleGenParticleFlatTableProducer_cfi import simpleGenParticleFlatTableProducer


# for BPHPark start with merged particles (pruned + packed),
# where pruned contain K* states, but not final states, 
# and packed contain final states (K pi).
# then you save also final states (granddaughters)
finalGenParticlesBPH = finalGenParticles.clone(
  src = cms.InputTag("mergedGenParticles"),
  select = cms.vstring(
        "drop *",
        "keep++ (abs(pdgId) == 511 || abs(pdgId) == 521 || abs(pdgId)==531)",  #keep all B0(=511) and B+/-(521) + their daughters and granddaughters
   )
)

genParticleBPHTable = simpleGenParticleFlatTableProducer.clone(
  src = cms.InputTag("finalGenParticlesBPH"),
  name = cms.string("BPHGenPart"),
  doc = cms.string("interesting gen particles for BPH"),  
  variables = cms.PSet(
      genParticleTable.variables,
      vx = Var("vx", float, doc="x coordinate of the production vertex position, in cm", precision=10),
      vy = Var("vy", float, doc="y coordinate of the production vertex position, in cm", precision=10),
      vz = Var("vz", float, doc="z coordinate of the production vertex position, in cm", precision=10),
  )
)



genParticleBPHSequence = cms.Sequence(finalGenParticlesBPH)
genParticleBPHTables = cms.Sequence(genParticleBPHTable)

