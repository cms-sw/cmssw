# Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
# A GEN-only job, a pi+ gun without SIM, with the GEN-only graph of customiseTruthGenOnly.
import FWCore.ParameterSet.Config as cms

from PhysicsTools.TruthInfo.customiseTruthGenOnly import customiseTruthGenOnly

process = cms.Process("GEN")
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.source = cms.Source("EmptySource")
process.maxEvents.input = 5
process.RandomNumberGeneratorService = cms.Service(
    "RandomNumberGeneratorService", generator=cms.PSet(initialSeed=cms.untracked.uint32(1234)))
process.generator = cms.EDProducer(
    "FlatRandomPtGunProducer",
    PGunParameters=cms.PSet(PartID=cms.vint32(211), MinPt=cms.double(10.), MaxPt=cms.double(20.),
                            MinEta=cms.double(-2.), MaxEta=cms.double(2.),
                            MinPhi=cms.double(-3.14159265359), MaxPhi=cms.double(3.14159265359)),
    AddAntiParticle=cms.bool(True),
    Verbosity=cms.untracked.int32(0),
)
process.generation_step = cms.Path(process.generator)
process.out = cms.OutputModule("PoolOutputModule", fileName=cms.untracked.string("genOnlyGraph.root"),
                               outputCommands=cms.untracked.vstring("drop *"))
process.e = cms.EndPath(process.out)

customiseTruthGenOnly(process)
# The gun writes its record as generator:unsmeared and the job has no vertex smearing.
process.truthGenOnlyGraphProducer.genEventHepMC = cms.InputTag("generator", "unsmeared")
process.truthGenOnlyGraphProducer.genEventHepMC3 = cms.InputTag("generator", "unsmeared")
process.truthGenOnlyLogicalGraphProducer.genEventHepMC = cms.InputTag("generator", "unsmeared")
process.truthGenOnlyLogicalGraphProducer.genEventHepMC3 = cms.InputTag("generator", "unsmeared")
