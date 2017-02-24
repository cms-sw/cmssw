import FWCore.ParameterSet.Config as cms


ntuple_event = cms.PSet(
    NtupleName = cms.string('HGCalTriggerNtupleEvent')
)

ntuple_gen = cms.PSet(
    NtupleName = cms.string('HGCalTriggerNtupleGen'),
    GenParticles = cms.InputTag('genParticles')
)

ntuple_digis = cms.PSet(
    NtupleName = cms.string('HGCalTriggerNtupleHGCDigis'),
    HGCDigisEE = cms.InputTag('mix:HGCDigisEE'),
    HGCDigisFH = cms.InputTag('mix:HGCDigisHEfront'),
    eeSimHits = cms.InputTag('g4SimHits:HGCHitsEE'),
    fhSimHits = cms.InputTag('g4SimHits:HGCHitsHEfront'),
    isSimhitComp = cms.bool(False)
)

ntuple_triggercells = cms.PSet(
    NtupleName = cms.string('HGCalTriggerNtupleHGCTriggerCells'),
    TriggerCells = cms.InputTag('hgcalTriggerPrimitiveDigiProducer:SingleCellClusterAlgoBestChoice')
)


hgcalTriggerNtuplizer = cms.EDAnalyzer(
    "HGCalTriggerNtupleManager",
    Ntuples = cms.VPSet(
        ntuple_event,
        ntuple_gen,
        ntuple_digis,
        ntuple_triggercells
    )
)
