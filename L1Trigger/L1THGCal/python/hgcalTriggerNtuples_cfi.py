import FWCore.ParameterSet.Config as cms


ntuple_event = cms.PSet(
    NtupleName = cms.string('HGCalTriggerNtupleEvent')
)

ntuple_gen = cms.PSet(
    NtupleName = cms.string('HGCalTriggerNtupleGen'),
    GenParticles = cms.InputTag('genParticles')
)

ntuple_gentau = cms.PSet(
    NtupleName = cms.string('HGCalTriggerNtupleGenTau'),
    GenParticles = cms.InputTag('genParticles'),
    isPythia8 = cms.bool(True)
)

ntuple_genjet = cms.PSet(
    NtupleName = cms.string('HGCalTriggerNtupleGenJet'),
    GenJets = cms.InputTag('ak4GenJetsNoNu')
)

ntuple_digis = cms.PSet(
    NtupleName = cms.string('HGCalTriggerNtupleHGCDigis'),
    HGCDigisEE = cms.InputTag('mix:HGCDigisEE'),
    HGCDigisFH = cms.InputTag('mix:HGCDigisHEfront'),
    HGCDigisBH = cms.InputTag('mix:HGCDigisHEback'),
    eeSimHits = cms.InputTag('g4SimHits:HGCHitsEE'),
    fhSimHits = cms.InputTag('g4SimHits:HGCHitsHEfront'),
    bhSimHits = cms.InputTag('g4SimHits:HcalHits'),
    isSimhitComp = cms.bool(False)
)

ntuple_triggercells = cms.PSet(
    NtupleName = cms.string('HGCalTriggerNtupleHGCTriggerCells'),
    TriggerCells = cms.InputTag('hgcalTriggerPrimitiveDigiProducer:calibratedTriggerCells')
)

ntuple_clusters = cms.PSet(
    NtupleName = cms.string('HGCalTriggerNtupleHGCClusters'),
    Clusters = cms.InputTag('hgcalTriggerPrimitiveDigiProducer:cluster2D')
)

ntuple_multicluster = cms.PSet(
    NtupleName = cms.string('HGCalTriggerNtupleHGCMulticlusters'),
    Multiclusters = cms.InputTag('hgcalTriggerPrimitiveDigiProducer:cluster3D')
)

hgcalTriggerNtuplizer = cms.EDAnalyzer(
    "HGCalTriggerNtupleManager",
    Ntuples = cms.VPSet(
        ntuple_event,
        ntuple_gen,
        ntuple_genjet,
        ntuple_digis,
        ntuple_triggercells,
        ntuple_clusters,
        ntuple_multicluster
    )
)
