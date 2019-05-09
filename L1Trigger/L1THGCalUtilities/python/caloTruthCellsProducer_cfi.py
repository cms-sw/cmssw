import FWCore.ParameterSet.Config as cms

from L1Trigger.L1THGCal.hgcalBackEndLayer1Producer_cfi import dummy_C2d_params

caloTruthCellsProducer = cms.EDProducer('CaloTruthCellsProducer',
    caloParticles = cms.InputTag('mix', 'MergedCaloTruth'),
    triggerCells = cms.InputTag('hgcalVFEProducer:HGCalVFEProcessorSums'),
    simHitsEE = cms.InputTag('g4SimHits:HGCHitsEE'),
    simHitsHEfront = cms.InputTag('g4SimHits:HGCHitsHEfront'),
    simHitsHEback = cms.InputTag('g4SimHits:HcalHits'),
    makeCellsCollection = cms.bool(True),
    dummyClustering = dummy_C2d_params.clone()
)
