import FWCore.ParameterSet.Config as cms

dtTriggerPhase2Shower = cms.EDProducer("DTTrigPhase2ShowerProd",
    bmtl1_hits_bxpersistence = cms.int32(16),
    debug = cms.untracked.bool(True),
    digiTag = cms.InputTag("CalibratedDigis"),
    dump_digis = cms.untracked.bool(False),
    nHits_per_bx_phi = cms.int32(8),
    nHits_per_bx_z = cms.int32(8),
    obdt_hits_bxpersistence_phi = cms.int32(4),
    obdt_hits_bxpersistence_z = cms.int32(4),
    obdt_wire_relaxing_time = cms.int32(2),
    scenario = cms.int32(0),
    showerTaggingAlgo = cms.int32(1),
    threshold_for_shower = cms.int32(6)
)
