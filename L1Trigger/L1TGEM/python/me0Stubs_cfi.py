import FWCore.ParameterSet.Config as cms

me0Stubs = cms.EDProducer("ME0StubProducer",
    # parameters for l1t::me0::Config
    skip_centroids = cms.bool(False), 
    ly_thresh_patid = cms.vint32(7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 5, 5, 4, 4, 4, 4, 4), 
    ly_thresh_eta = cms.vint32(4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4), 
    max_span = cms.int32(37), 
    width = cms.int32(192), 
    deghost_pre = cms.bool(True), 
    deghost_post = cms.bool(True), 
    group_width = cms.int32(8), 
    ghost_width = cms.int32(1), 
    x_prt_en = cms.bool(True), 
    en_non_pointing = cms.bool(False), 
    cross_part_seg_width = cms.int32(4), 
    num_outputs = cms.int32(4), 
    check_ids = cms.bool(False), 
    edge_distance = cms.int32(2), 
    num_or = cms.int32(2),
    mse_thresh = cms.double(0.75),
    # input collections : GEMPadDigis
    InputCollection = cms.InputTag("GEMPadDigis"),
)