import FWCore.ParameterSet.Config as cms

tpFastJets = cms.EDProducer("TPFastJetProducer",
    TrackingParticleInputTag = cms.InputTag("mix", "MergedTrackTruth"),
    MCTruthStubInputTag = cms.InputTag("TTStubAssociatorFromPixelDigis", "StubAccepted"),
    tp_ptMin = cms.double(2.0),       # minimum tp pt [GeV]
    tp_etaMax = cms.double(2.4),      # maximum tp eta
    tp_zMax = cms.double(15.),        # max tp z0 [cm]
    tp_nStubMin = cms.int32(4),       # minimum number of stubs
    tp_nStubLayerMin = cms.int32(4),  # minimum number of layers with stubs 
    coneSize=cms.double(0.4),         # cone size for anti-kt fast jet 
)
