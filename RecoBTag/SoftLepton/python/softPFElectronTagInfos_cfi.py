import FWCore.ParameterSet.Config as cms

softPFElectronsTagInfos = cms.EDProducer("SoftPFElectronTagInfoProducer",
    primaryVertex = cms.InputTag("offlinePrimaryVertices"),
    jets = cms.InputTag("ak4PFJetsCHS"),
    electrons = cms.InputTag("gedGsfElectrons"),
    DeltaRElectronJet=cms.double(0.4),
    MaxSip3Dsig=cms.double(200)
)

from Configuration.Eras.Modifier_phase2_timing_cff import phase2_timing
phase2_timing.toModify(
    softPFElectronsTagInfos,
    primaryVertex = cms.InputTag("offlinePrimaryVertices4D"),
)
