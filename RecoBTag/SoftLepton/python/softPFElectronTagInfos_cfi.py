import FWCore.ParameterSet.Config as cms

softPFElectronsTagInfos = cms.EDProducer("SoftPFElectronTagInfoProducer",
    primaryVertex = cms.InputTag("offlinePrimaryVertices"),
    jets = cms.InputTag("ak4PFJetsCHS"),
    electrons = cms.InputTag("gedGsfElectrons"),
    DeltaRElectronJet=cms.double(0.4),
    MaxSip3D=cms.double(200)
)
