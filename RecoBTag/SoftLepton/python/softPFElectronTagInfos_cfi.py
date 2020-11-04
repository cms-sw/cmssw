import FWCore.ParameterSet.Config as cms

softPFElectronsTagInfos = cms.EDProducer("SoftPFElectronTagInfoProducer",
    primaryVertex = cms.InputTag("offlinePrimaryVertices"),
    jets = cms.InputTag("ak4PFJetsCHS"),
    electrons = cms.InputTag("gedGsfElectrons"),
    DeltaRElectronJet=cms.double(0.4),
    MaxSip3Dsig=cms.double(200)
)

from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA
pp_on_AA.toModify(softPFElectronsTagInfos, jets = "akCs4PFJets")
