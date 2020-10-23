import FWCore.ParameterSet.Config as cms

softPFElectronsTagInfos = cms.EDProducer("SoftPFElectronTagInfoProducer",
    primaryVertex = cms.InputTag("offlinePrimaryVertices"),
    jets = cms.InputTag("ak4PFJetsCHS"),
    electrons = cms.InputTag("gedGsfElectrons"),
    DeltaRElectronJet=cms.double(0.4),
    MaxSip3Dsig=cms.double(200)
)

from Configuration.Eras.Modifier_pp_on_AA_2018_cff import pp_on_AA_2018
from Configuration.Eras.Modifier_pp_on_PbPb_run3_cff import pp_on_PbPb_run3
(pp_on_AA_2018 | pp_on_PbPb_run3).toModify(softPFElectronsTagInfos, jets = "akCs4PFJets")
