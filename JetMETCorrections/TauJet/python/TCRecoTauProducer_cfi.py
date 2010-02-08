import FWCore.ParameterSet.Config as cms

tcRecoTauProducer = cms.EDProducer("TCRecoTauProducer",
	CaloRecoTauProducer = cms.InputTag("caloRecoTauProducer"),
        ## TauJet jet energy correction parameters
        src            = cms.InputTag("iterativeCone5CaloJets"),
        tagName        = cms.string("IterativeCone0.4_EtScheme_TowerEt0.5_E0.8_Jets871_2x1033PU_tau"),
        TauTriggerType = cms.int32(1)
)


