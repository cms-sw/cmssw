import FWCore.ParameterSet.Config as cms

slimmedCaloJets = cms.EDProducer("CaloJetSlimmer",
    src = cms.InputTag("ak4CaloJets"),
    cut = cms.string("pt>20")
)

from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA
pp_on_AA.toModify(slimmedCaloJets, src = 'akPu4CaloJets') 
