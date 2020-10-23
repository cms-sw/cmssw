import FWCore.ParameterSet.Config as cms

slimmedCaloJets = cms.EDProducer("CaloJetSlimmer",
    src = cms.InputTag("ak4CaloJets"),
    cut = cms.string("pt>20")
)

from Configuration.Eras.Modifier_pp_on_AA_2018_cff import pp_on_AA_2018
from Configuration.Eras.Modifier_pp_on_PbPb_run3_cff import pp_on_PbPb_run3
(pp_on_AA_2018 | pp_on_PbPb_run3).toModify(slimmedCaloJets, src = 'akPu4CaloJets') 
