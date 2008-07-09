import FWCore.ParameterSet.Config as cms

from HLTrigger.btau.jetTag.jetReco_cff import *
hltBSoftmuonHighestEtJets = cms.EDFilter("LargestEtCaloJetSelector",
    filter = cms.bool(False),
    src = cms.InputTag("iterativeCone5CaloJets"),
    maxNumber = cms.uint32(2)
)

hltBSoftmuonL25Jets = cms.EDFilter("EtMinCaloJetSelector",
    filter = cms.bool(False),
    src = cms.InputTag("hltBSoftmuonHighestEtJets"),
    etMin = cms.double(20.0)
)

hltBSoftmuonL25jetselection = cms.Sequence(hltBSoftmuonHighestEtJets*hltBSoftmuonL25Jets)

