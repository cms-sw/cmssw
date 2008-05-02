import FWCore.ParameterSet.Config as cms

from HLTrigger.btau.jetTag.jetReco_cff import *
hltBLifetimeHighestEtJets = cms.EDFilter("LargestEtCaloJetSelector",
    filter = cms.bool(False),
    src = cms.InputTag("iterativeCone5CaloJets"),
    maxNumber = cms.uint32(4)
)

hltBLifetimeL25Jets = cms.EDFilter("EtMinCaloJetSelector",
    filter = cms.bool(False),
    src = cms.InputTag("hltBLifetimeHighestEtJets"),
    etMin = cms.double(35.0)
)

hltBLifetimeL25jetselection = cms.Sequence(hltBLifetimeHighestEtJets*hltBLifetimeL25Jets)

