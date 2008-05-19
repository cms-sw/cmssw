import FWCore.ParameterSet.Config as cms

from HLTrigger.btau.jetTag.jetReco_cff import *
# Select jets used for regional tracking & considered for b tagging
hltBSoftmuonHighestEtJets = cms.EDFilter("LargestEtCaloJetSelector",
    filter = cms.bool(False),
    src = cms.InputTag("hltIterativeCone5CaloJets"),
    maxNumber = cms.uint32(2) ## Take only 2 highest Et jets...

)

hltBSoftmuonL25Jets = cms.EDFilter("EtMinCaloJetSelector",
    filter = cms.bool(False),
    src = cms.InputTag("hltBSoftmuonHighestEtJets"),
    etMin = cms.double(20.0) ## ...and require jet Et greater than cut.

)

hltBSoftmuonL25jetselection = cms.Sequence(hltBSoftmuonHighestEtJets*hltBSoftmuonL25Jets)

