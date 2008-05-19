import FWCore.ParameterSet.Config as cms

from HLTrigger.btau.jetTag.jetReco_cff import *
# Select jets used for regional tracking & considered for b tagging
hltBLifetimeHighestEtJets = cms.EDFilter("LargestEtCaloJetSelector",
    filter = cms.bool(False),
    src = cms.InputTag("hltIterativeCone5CaloJets"),
    maxNumber = cms.uint32(4) ## Take only 4 highest Et jets...

)

hltBLifetimeL25Jets = cms.EDFilter("EtMinCaloJetSelector",
    filter = cms.bool(False),
    src = cms.InputTag("hltBLifetimeHighestEtJets"),
    etMin = cms.double(35.0) ## ...and require jet Et greater than cut.

)

hltBLifetimeL25jetselection = cms.Sequence(hltBLifetimeHighestEtJets*hltBLifetimeL25Jets)

