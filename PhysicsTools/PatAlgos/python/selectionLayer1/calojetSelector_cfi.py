import FWCore.ParameterSet.Config as cms

# module to select Jets
# See https://twiki.cern.ch/twiki/bin/view/CMS/SWGuidePhysicsCutParser
# on how to use the cut-string
#
selectedCaloJets = cms.EDFilter("SlimCaloJetSelector",
    src = cms.InputTag("ak4CaloJets"),
    cut = cms.string("pt>10")
)


