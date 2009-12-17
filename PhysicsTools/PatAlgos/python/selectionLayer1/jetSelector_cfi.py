import FWCore.ParameterSet.Config as cms

# module to select Jets
# See https://twiki.cern.ch/twiki/bin/view/CMS/SWGuidePhysicsCutParser
# on how to use the cut-string
#
selectedPatAK5CaloJets = cms.EDFilter("PATJetSelector",
    src = cms.InputTag("patAK5CaloJets"),
    cut = cms.string("")
)


