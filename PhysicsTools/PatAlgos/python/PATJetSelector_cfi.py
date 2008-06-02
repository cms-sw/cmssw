import FWCore.ParameterSet.Config as cms

# module to select Jets
# See https://twiki.cern.ch/twiki/bin/view/CMS/SWGuidePhysicsCutParser
# on how to use the cut-string
selectedLayer1Jets = cms.EDFilter("PATJetSelector",
    src = cms.InputTag("allLayer1Jets"),
    cut = cms.string('et > 15. & abs(eta) < 2.4 & nConstituents > 0')
)


