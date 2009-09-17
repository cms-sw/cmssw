import FWCore.ParameterSet.Config as cms

# module to select Taus
# See https://twiki.cern.ch/twiki/bin/view/CMS/SWGuidePhysicsCutParser
# on how to use the cut-string
#
selectedLayer1Taus = cms.EDFilter("PATTauSelector",
    src = cms.InputTag("allLayer1Taus"),
    cut = cms.string("")
)


