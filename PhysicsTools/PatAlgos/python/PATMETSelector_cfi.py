import FWCore.ParameterSet.Config as cms

# module to select METs
# See https://twiki.cern.ch/twiki/bin/view/CMS/SWGuidePhysicsCutParser
# on how to use the cut-string
selectedLayer1METs = cms.EDFilter("PATMETSelector",
    src = cms.InputTag("allLayer1METs"),
    cut = cms.string('et > 0.')
)


