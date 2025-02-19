import FWCore.ParameterSet.Config as cms

# module to select met
# See https://twiki.cern.ch/twiki/bin/view/CMS/SWGuidePhysicsCutParser
# on how to use the cut-string
#
selectedPatMET = cms.EDFilter("PATMETSelector",
    src = cms.InputTag("patMETs"),
    cut = cms.string("")
)


