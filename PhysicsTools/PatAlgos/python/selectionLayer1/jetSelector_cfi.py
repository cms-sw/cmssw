import FWCore.ParameterSet.Config as cms

# module to select Jets
# See https://twiki.cern.ch/twiki/bin/view/CMS/SWGuidePhysicsCutParser
# on how to use the cut-string
#
selectedPatJets = cms.EDFilter("PATJetSelector",
    src = cms.InputTag("patJets"),
    cut = cms.string(""),    
    cutLoose = cms.string(""),
    nLoose = cms.uint32(0),
)


