import FWCore.ParameterSet.Config as cms

# module to select Photons
# See https://twiki.cern.ch/twiki/bin/view/CMS/SWGuidePhysicsCutParser
# on how to use the cut-string
#
selectedPatPhotons = cms.EDFilter("PATPhotonSelector",
    src = cms.InputTag("patPhotons"),
    cut = cms.string("")
)


