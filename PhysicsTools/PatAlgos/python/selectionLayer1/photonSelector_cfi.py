import FWCore.ParameterSet.Config as cms

# module to select Photons
# See https://twiki.cern.ch/twiki/bin/view/CMS/SWGuidePhysicsCutParser
# on how to use the cut-string
#
# These are only dummy cuts
selectedLayer1Photons = cms.EDFilter("PATPhotonSelector",
    src = cms.InputTag("allLayer1Photons"),
    cut = cms.string('pt > 0. & abs(eta) < 12.')
)


