import FWCore.ParameterSet.Config as cms

# module to select Muons
# See https://twiki.cern.ch/twiki/bin/view/CMS/SWGuidePhysicsCutParser
# on how to use the cut-string
#
# These are only dummy cuts
selectedLayer1Muons = cms.EDFilter("PATMuonSelector",
    src = cms.InputTag("allLayer1Muons"),
    cut = cms.string('pt > 0. & abs(eta) < 12.')
)


