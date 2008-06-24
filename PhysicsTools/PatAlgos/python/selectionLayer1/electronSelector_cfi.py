import FWCore.ParameterSet.Config as cms

# module to select Electrons
# See https://twiki.cern.ch/twiki/bin/view/CMS/SWGuidePhysicsCutParser
# on how to use the cut-string
#
# These are only dummy cuts
selectedLayer1Electrons = cms.EDFilter("PATElectronSelector",
    src = cms.InputTag("allLayer1Electrons"),
    cut = cms.string('pt > 0. & abs(eta) < 12.')
)


