import FWCore.ParameterSet.Config as cms

# module to select Electrons
# See https://twiki.cern.ch/twiki/bin/view/CMS/SWGuidePhysicsCutParser
# on how to use the cut-string
selectedLayer1Electrons = cms.EDFilter("PATElectronSelector",
    src = cms.InputTag("allLayer1Electrons"),
    cut = cms.string('pt > 10. & abs(eta) < 2.4')
)


