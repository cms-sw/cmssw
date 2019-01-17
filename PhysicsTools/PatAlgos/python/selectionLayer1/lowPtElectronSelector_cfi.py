import FWCore.ParameterSet.Config as cms

# module to select Electrons
# See https://twiki.cern.ch/twiki/bin/view/CMS/SWGuidePhysicsCutParser
# on how to use the cut-string
#
selectedPatLowPtElectrons = cms.EDFilter("PATElectronSelector",
    src = cms.InputTag("patLowPtElectrons"),
    cut = cms.string("")
)


