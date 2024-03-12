import FWCore.ParameterSet.Config as cms

# module to select Electrons
# See https://twiki.cern.ch/twiki/bin/view/CMS/SWGuidePhysicsCutParser
# on how to use the cut-string
#
selectedPatElectrons = cms.EDFilter("PATElectronSelector",
    src = cms.InputTag("patElectrons"),
    cut = cms.string("")
)


# foo bar baz
# 91thOtYd5MXnp
# joCcuiOFp0cmg
