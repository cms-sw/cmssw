import FWCore.ParameterSet.Config as cms

pfElectronsPtGt5 = cms.EDFilter("PtMinPFCandidateSelector",
    src = cms.InputTag("pfAllElectrons"),
    ptMin = cms.double(5.0)
)




# foo bar baz
# 9AF2JE3Dv9aPm
# 3M3q5HDxyok11
