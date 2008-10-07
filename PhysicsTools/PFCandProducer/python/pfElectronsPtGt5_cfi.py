import FWCore.ParameterSet.Config as cms

pfElectronsPtGt5 = cms.EDProducer("PtMinPFCandidateSelector",
    src = cms.InputTag("pfAllElectrons"),
    ptMin = cms.double(5.0)
)




