import FWCore.ParameterSet.Config as cms

hltL1P2GTTau = cms.EDFilter("HLTP2GTTauFilter",
    maxAbsEta = cms.double(1e+99),
    minN = cms.uint32(1),
    minPt = cms.double(5.0),
    l1GTAlgoBlockTag = cms.InputTag("l1tGTAlgoBlockProducer"),
    l1GTAlgoNames = cms.vstring("pDoublePuppiTau52_52"),
    saveTags = cms.bool(True)
)
