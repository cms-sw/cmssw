import FWCore.ParameterSet.Config as cms

dimuonsGlobal = cms.EDFilter("CandViewRefSelector",
    src = cms.InputTag("dimuons"),
    cut = cms.string('charge = 0 & mass > 20 & daughter(0).isGlobalMuon = 1 & daughter(1).isGlobalMuon = 1')
)


