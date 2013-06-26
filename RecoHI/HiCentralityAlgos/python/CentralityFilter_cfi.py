import FWCore.ParameterSet.Config as cms

centralityFilter = cms.EDFilter("CentralityFilter",
                                centralityBase = cms.string("HF"),
                                selectedBins = cms.vint32(0),
                                src = cms.InputTag("hiCentrality")
                                )

