import FWCore.ParameterSet.Config as cms

centralityFilter = cms.EDFilter("CentralityFilter",
                                selectedBins = cms.vint32(0),
                                BinLabel = cms.InputTag("centralityBin","HFtowers")
                                )

