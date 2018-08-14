import FWCore.ParameterSet.Config as cms

dturospacker = cms.EDProducer("DTuROSDigiToRaw",
                                digiColl = cms.InputTag("simMuonDTDigis"),
                                debug = cms.untracked.bool(True),
                             )
