import FWCore.ParameterSet.Config as cms

dturospacker = cms.EDProducer("DTuROSDigiToRaw",
                                DTDigi_Source = cms.InputTag("simMuonDTDigis"),
                                debug = cms.untracked.bool(True),
                             )
