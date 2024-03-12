import FWCore.ParameterSet.Config as cms

dturospacker = cms.EDProducer("DTuROSDigiToRaw",
                                digiColl = cms.InputTag("simMuonDTDigis"),
                                debug = cms.untracked.bool(True),
                             )
# foo bar baz
# QeHzfWSyPIR28
# kj2ZqOM7tr5JY
