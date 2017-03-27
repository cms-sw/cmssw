import FWCore.ParameterSet.Config as cms

ME0ChamberMasker = cms.EDProducer('ME0ChamberMasker',
                                 digiTag = cms.InputTag('simMuonME0Digis'),
                                 ME0Minus = cms.bool(False),
                                 ME0Plus = cms.bool(False),
)
