import FWCore.ParameterSet.Config as cms

ME0ChamberMasker = cms.EDProducer('ME0ChamberMasker',
                                 digiTag  = cms.InputTag('simMuonME0Digis'),
                                 me0Minus = cms.bool(True),
                                 me0Plus  = cms.bool(True),
)
