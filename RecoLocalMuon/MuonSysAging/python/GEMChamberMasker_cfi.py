import FWCore.ParameterSet.Config as cms

GEMChamberMasker = cms.EDProducer('GEMChamberMasker',
                                 digiTag   = cms.InputTag('simMuonGEMDigis'),
                                 ge11Minus = cms.bool(True),
                                 ge11Plus  = cms.bool(True),
                                 ge21Minus = cms.bool(True),
                                 ge21Plus  = cms.bool(True),
)
