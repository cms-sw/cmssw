import FWCore.ParameterSet.Config as cms

GEMChamberMasker = cms.EDProducer('GEMChamberMasker',
                                 digiTag = cms.InputTag('simMuonGEMDigis'),
                                 GE11Minus = cms.bool(False),
                                 GE11Plus = cms.bool(False),
                                 GE21Minus = cms.bool(False),
                                 GE21Plus = cms.bool(False),
)
