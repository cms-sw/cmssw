import FWCore.ParameterSet.Config as cms

GEMChamberMasker = cms.EDProducer('GEMChamberMasker',
                                 digiTag = cms.InputTag('simMuonGEMDigis'),
                                 GE11Minus = cms.bool(True),
                                 GE11Plus = cms.bool(True),
                                 GE21Minus = cms.bool(True),
                                 GE21Plus = cms.bool(True),
                                 maskedGEMIDs = cms.vint32()
)
