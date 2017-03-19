import FWCore.ParameterSet.Config as cms

RPCChamberMasker = cms.EDProducer('RPCChamberMasker',
                                 digiTag = cms.InputTag('preRPCDigis'),
                                 maskedRPCIDs = cms.vint32()
)
