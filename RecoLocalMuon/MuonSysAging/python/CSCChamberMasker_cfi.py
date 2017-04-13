
import FWCore.ParameterSet.Config as cms

CSCChamberMasker = cms.EDProducer('CSCChamberMasker',
        stripDigiTag = cms.InputTag("simMuonCSCDigis","MuonCSCStripDigi"),
        wireDigiTag = cms.InputTag("simMuonCSCDigis","MuonCSCWireDigi"),
        comparatorDigiTag = cms.InputTag("simMuonCSCDigis","MuonCSCComparatorDigi"),
        rpcDigiTag = cms.InputTag("simMuonCSCDigis","MuonCSCRPCDigi"),
        alctDigiTag = cms.InputTag("simMuonCSCDigis","MuonCSCALCTDigi"),
        clctDigiTag = cms.InputTag("simMuonCSCDigis","MuonCSCCLCTDigi"),
        )
