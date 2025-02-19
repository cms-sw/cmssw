import FWCore.ParameterSet.Config as cms
 
dumpCSCdigi = cms.EDAnalyzer("CSCViewDigi",
               wireDigiTag = cms.InputTag("muonCSCDigis","MuonCSCWireDigi"),
	       stripDigiTag = cms.InputTag("muonCSCDigis","MuonCSCStripDigi"),
	       comparatorDigiTag = cms.InputTag("muonCSCDigis","MuonCSCComparatorDigi"),
	       rpcDigiTag = cms.InputTag("muonCSCDigis","MuonCSCRPCDigi"),
               alctDigiTag = cms.InputTag("muonCSCDigis","MuonCSCALCTDigi"),
               clctDigiTag = cms.InputTag("muonCSCDigis","MuonCSCCLCTDigi"),
               corrclctDigiTag = cms.InputTag("muonCSCDigis","MuonCSCCorrelatedLCTDigi"),
	       statusDigiTag = cms.InputTag("muonCSCDigis","MuonCSCDCCFormatStatusDigi"),
	       statusCFEBTag = cms.InputTag("muonCSCDigis","MuonCSCCFEBStatusDigi"),
	       DDUstatusDigiTag = cms.InputTag("muonCSCDigis","MuonCSCDDUStatusDigi"),
	       DCCstatusDigiTag = cms.InputTag("muonCSCDigis","MuonCSCDCCStatusDigi"),
	       	
               WiresDigiDump = cms.untracked.bool(True),
	       StripDigiDump = cms.untracked.bool(True),
	       ComparatorDigiDump = cms.untracked.bool(True),
	       RpcDigiDump = cms.untracked.bool(True),
               AlctDigiDump = cms.untracked.bool(True),
               ClctDigiDump = cms.untracked.bool(True),
               CorrClctDigiDump = cms.untracked.bool(True),
	       StatusCFEBDump = cms.untracked.bool(True),
	       StatusDigiDump = cms.untracked.bool(False),
	       DDUStatus = cms.untracked.bool(True),
	       DCCStatus = cms.untracked.bool(True)	
)