import FWCore.ParameterSet.Config as cms

process = cms.Process("Ana")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = ""
process.MessageLogger.cerr.FwkReport.reportEvery = 1000

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
	#High PU Run
       '/store/data/Run2011B/ZeroBiasHPF0/RAW/v1/000/179/828/006295EB-35FF-E011-9F6A-003048F118D2.root',
       '/store/data/Run2011B/ZeroBiasHPF0/RAW/v1/000/179/828/008702AE-31FF-E011-89C5-BCAEC5329717.root',
       '/store/data/Run2011B/ZeroBiasHPF0/RAW/v1/000/179/828/00C475AF-2FFF-E011-9D1A-E0CB4E4408D5.root',
       '/store/data/Run2011B/ZeroBiasHPF0/RAW/v1/000/179/828/00C5D097-40FF-E011-AD8E-003048F024FA.root',
       '/store/data/Run2011B/ZeroBiasHPF0/RAW/v1/000/179/828/02D0731B-2BFF-E011-8B75-003048F11C58.root',
       '/store/data/Run2011B/ZeroBiasHPF0/RAW/v1/000/179/828/02EE63F8-35FF-E011-9B82-0030486780E6.root',
       '/store/data/Run2011B/ZeroBiasHPF0/RAW/v1/000/179/828/02F71ADF-43FF-E011-934C-002481E0D958.root',
       '/store/data/Run2011B/ZeroBiasHPF0/RAW/v1/000/179/828/041E1BF8-33FF-E011-99D6-003048F118D2.root',
       '/store/data/Run2011B/ZeroBiasHPF0/RAW/v1/000/179/828/06374143-2AFF-E011-9D90-003048F01E88.root',
       '/store/data/Run2011B/ZeroBiasHPF0/RAW/v1/000/179/828/081BACB5-2FFF-E011-93EC-003048D37666.root',
       '/store/data/Run2011B/ZeroBiasHPF0/RAW/v1/000/179/828/088937ED-35FF-E011-BA7A-001D09F24D8A.root',
       '/store/data/Run2011B/ZeroBiasHPF0/RAW/v1/000/179/828/0A039101-2EFF-E011-B35E-003048D37580.root',
       '/store/data/Run2011B/ZeroBiasHPF0/RAW/v1/000/179/828/0A8780ED-3CFF-E011-A301-BCAEC532972D.root'
    )
)
# Event Setup
##############
process.load("Configuration.StandardSequences.Services_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag ='GR_R_44_V13::All'
process.load("EventFilter.CSCTFRawToDigi.csctfunpacker_cfi")
process.load("EventFilter.CSCRawToDigi.cscUnpacker_cfi")

# HLT Trigger Filter

process.load("HLTrigger.HLTfilters.hltHighLevel_cfi")
# change algorithms to the HLT trigger(s) you want to filter by
# Examples
# "HLT_L2Mu11","HLT_Mu9","HLT_DoubleMu3","HLT_IsoMu3","HLT_MET100","HLT_Jet50U","HLT_QuadJet15U"
process.hltHighLevel.HLTPaths = cms.vstring("HLT_ZeroBias","HLT_ZeroBias_v1","HLT_ZeroBias_v2","HLT_ZeroBias_v3","HLT_ZeroBias_v4","HLT_ZeroBias_v5","HLT_ZeroBias_part0_v1",'HLT_ZeroBias_part1_v1','HLT_ZeroBias_part2_v1','HLT_ZeroBias_part3_v1')
process.hltHighLevel.eventSetupPathsKey = cms.string("")
process.hltHighLevel.andOr = cms.bool(True) # False Only takes events with all triggers at same time.
process.hltHighLevel.throw = False

# LCT Occupancies Analyzer

process.lctOccupanciesCSC = cms.EDAnalyzer('LCTOccupancies',
	 lctsTag = cms.InputTag("muonCSCDigis","MuonCSCCorrelatedLCTDigi"),
	 vertexColTag = cms.InputTag("offlinePrimaryVertices"),
	 outTreeFileName = cms.untracked.string("LctOccTreeHighPU179828inTimeOnly.root"),
	 haveRECO = cms.untracked.bool(False),
	 singleSectorNum = cms.untracked.int32(-1) #-1 for sum over all sectors
)
process.TFileService = cms.Service("TFileService",
	fileName = cms.string(
		"LCTOccupanciesOutput_data.root"
))

#This path is for Real Data
process.p = cms.Path(process.hltHighLevel*process.csctfunpacker*process.muonCSCDigis*process.lctOccupanciesCSC)
