import FWCore.ParameterSet.Config as cms 
process = cms.Process("ANA")
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = cms.untracked.string('INFO')

#input files
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(50000) )
#process.MessageLogger.cerr.FwkReport.reportEvery = 500

process.source = cms.Source("PoolSource",
	fileNames = cms.untracked.vstring(
		'/store/data/Run2010B/Mu/RAW/v1/000/147/222/16609CFA-4AD0-DF11-B663-0030487CD6B4.root',
		'/store/data/Run2010B/Mu/RAW/v1/000/147/222/267120BC-52D0-DF11-939B-0030487C635A.root',
		'/store/data/Run2010B/Mu/RAW/v1/000/147/222/288A5325-41D0-DF11-8C65-000423D996C8.root',
		'/store/data/Run2010B/Mu/RAW/v1/000/147/222/2C8A9DA5-5ED0-DF11-8CE6-000423D94494.root',
		'/store/data/Run2010B/Mu/RAW/v1/000/147/222/341E87BC-3FD0-DF11-B89B-0030487CD7B4.root',
		'/store/data/Run2010B/Mu/RAW/v1/000/147/222/366546E2-48D0-DF11-893B-000423D987E0.root',
		'/store/data/Run2010B/Mu/RAW/v1/000/147/222/420ADFBB-52D0-DF11-81FB-0030487A1884.root',
		'/store/data/Run2010B/Mu/RAW/v1/000/147/222/66854D3C-56D0-DF11-8320-0030487CAF0E.root',
		'/store/data/Run2010B/Mu/RAW/v1/000/147/222/6EEB6090-49D0-DF11-AD74-0030487C8CB8.root',
		'/store/data/Run2010B/Mu/RAW/v1/000/147/222/72F547C6-46D0-DF11-9E03-00304879EDEA.root',
		'/store/data/Run2010B/Mu/RAW/v1/000/147/222/74EA8337-4DD0-DF11-9B4F-0019B9F70607.root',
		'/store/data/Run2010B/Mu/RAW/v1/000/147/222/9A34859A-50D0-DF11-A30E-003048F118E0.root',
		'/store/data/Run2010B/Mu/RAW/v1/000/147/222/9C66DF41-43D0-DF11-B5B0-0030487C6A66.root',
		'/store/data/Run2010B/Mu/RAW/v1/000/147/222/D4698BD0-54D0-DF11-8348-0030487CD6D2.root',
		'/store/data/Run2010B/Mu/RAW/v1/000/147/222/E09B7D79-4ED0-DF11-8CE6-0030487CD716.root',
		'/store/data/Run2010B/Mu/RAW/v1/000/147/222/E2A45EA6-44D0-DF11-9DFB-0030487CD7CA.root'
	
		#'/store/data/Run2010A/Mu/RAW/v1/000/141/881/10FA9D74-DA9A-DF11-B5F9-0030487CD812.root',
		#'/store/data/Run2010A/Mu/RAW/v1/000/141/881/2CC453A6-D09A-DF11-BAB6-0030487A18D8.root',
		#'/store/data/Run2010A/Mu/RAW/v1/000/141/881/3043E4AB-D59A-DF11-B795-001617E30CC2.root',
		#'/store/data/Run2010A/Mu/RAW/v1/000/141/881/B628B0FC-CA9A-DF11-9DEF-0030487A3C92.root',
		#'/store/data/Run2010A/Mu/RAW/v1/000/141/881/D657B0F6-E59A-DF11-87B1-001D09F2AD7F.root',
		#'/store/data/Run2010A/Mu/RAW/v1/000/141/881/DC645B02-E59A-DF11-BFD0-001617E30E28.root',
		#'/store/data/Run2010A/Mu/RAW/v1/000/141/881/E68BE446-C39A-DF11-97BC-0030487CD180.root',
		#'/store/data/Run2010A/Mu/RAW/v1/000/141/881/F04CD612-C79A-DF11-A46E-001D09F29169.root',
		#'/store/data/Run2010A/Mu/RAW/v1/000/141/881/F88D8FAD-DE9A-DF11-8984-001D09F2AF1E.root'
	
		#'/store/data/Run2010A/Mu/RAW/v1/000/139/980/0E203E97-A18D-DF11-99E0-001D09F244DE.root',
		#'/store/data/Run2010A/Mu/RAW/v1/000/139/980/14FCABCC-A38D-DF11-AAA3-000423D94908.root',
		#'/store/data/Run2010A/Mu/RAW/v1/000/139/980/28963EC8-A38D-DF11-9767-00304879FC6C.root',
		#'/store/data/Run2010A/Mu/RAW/v1/000/139/980/404556FB-9B8D-DF11-9FA2-0030487CD162.root',
		#'/store/data/Run2010A/Mu/RAW/v1/000/139/980/8E98AA82-A68D-DF11-8022-001D09F2AD84.root',
		#'/store/data/Run2010A/Mu/RAW/v1/000/139/980/A0BC5181-9D8D-DF11-8946-0030487C7E18.root',
		#'/store/data/Run2010A/Mu/RAW/v1/000/139/980/AEA205B3-AA8D-DF11-80FE-003048F1C832.root',
		#'/store/data/Run2010A/Mu/RAW/v1/000/139/980/B02AAC80-9D8D-DF11-8647-0030487CD840.root',
		#'/store/data/Run2010A/Mu/RAW/v1/000/139/980/CC31444A-A08D-DF11-BD19-001D09F28EC1.root',
		#'/store/data/Run2010A/Mu/RAW/v1/000/139/980/D24AB749-A08D-DF11-9265-0030487C7E18.root',
		#'/store/data/Run2010A/Mu/RAW/v1/000/139/980/E4957782-9F8D-DF11-BA7B-001D09F2423B.root',
		#'/store/data/Run2010A/Mu/RAW/v1/000/139/980/F8B3DD0E-A38D-DF11-8DE7-003048F024C2.root'
		
	)#,
#	eventsToProcess = cms.untracked.VEventRange(
#		'147222:210044653','147222:209976376','147222:209506441',
#		'147222:210347427','147222:211092472',
#		'147222:299145451','147222:302040379',
#		'147222:302803126','147222:304335179','147222:305209888',
#		'147222:305240816','147222:306925924','147222:307099760',
#		'147222:306974089','147222:308727818','147222:309723179',
#		'147222:310551819','147222:311153378','147222:311955219',
#		'147222:312396726','147222:312641904','147222:313905427',
#		'147222:313649869','147222:315422096','147222:315360569',
#		'147222:316712592','147222:318106692','147222:321029107','147222:320871767',
#		'147222:372985610','147222:374142447','147222:374468716','147222:385982861',
#		'147222:386415260','147222:387399209','147222:388331373','147222:389358670',
#		'147222:389621756','147222:389697470','147222:393447336'
#	)

)

process.load("DQMServices.Core.DQM_cfg")
process.load("CondCore.DBCommon.CondDBSetup_cfi")
process.load("EventFilter.CSCRawToDigi.cscUnpacker_cfi")
process.load("EventFilter.CSCTFRawToDigi.csctfunpacker_cfi")
process.load("EventFilter.DTTFRawToDigi.dttfunpacker_cfi")
#process.load("Configuration.StandardSequences.RawToDigi_Data_cff")
# include files needed to constuct CSC Geometry
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = ('GR10_P_V10::All')

# CSC Emulator on data
import L1Trigger.CSCTrackFinder.csctfTrackDigis_cfi
process.simCsctfTrackDigis = L1Trigger.CSCTrackFinder.csctfTrackDigis_cfi.csctfTrackDigis.clone()
import L1Trigger.CSCTrackFinder.csctfDigis_cfi
process.simCsctfTrackDigis.SectorReceiverInput = cms.untracked.InputTag("csctfunpacker")
process.simCsctfTrackDigis.DTproducer = cms.untracked.InputTag("dttfunpacker")
#process.simCsctfTrackDigis.SectorProcessor.initializeFromPSet = cms.bool(True)
#process.simCsctfTrackDigis.SectorProcessor.firmwareSP = cms.uint32(20100629)
#process.simCsctfTrackDigis.SectorProcessor.firmwareSP = cms.uint32(20100210)
#process.simCsctfTrackDigis.SectorProcessor.firmwareFA = cms.uint32(20090521)
#process.simCsctfTrackDigis.SectorProcessor.firmwareDD = cms.uint32(20090521)
#process.simCsctfTrackDigis.SectorProcessor.firmwareVM = cms.uint32(20090521)
#process.simCsctfTrackDigis.SectorProcessor.trigger_on_ME1a = cms.bool(True)
#process.simCsctfTrackDigis.SectorProcessor.trigger_on_ME1b = cms.bool(True)
process.simCsctfTrackDigis.readDtDirect = cms.bool(True)

# analysis & output
process.FEVT = cms.OutputModule("PoolOutputModule",
	fileName = cms.untracked.string("DataStructure.root"),
	outputCommands = cms.untracked.vstring("keep *")
)

process.comp = cms.EDAnalyzer("CsctfDatEmu",
	dataTrackProducer = cms.InputTag("csctfunpacker"),
	emulTrackProducer = cms.InputTag("simCsctfTrackDigis"),
	lctProducer	  = cms.InputTag("csctfunpacker"),
	outFile = cms.untracked.string("coreSwitchRun147222.root")
	#outFile = cms.untracked.string("coreSwitchRun139980.root")
)

process.dComp = cms.EDAnalyzer("dtReciever",
	outFile = cms.untracked.string("dtComp.root")
)

process.p = cms.Path(
 process.muonCSCDigis
 *process.csctfunpacker
 *process.dttfunpacker
 *process.simCsctfTrackDigis
 *process.comp
)
#process.p = cms.Path(process.comp)
#process.outpath = cms.EndPath(process.FEVT)
#process.schedule = cms.Schedule(process.p, process.outpath)
