import FWCore.ParameterSet.Config as cms
import sys


print("Starting CSCTF Data Analyzer")
process = cms.Process("CSCTFEFF")
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1000

#*****************************************************************************************************************************
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10000) )
#*****************************************************************************************************************************

fileOutName = "AnaDataHists.root"

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring()
)

process.source.fileNames.extend([
#High PU Run
"/store/data/Run2011B/L1MuHPF/RAW/v1/000/178/208/FEAA124F-7EF3-E011-9208-003048673374.root",
"/store/data/Run2011B/L1MuHPF/RAW/v1/000/178/208/FE03EB11-5DF3-E011-B717-003048F117EC.root",
"/store/data/Run2011B/L1MuHPF/RAW/v1/000/178/208/FAB7E40D-76F3-E011-8566-BCAEC53296FF.root",
"/store/data/Run2011B/L1MuHPF/RAW/v1/000/178/208/FA1E6192-81F3-E011-B876-BCAEC53296FF.root",
"/store/data/Run2011B/L1MuHPF/RAW/v1/000/178/208/FA085011-7FF3-E011-9DE3-003048D2BC4C.root",
"/store/data/Run2011B/L1MuHPF/RAW/v1/000/178/208/F8A8B8E4-86F3-E011-BC5A-BCAEC53296F7.root",
"/store/data/Run2011B/L1MuHPF/RAW/v1/000/178/208/F8A32D30-80F3-E011-9CCE-001D09F24489.root",
"/store/data/Run2011B/L1MuHPF/RAW/v1/000/178/208/F86797FF-7DF3-E011-AC73-003048F024FE.root",
"/store/data/Run2011B/L1MuHPF/RAW/v1/000/178/208/F6C69A11-7FF3-E011-A256-001D09F24EAC.root",
"/store/data/Run2011B/L1MuHPF/RAW/v1/000/178/208/F62BF6A4-7BF3-E011-8A47-E0CB4E4408E3.root",
"/store/data/Run2011B/L1MuHPF/RAW/v1/000/178/208/F491E355-60F3-E011-9860-0030486780E6.root"
])

# Event Setup
##############
process.load("Configuration.StandardSequences.Services_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.GeometryDB_cff")
process.load("EventFilter.CSCTFRawToDigi.csctfunpacker_cfi")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag ='GR_R_43_V3'

# L1 Emulator
#############PtEffStatsFilename
process.load("Configuration.StandardSequences.SimL1Emulator_cff")


#prints out Alex's Firmware debugging output
#process.simCsctfTrackDigis.SectorProcessor.isCoreVerbose = True

#process.simCsctfTrackDigis.SectorProcessor.initializeFromPSet = True

#Configurable options when PSet is True
#process.simCsctfTrackDigis.SectorProcessor.mindetap = cms.uint32(4)
#process.simCsctfTrackDigis.SectorProcessor.mindphip = cms.uint32(128)
#process.simCsctfTrackDigis.SectorProcessor.straightp = cms.uint32(60)
#process.simCsctfTrackDigis.SectorProcessor.curvedp = cms.uint32(200)
#process.simCsctfTrackDigis.SectorProcessor.firmwareSP = cms.uint32(20110204)
#process.simCsctfTrackDigis.SectorProcessor.EtaWindows = cms.vuint32(4,4,4,4,4,4,4)


# CSCTFEfficiency Analyzer
# defualt values
process.cscTFEfficiency = cms.EDAnalyzer('CSCTFEfficiency',
  inputTag = cms.untracked.InputTag("csctfunpacker"),
  MinPtSim = cms.untracked.double(2.0),
  MaxPtSim = cms.untracked.double(500.0),
  MinEtaSim = cms.untracked.double(0.9),
  MaxEtaSim = cms.untracked.double(2.4),
  MinPtTF = cms.untracked.double(-1),
  MinQualityTF = cms.untracked.double(2),
  GhostLoseParam = cms.untracked.string("Q"),
  InputData = cms.untracked.bool(False),
  MinMatchR = cms.untracked.double(0.5),     
  MinPtHist = cms.untracked.double(-0.5),                           
  MaxPtHist = cms.untracked.double(140.5),
  BinsPtHist = cms.untracked.double(70),
  SaveHistImages = cms.untracked.bool(False),
  SingleMuSample = cms.untracked.bool(False),
  NoRefTracks = cms.untracked.bool(True),
  StatsFilename = cms.untracked.string("/dev/null"),
  PtEffStatsFilename = cms.untracked.string("/dev/null"),
  type_of_data = cms.untracked.int32(0)
)

#                     Data Type Key
#------------------------------------------------------------
# Num  |       Name         | track source | Mode info? 
#------------------------------------------------------------
#  0  | L1CSCTrack          | CSCs         | yes
#  1  | L1MuRegionalCand    | CSCs         | no
#  2  | L1MuGMTExtendedCand | GMT          | no
#process.cscTFEfficiency.type_of_data = 0

#=Controls the cut values for TF track selection

#process.cscTFEfficiency.MinQualityTF = 2

#=Use False to run Simulated Data or True for Real Data
#process.cscTFEfficiency.InputData = True

#=Controls the maximum R value for matching
#process.cscTFEfficiency.MaxMatchR = 0.5

#=Controls minimum value on x-axis of PT Hist
#process.cscTFEfficiency.MinPtHist = -0.5

#=Controls maximum value on x-axis of PT Hist
#process.cscTFEfficiency.MaxPtHist = 20.5

#=Controls the number of bins used to create the PT Hist
#process.cscTFEfficiency.BinsPtHist = 21

#=Controls the name of the statistics file output
#process.cscTFEfficiency.StatsFilename = statName

#=Controls output of validation histogram images:
#process.cscTFEfficiency.SaveHistImages = False

#=Controls Ghost Validation Counting (Default False):
#process.cscTFEfficiency.SingleMuSample = True 

#=Controls ghost selection method. Use quality "Q" or match value "R" as metric 
	#=Best candidate is considered real track, others considered ghosts. 
	#=Default Q
#process.cscTFEfficiency.GhostLoseParam = "R"

#=Controls the name of the output file for the Pt Efficiency Stats
#process.cscTFEfficiency.PtEffStatsFilename = PtEffStatsName

process.TFileService = cms.Service("TFileService",
	fileName = cms.string(
		fileOutName
))

process.FEVT = cms.OutputModule("PoolOutputModule",
	fileName = cms.untracked.string("testAnaData.root"),
	outputCommands = cms.untracked.vstring(	
		"keep *"
	)
)

process.p = cms.Path(process.csctfunpacker*process.cscTFEfficiency)

#to create testEfff.root
#process.outpath = cms.EndPath(process.FEVT)


