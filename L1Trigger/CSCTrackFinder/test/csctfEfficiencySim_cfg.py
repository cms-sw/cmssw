import FWCore.ParameterSet.Config as cms
import sys


print("Starting CSCTF Efficiency Analyzer")
process = cms.Process("CSCTFEFF")
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1000

#*****************************************************************************************************************************
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10000) )
#*****************************************************************************************************************************

fileOutName = "EffSimHists.root"

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring()
)

process.source.fileNames.extend([
"file:PtGun_cfi_py_GEN_SIM_DIGI.root"
])

# Event Setup
##############
process.load("Configuration.StandardSequences.Services_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.Simulation_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag ='START43_V4::All'

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
  type_of_data = cms.untracked.int32(0),
  inputTag = cms.untracked.InputTag("simCsctfTrackDigis"),
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
  NoRefTracks = cms.untracked.bool(False),
  StatsFilename = cms.untracked.string("/dev/null"),
  PtEffStatsFilename = cms.untracked.string("/dev/null")
)

#                     Data Type Key
#------------------------------------------------------------
# Num  |       Name         | track source | Mode info? 
#------------------------------------------------------------
#  0  | L1CSCTrack          | CSCs         | yes
#  1  | L1MuRegionalCand    | CSCs         | no
#  2  | L1MuGMTExtendedCand | GMT          | no
#process.cscTFEfficiency.type_of_data = 0
#process.cscTFEfficiency.inputTag = cms.untracked.InputTag("simCsctfTrackDigis")
#process.cscTFEfficiency.type_of_data = 1
#process.cscTFEfficiency.inputTag = cms.untracked.InputTag("simCsctfDigis","CSC")
#process.cscTFEfficiency.type_of_data = 2
#process.cscTFEfficiency.inputTag = cms.untracked.InputTag("simGmtDigis")

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
	fileName = cms.untracked.string("testEff.root"),
	outputCommands = cms.untracked.vstring(	
		"keep *"
	)
)

process.p = cms.Path(process.simCscTriggerPrimitiveDigis*process.simDtTriggerPrimitiveDigis*process.simCsctfTrackDigis*process.simCsctfDigis*process.cscTFEfficiency)

#to create testEfff.root
#process.outpath = cms.EndPath(process.FEVT)


