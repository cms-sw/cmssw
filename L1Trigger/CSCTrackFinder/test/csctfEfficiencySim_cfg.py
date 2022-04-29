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
       '/store/relval/CMSSW_5_2_0_pre4/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/START52_V1-v1/0033/18FF7335-B551-E111-9992-003048FFD7D4.root',
       '/store/relval/CMSSW_5_2_0_pre4/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/START52_V1-v1/0033/1EAEFA31-B451-E111-91D5-002618943869.root',
       '/store/relval/CMSSW_5_2_0_pre4/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/START52_V1-v1/0033/3CCACBDE-BA51-E111-B9CF-00261894398A.root',
       '/store/relval/CMSSW_5_2_0_pre4/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/START52_V1-v1/0033/3E77FF2F-B451-E111-808B-002618FDA210.root',
       '/store/relval/CMSSW_5_2_0_pre4/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/START52_V1-v1/0033/724975B7-B551-E111-B6AE-003048FFD760.root',
       '/store/relval/CMSSW_5_2_0_pre4/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/START52_V1-v1/0033/8CA0ED33-B451-E111-8FAC-002618943964.root',
       '/store/relval/CMSSW_5_2_0_pre4/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/START52_V1-v1/0033/D6338531-B551-E111-AFF9-0026189438A0.root',
       '/store/relval/CMSSW_5_2_0_pre4/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/START52_V1-v1/0033/D6BC8E30-B551-E111-ACA4-00304867908C.root',
       '/store/relval/CMSSW_5_2_0_pre4/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/START52_V1-v1/0033/E6CD4B79-C051-E111-91F6-0018F3D09636.root',
       '/store/relval/CMSSW_5_2_0_pre4/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/START52_V1-v1/0033/F8E9A938-B651-E111-9CDF-0026189438CB.root',
       '/store/relval/CMSSW_5_2_0_pre4/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/START52_V1-v1/0036/1629E91E-F251-E111-B631-001A92810AE0.root'
])

# Event Setup
##############
process.load("Configuration.StandardSequences.Services_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.GeometryDB_cff")
process.load("Configuration.StandardSequences.Simulation_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag ='START52_V1::All'

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
#process.simCsctfTrackDigis.SectorProcessor.PTLUT.PtMethod = 28
#process.simCsctfTrackDigis.SectorProcessor.EtaWindows = cms.vuint32(4,4,4,4,4,4,4)

# CSCTFEfficiency Analyzer
# defualt values
process.cscTFEfficiency = cms.EDAnalyzer('CSCTFEfficiency',
  type_of_data = cms.untracked.int32(0),
  inputTag = cms.untracked.InputTag("simCsctfTrackDigis"),
  MinPtSim = cms.untracked.double(2.0),
  MaxPtSim = cms.untracked.double(100.0),
  MinEtaSim = cms.untracked.double(0.9),
  MaxEtaSim = cms.untracked.double(2.4),
  MinPtTF = cms.untracked.double(-1),
  MinQualityTF = cms.untracked.double(1),
  CutOnModes = cms.untracked.vuint32(),
  GhostLoseParam = cms.untracked.string("Q"),
  InputData = cms.untracked.bool(False),
  MinMatchR = cms.untracked.double(0.5),     
  MinPtHist = cms.untracked.double(-0.5),                           
  MaxPtHist = cms.untracked.double(100.5),
  BinsPtHist = cms.untracked.double(20),
  SaveHistImages = cms.untracked.bool(False),
  SingleMuSample = cms.untracked.bool(False),
  NoRefTracks = cms.untracked.bool(False),
  StatsFilename = cms.untracked.string("/dev/null"),
  PtEffStatsFilename = cms.untracked.string("/dev/null"),
  HistoDescription = cms.untracked.string("")
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

#=Only Allows These Modes, if length=0, then all
#process.cscTFEfficiency.CutOnModes = cms.untracked.vuint32(2,3,4)

#=Adds a Description to the Upper Left of Main Plots (TLatex)
#process.cscTFEfficiency.HistoDescription = "FW/PTLUT 2012"

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

process.p = cms.Path(process.simCsctfTrackDigis*process.simCsctfDigis*process.cscTFEfficiency)
#process.p = cms.Path(process.simCscTriggerPrimitiveDigis*process.simDtTriggerPrimitiveDigis*process.simCsctfTrackDigis*process.simCsctfDigis*process.cscTFEfficiency)

#to create testEfff.root
#process.outpath = cms.EndPath(process.FEVT)


