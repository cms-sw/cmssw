import FWCore.ParameterSet.Config as cms

process = cms.Process("MyRawToDigi")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
#process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.Geometry.GeometryExtended2017Reco_cff')
#process.load('Configuration.Geometry.GeometryExtended2017NewFPixReco_cff')

#process.load("Configuration.StandardSequences.MagneticField_38T_cff")
#process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')
process.load("Configuration.StandardSequences.Services_cff")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
# to use no All 
# 2015
#process.GlobalTag.globaltag = 'GR_P_V56' # for 247607
#process.GlobalTag.globaltag = 'PRE_R_71_V3' #2014

#2017
#process.GlobalTag.globaltag='81X_upgrade2017_realistic_v26'
# AUTO conditions 
#process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_data', '')
#process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run1_data', '')
#process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')
#process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_design', '')
#process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgrade2017', '')
#process.GlobalTag = GlobalTag(process.GlobalTag, '76X_upgrade2017_design_v8', '')
#process.GlobalTag.globaltag ="81X_dataRun2_relval_v14"
#process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2017_realistic', '')
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2018_realistic', '')
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(128))

process.source = cms.Source("PoolSource",
fileNames =  cms.untracked.vstring(
# 'file:/afs/cern.ch/work/d/dkotlins/public/MC/mu_phase1/pt100_81/raw/raw1_formatfix.root'
#'root://cms-xrd-global.cern.ch//store/relval/CMSSW_8_1_0/RelValMinBias_13/GEN-SIM-DIGI-RAW/81X_upgrade2017_realistic_v26_HLT2017Trk-v1/10000/06A2997E-3BC1-E611-B286-0CC47A78A30E.root'
#2018 CMSSW_9_4_0
'/store/relval/CMSSW_9_4_0/RelValTTbar_13/GEN-SIM-DIGI-RAW/PU25ns_94X_upgrade2018_realistic_v5-v1/10000/F87005CD-CBC8-E711-A9F5-0CC47A4D7694.root'

#2017 CMSSW_9_2_0
# download this file
# /store/relval/CMSSW_9_2_0/RelValTTbar_13/GEN-SIM-DIGI-RAW/PU25ns_91X_upgrade2017_realistic_v5_PU50-v1/10000/7C654D7C-9E40-E711-8690-0025905A48BC.root
#'file:/afs/cern.ch/work/s/sdubey/data/Raw_Data_Phase1/7C654D7C-9E40-E711-8690-0025905A48BC.root'
#'file:/home/fpantale/data/920/PU50/085D5AAF-9E40-E711-B12A-0025905A609E.root'
#2016 CMSSW_8_1_0
#'file:/afs/cern.ch/work/s/sdubey/data/Raw_Data_Phase1/0216ABF7-19B1-E611-8786-0025905A60F8.root'
#'file:/afs/cern.ch/work/s/sdubey/data/9279A7C3-59ED-E511-95C8-0025905A60F8.root'
#'file:/afs/cern.ch/work/s/sdubey/data/RawToDigi/2EF61B7D-F216-E211-98C3-001D09F28D54.root'
#'file:/afs/cern.ch/work/s/sdubey/data/RawToDigi/data_phase1/0A176EE8-38C1-E611-B912-0CC47A4D765A.root'
#'/store/backfill/1/data/Tier0_Test_SUPERBUNNIES_vocms015/Commissioning/RAW/v82/000/276/357/00000/1AA497F3-EC6D-E611-A6B3-02163E0146CB.root'
#'/store/relval/CMSSW_8_1_0/MET/RAW-RECO/HighMET-81X_dataRun2_relval_v14_RelVal_met2016B-v1/10000/182C5786-C8BE-E611-B1C2-0CC47A7    8A33E.root'
 )
)
#file=/store/relval/CMSSW_8_1_0/RelValMinBias_13/GEN-SIM-DIGI-RAW/81X_upgrade2017_realistic_v26_HLT2017Trk-v1/10000/06A2997E-3BC1-E611-B286-0CC47A78A30E.root


process.load("EventFilter.SiPixelRawToDigi.SiPixelRawToDigi_cfi")

process.siPixelDigis.InputLabel = 'rawDataCollector'
process.siPixelDigis.IncludeErrors = False #True
process.siPixelDigis.Timing = False 
process.siPixelDigis.UsePhase1 = cms.bool(True)
# do the calibration ADC -> Electrons as required in clustering and apply the channel threshold
process.siPixelDigis.ConvertADCtoElectrons = cms.bool(False)

process.MessageLogger = cms.Service("MessageLogger",
    #debugModules = cms.untracked.vstring('siPixelDigis'),
    destinations = cms.untracked.vstring('log'),
    log = cms.untracked.PSet( threshold = cms.untracked.string('WARNING'))
    #log = cms.untracked.PSet( threshold = cms.untracked.string('DEBUG'))
)

process.p = cms.Path(process.siPixelDigis)
