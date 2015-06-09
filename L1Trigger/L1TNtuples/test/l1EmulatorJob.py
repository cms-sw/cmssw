import FWCore.ParameterSet.Config as cms

# make L1 ntuples from RAW+RECO

process = cms.Process("L1EMULATOR")

# import of standard configurations
process.load('Configuration/StandardSequences/Services_cff')
process.load('FWCore/MessageService/MessageLogger_cfi')
process.load('Configuration/StandardSequences/GeometryIdeal_cff')
process.load('Configuration/StandardSequences/MagneticField_38T_cff')
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.load('Configuration/EventContent/EventContent_cff')
process.load("Configuration.StandardSequences.RawToDigi_Data_cff")
process.load('Configuration/StandardSequences/L1HwVal_cff')
process.load('Configuration/StandardSequences/EndOfProcess_cff')

# global tag
process.GlobalTag.globaltag = 'GR_P_V14::All'

# output file
#process.TFileService = cms.Service("TFileService",
#    fileName = cms.string('L1EmulHistos.root')
#)



# which systems to compare
# ETP,HTP,RCT,GCT, DTP,DTF,CTP,CTF,RPC, LTC,GMT,GT
process.l1compare.COMPARE_COLLS = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]

#dqm
process.load('DQM.L1TMonitor.L1TDEMON_cfi')
process.load('DQM.L1TMonitor.L1TdeECAL_cfi')
process.load('DQM.L1TMonitor.L1TdeGCT_cfi')
process.load('DQM.L1TMonitor.L1TdeRCT_cfi')
process.load('DQM.L1TMonitor.L1TdeCSCTF_cfi')
process.load('DQM.L1TMonitor.l1GtHwValidation_cfi')

process.l1demon.HistFile='L1EmulHistos.root'
process.l1demon.disableROOToutput=cms.untracked.bool(False)

# temporary fixes

# for GT
process.valGtDigis.RecordLength = cms.vint32(3, 5)
process.valGtDigis.AlternativeNrBxBoardDaq = 0x101
process.valGtDigis.AlternativeNrBxBoardEvm = 0x2

# for CSCTF
process.l1decsctf.PTLUT = cms.PSet(
    LowQualityFlag = cms.untracked.uint32(4),
    ReadPtLUT = cms.bool(False),
    PtMethod = cms.untracked.uint32(1)
)

# path
process.p = cms.Path(
    process.RawToDigi
    +process.L1HwVal
    +process.l1demon
    +process.l1demonecal
    +process.l1demongct
    +process.l1decsctf
    +process.l1GtHwValidation
    +process.l1tderct
)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100) )

readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring() 
process.source = cms.Source ("PoolSource",
                             fileNames = readFiles,
                             secondaryFileNames = secFiles
                             )

readFiles.extend( [
    '/store/data/Commissioning10/MinimumBias/RAW/v4/000/131/511/FEED5CBB-5434-DF11-BDBC-0030487C90EE.root',
    '/store/data/Commissioning10/MinimumBias/RAW/v4/000/131/511/D6D10EE4-4A34-DF11-8827-000423D991D4.root',
    '/store/data/Commissioning10/MinimumBias/RAW/v4/000/131/511/D470C00D-6934-DF11-8FC1-000423D99F3E.root',
    '/store/data/Commissioning10/MinimumBias/RAW/v4/000/131/511/D464BB8B-6534-DF11-87F1-000423D99BF2.root',
    '/store/data/Commissioning10/MinimumBias/RAW/v4/000/131/511/CA19678C-5E34-DF11-BE1C-000423D99896.root',
    '/store/data/Commissioning10/MinimumBias/RAW/v4/000/131/511/BA84F50D-6234-DF11-BBB5-000423D33970.root',
    '/store/data/Commissioning10/MinimumBias/RAW/v4/000/131/511/A62D0EF7-6634-DF11-B74F-000423D99896.root',
    '/store/data/Commissioning10/MinimumBias/RAW/v4/000/131/511/A490BCAD-6E34-DF11-8922-0030487CD7E0.root',
    '/store/data/Commissioning10/MinimumBias/RAW/v4/000/131/511/907DFCDC-3534-DF11-AC71-000423D98FBC.root',
    '/store/data/Commissioning10/MinimumBias/RAW/v4/000/131/511/8AF95A28-6B34-DF11-94AC-000423D98800.root'
] )

secFiles.extend( [
#       '/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/122/318/94CEE17E-79D8-DE11-97D5-001D09F28F11.root',
#       '/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/122/318/7EDBAEFA-70D8-DE11-ACFE-001617DBCF6A.root',
#       '/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/122/318/648548EF-75D8-DE11-A26F-000423D94A04.root',
#       '/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/122/318/2AEB364C-7CD8-DE11-A15C-001D09F241B9.root',
#       '/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/122/318/16A68DCA-73D8-DE11-88FA-001617DBD224.root',
#       '/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/122/318/0C749FE8-7AD8-DE11-B837-001D09F29114.root'
       ] )
