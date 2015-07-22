import FWCore.ParameterSet.Config as cms

process = cms.Process('RecAna')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.RawToDigi_Data_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.163 $'),
    annotation = cms.untracked.string('Reconstruction.py nevts:10'),
    name = cms.untracked.string('PyReleaseValidation')
    )
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
    )
process.options = cms.untracked.PSet(
    
    )
# Input source
process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
        'file:/tmp/sunanda/0000E8D3-F58A-E411-8537-001E67396DEC.root',
#   'root://cmsxrootd.fnal.gov//store/data/Run2012C/HcalNZS/RAW/v1/000/203/522/F233228C-D206-E211-BC2E-0025901D629C.root',
#   'root://cmsxrootd.fnal.gov//store/data/Run2012C/HcalNZS/RAW/v1/000/197/556/D673F957-93C0-E111-8332-003048D2BF1C.root',
#   'root://cmsxrootd.fnal.gov//store/data/Run2012C/HcalNZS/RAW/v1/000/198/941/3AA08F69-9ECD-E111-A338-5404A63886C5.root',
#   'root://cmsxrootd.fnal.gov//store/data/Run2012C/HcalNZS/RAW/v1/000/197/601/BAC19C2C-3EC1-E111-8A78-001D09F241B9.root',
#    'root://cmsxrootd.fnal.gov//store/data/Run2012C/HcalNZS/RAW/v1/000/197/610/F68235DD-01C1-E111-9583-0019B9F72CE5.root',
#   'root://cmsxrootd.fnal.gov//store/data/Run2012C/HcalNZS/RAW/v1/000/197/719/2AADB426-7BC1-E111-9053-001D09F24353.root', 
#    'root://cmsxrootd.fnal.gov//store/data/Run2012A/HcalNZS/RAW/v1/000/190/456/88BBA906-3B7F-E111-BD13-001D09F2441B.root',
#    'root://cmsxrootd.fnal.gov//store/data/Run2012C/HcalNZS/RAW/v1/000/199/975/24F29491-FADA-E111-A19B-BCAEC5329716.root',
    )
)

############################################################################
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['run2_mc'] 

############################################################################
#### Analysis

process.load("Calibration.HcalCalibAlgos.recAnalyzerMinbias_cfi")

process.schedule = cms.Path(process.RecAnalyzerMinbias*process.endOfProcess)
