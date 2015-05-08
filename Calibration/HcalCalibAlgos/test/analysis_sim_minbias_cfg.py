import FWCore.ParameterSet.Config as cms

process = cms.Process('SimAna')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.RawToDigi_Data_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1000


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
     'root://cmsxrootd.fnal.gov//store/data/Run2012C/HcalNZS/RAW/v1/000/197/559/FA519929-93C0-E111-9C4A-BCAEC532971C.root',
    )
)

############################################################################
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag=autoCond['run2_mc']

############################################################################
#### Analysis

process.minbiasana = cms.EDAnalyzer("SimAnalyzerMinbias",
                                    HistOutFile = cms.untracked.string('simanalyzer.root'),
                                    TimeCut = cms.untracked.double(100.0)
                                    )

process.schedule = cms.Path(    
    process.minbiasana*
    process.endOfProcess   
    )


