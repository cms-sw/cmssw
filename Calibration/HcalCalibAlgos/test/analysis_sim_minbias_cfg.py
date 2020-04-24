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
     'file:/tmp/sunanda/0000E8D3-F58A-E411-8537-001E67396DEC.root',
    )
)

############################################################################
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag=autoCond['run2_mc']

############################################################################
#### Analysis

process.load('Calibration.HcalCalibAlgos.simAnalyzerMinbias_cfi')

process.schedule = cms.Path(    
    process.simAnalyzerMinbias*
    process.endOfProcess   
    )


