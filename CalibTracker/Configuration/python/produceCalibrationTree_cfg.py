import FWCore.ParameterSet.Config as cms

process = cms.Process('CALIB')
process.load('CalibTracker.Configuration.setupCalibrationTree_cff')
process.load('Configuration.StandardSequences.Geometry_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.MixingNoPileUp_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = 'CRAFT09_R_V4::All'

process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.StandardSequences.Services_cff')
process.add_( cms.Service( "TFileService",
                           fileName = cms.string( 'calibTree.root' ),
                           closeFileFast = cms.untracked.bool(True)  ) )

#following ignored by CRAB
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1000) )
process.source = cms.Source (
    "PoolSource",
#    fileNames = cms.untracked.vstring('/store/data/CRAFT09/Cosmics/ALCARECO/v1/000/109/046/E6C9E810-BD7C-DE11-8F94-000423D9997E.root'),
fileNames = cms.untracked.vstring('file:../../CMSSW_3_2_7/src/E6C9E810-BD7C-DE11-8F94-000423D9997E.root'),
    secondaryFileNames = cms.untracked.vstring())

