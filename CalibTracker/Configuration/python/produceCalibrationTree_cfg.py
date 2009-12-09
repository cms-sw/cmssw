import FWCore.ParameterSet.Config as cms

process = cms.Process('CALIB')
process.load('CalibTracker.Configuration.setupCalibrationTree_cff')
process.load('Configuration.StandardSequences.Geometry_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.MixingNoPileUp_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = 'GR09_R_V1::All'

process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.StandardSequences.Services_cff')
process.add_( cms.Service( "TFileService",
                           fileName = cms.string( 'calibTree.root' ),
                           closeFileFast = cms.untracked.bool(True)  ) )

#following ignored by CRAB
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1000) )
process.source = cms.Source (
    "PoolSource",
    #from run = 123596, dataset = /MinimumBias/BeamCommissioning09-SD_AllMinBias-skim_GR09_P_V7_v1/RAW-RECO 
    fileNames = cms.untracked.vstring('/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_AllMinBias-skim_GR09_P_V7_v1/0099/FE729FAE-40E3-DE11-9EE1-00261894386A.root'),
    secondaryFileNames = cms.untracked.vstring())

