import FWCore.ParameterSet.Config as cms

process = cms.Process('CALIB')
process.load('CalibTracker.SiStripLorentzAngle.Tree_REDIGI_cff')
process.load('Configuration.StandardSequences.GeometryIdeal_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.MixingNoPileUp_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = 'DESIGN_31X_V7::All'
process.simSiStripDigis.APVpeakmode = True

process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.StandardSequences.Services_cff')
process.add_( cms.Service( "TFileService",
                           fileName = cms.string( 'calibTree_peak.root' ),
                           closeFileFast = cms.untracked.bool(True)  ) )

#following ignored by CRAB
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )
process.source = cms.Source (
    "PoolSource",
    fileNames = cms.untracked.vstring('/store/relval/CMSSW_3_1_1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V2-v1/0002/EA8E5AF7-576B-DE11-BA98-001D09F24498.root'),
    secondaryFileNames = cms.untracked.vstring())

