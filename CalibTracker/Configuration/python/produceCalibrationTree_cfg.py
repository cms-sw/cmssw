import FWCore.ParameterSet.Config as cms

process = cms.Process('CALIB')
process.load('CalibTracker.Configuration.setupCalibrationTree_cff')
process.load('Configuration.StandardSequences.Geometry_cff')
process.load('Configuration/StandardSequences/MagneticField_AutoFromDBCurrent_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = 'GR09_R_V1::All'

process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.StandardSequences.Services_cff')
process.add_( cms.Service( "TFileService",
                           fileName = cms.string( 'calibTreeTest.root' ),
                           closeFileFast = cms.untracked.bool(True)  ) )

process.load('CalibTracker.SiStripCommon.theBigNtuple_cfi')
process.TkCalPath = cms.Path( process.theBigNtuple * process.TkCalFullSequence     )
process.TkPathDigi = cms.Path (process.theBigNtupleDigi)
process.endPath = cms.EndPath(process.bigShallowTree)

#process.schedule = cms.Schedule( process.TkCalPath )


#following ignored by CRAB
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100) )
process.source = cms.Source (
    "PoolSource",
    #from run = 123596, dataset = /MinimumBias/BeamCommissioning09-SD_AllMinBias-skim_GR09_P_V7_v1/RAW-RECO 
    fileNames=cms.untracked.vstring('/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/Dec9thReReco_BSCNOBEAMHALO-v1/0000/5A8A4A57-22E5-DE11-81B0-0026189438B5.root',
                                    '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/Dec9thReReco_BSCNOBEAMHALO-v1/0000/1A202DCE-1DE5-DE11-A176-002618943985.root'
                                    ),
    secondaryFileNames = cms.untracked.vstring())

