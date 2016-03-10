import FWCore.ParameterSet.Config as cms

process = cms.Process('CALIB')
process.load('CalibTracker.Configuration.setupCalibrationTree_cff')
process.load('Configuration.StandardSequences.Geometry_cff')
process.load('Configuration/StandardSequences/MagneticField_AutoFromDBCurrent_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = 'auto:run2_data'

process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.StandardSequences.Services_cff')
process.add_( cms.Service( "TFileService",
                           fileName = cms.string( 'calibTreeTest.root' ),
                           closeFileFast = cms.untracked.bool(True)  ) )

process.load('CalibTracker.SiStripCommon.theBigNtuple_cfi')
process.TkCalPath_AB   = cms.Path( process.theBigNtuple * process.TkCalSeq_AllBunch     )
process.TkCalPath_AB0T = cms.Path( process.theBigNtuple * process.TkCalSeq_AllBunch0T   )
process.TkCalPath_IB   = cms.Path( process.theBigNtuple * process.TkCalSeq_IsoBunch     )
process.TkCalPath_IB0T = cms.Path( process.theBigNtuple * process.TkCalSeq_IsoBunch0T   )
process.TkPathDigi     = cms.Path (process.theBigNtupleDigi)
process.endPath        = cms.EndPath(process.bigShallowTree)

process.schedule = cms.Schedule( process.TkCalPath_AB, process.TkCalPath_AB0T, process.TkCalPath_IB, process.TkCalPath_IB0T )


#following ignored by CRAB
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100) )
process.source = cms.Source (
    "PoolSource",
    fileNames=cms.untracked.vstring(
      '/store/data/Run2015D/ZeroBias/ALCARECO/SiStripCalZeroBias-16Dec2015-v1/60002/9A39175A-8BAC-E511-B537-00261894393C.root',
      '/store/data/Run2015D/ZeroBias/ALCARECO/SiStripCalZeroBias-16Dec2015-v1/60003/F632A9B7-F9A8-E511-AB7B-001E67398390.root',
      '/store/data/Run2015D/ZeroBias/ALCARECO/SiStripCalZeroBias-16Dec2015-v1/60006/3A69DC8F-3BAA-E511-B41E-3417EBE645E2.root',
      '/store/data/Run2015D/ZeroBias/ALCARECO/SiStripCalZeroBias-16Dec2015-v1/60006/48FF6A16-CCAB-E511-A8C4-0025905A612C.root',
      '/store/data/Run2015D/ZeroBias/ALCARECO/SiStripCalZeroBias-16Dec2015-v1/60006/6401E5E5-F4AC-E511-8FBA-0CC47A4C8F0A.root',
      '/store/data/Run2015D/ZeroBias/ALCARECO/SiStripCalZeroBias-16Dec2015-v1/60006/6A9970CA-39AA-E511-8859-0CC47A4DED1A.root',
      '/store/data/Run2015D/ZeroBias/ALCARECO/SiStripCalZeroBias-16Dec2015-v1/60006/6EA49B98-DEAC-E511-A459-0025905A60E0.root',
      '/store/data/Run2015D/ZeroBias/ALCARECO/SiStripCalZeroBias-16Dec2015-v1/60006/80756625-3DAA-E511-92BA-001D09FDD6AB.root',
      '/store/data/Run2015D/ZeroBias/ALCARECO/SiStripCalZeroBias-16Dec2015-v1/60006/927C387E-B5AC-E511-87C5-0CC47A78A478.root',
      '/store/data/Run2015D/ZeroBias/ALCARECO/SiStripCalZeroBias-16Dec2015-v1/60006/969B7185-01AD-E511-86C4-0CC47A78A3EE.root',
      '/store/data/Run2015D/ZeroBias/ALCARECO/SiStripCalZeroBias-16Dec2015-v1/60006/9E36132E-19AD-E511-AD43-0CC47A4D760C.root',
      '/store/data/Run2015D/ZeroBias/ALCARECO/SiStripCalZeroBias-16Dec2015-v1/60006/A6870647-39AA-E511-8754-549F358EB748.root',
      '/store/data/Run2015D/ZeroBias/ALCARECO/SiStripCalZeroBias-16Dec2015-v1/60006/C243CE41-3AAA-E511-ACCE-A0369F7FC544.root',
      '/store/data/Run2015D/ZeroBias/ALCARECO/SiStripCalZeroBias-16Dec2015-v1/60006/C412AC3F-4AAC-E511-9881-0025905A6068.root',
      '/store/data/Run2015D/ZeroBias/ALCARECO/SiStripCalZeroBias-16Dec2015-v1/60006/D8B72B94-3AAC-E511-B81D-0025905A6066.root',
      '/store/data/Run2015D/ZeroBias/ALCARECO/SiStripCalZeroBias-16Dec2015-v1/60006/F6363144-3AAA-E511-A708-002590D0B016.root',
      '/store/data/Run2015D/ZeroBias/ALCARECO/SiStripCalZeroBias-16Dec2015-v1/60006/F88F1B6D-3FAA-E511-8ADA-A0369F7F9EDC.root',
      '/store/data/Run2015D/ZeroBias/ALCARECO/SiStripCalZeroBias-16Dec2015-v1/60006/FAD08497-DEAC-E511-9CB5-0CC47A4D762E.root',
      '/store/data/Run2015D/ZeroBias/ALCARECO/SiStripCalZeroBias-16Dec2015-v1/60007/CC5DD2C0-65AA-E511-8A92-549F35AC7E56.root'
                                    ),
    secondaryFileNames = cms.untracked.vstring())

