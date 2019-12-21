import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Modifier_ctpps_2016_cff import ctpps_2016
process = cms.Process('ctppsDQMfromAOD', ctpps_2016)

# minimum of logs
process.MessageLogger = cms.Service("MessageLogger",
  statistics = cms.untracked.vstring(),
  destinations = cms.untracked.vstring('cout'),
  cout = cms.untracked.PSet(
      # threshold = cms.untracked.string('WARNING')
      # threshold = cms.untracked.string('INFO')
  )
)

# load DQM framework
process.load("DQM.Integration.config.environment_cfi")
process.dqmEnv.subSystemFolder = "CTPPS"
process.dqmEnv.eventInfoFolder = "EventInfo"
process.dqmSaver.path = ""
process.dqmSaver.tag = "CTPPS"

# import of standard configurations for pixel
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

from Configuration.AlCa.GlobalTag import GlobalTag
# process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_data', '')
process.GlobalTag = GlobalTag(process.GlobalTag, '106X_dataRun2_v21', '')

# raw data source
process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring(
    # run 314255, alignment run April 2018
    # '/store/data/Commissioning2018/ZeroBias1/AOD/PromptReco-v1/000/314/255/00000/0098845D-D642-E811-8E30-FA163EE31D2A.root'
    # '/store/data/Run2018A/ZeroBias/AOD/PromptReco-v1/000/315/512/00000/34E72BFF-B64F-E811-A60F-FA163E9C8F11.root'
    '/store/data/Run2018A/ZeroBias/AOD/PromptReco-v1/000/315/512/00000/40896981-AD4F-E811-BA55-FA163E2E52D5.root'
    '/store/data/Run2018A/ZeroBias/AOD/PromptReco-v1/000/315/512/00000/2C32C288-AE4F-E811-A240-FA163E96604B.root',
    '/store/data/Run2018A/ZeroBias/AOD/PromptReco-v1/000/315/512/00000/580FDF29-B04F-E811-9D2C-FA163E8B221F.root',
    '/store/data/Run2018A/ZeroBias/AOD/PromptReco-v1/000/315/512/00000/1A8EE801-B14F-E811-886F-FA163E4830E6.root',
    '/store/data/Run2018A/ZeroBias/AOD/PromptReco-v1/000/315/512/00000/DEF10CD7-B24F-E811-BB39-FA163E5BE9EA.root',
    '/store/data/Run2018A/ZeroBias/AOD/PromptReco-v1/000/315/512/00000/9C00F693-B34F-E811-8FC8-FA163E1B57DB.root',
    '/store/data/Run2018A/ZeroBias/AOD/PromptReco-v1/000/315/512/00000/C6A0D209-B54F-E811-A8CB-FA163EE24CAD.root',
    '/store/data/Run2018A/ZeroBias/AOD/PromptReco-v1/000/315/512/00000/F8938BB9-B34F-E811-81E8-02163E015DCD.root',
    '/store/data/Run2018A/ZeroBias/AOD/PromptReco-v1/000/315/512/00000/B8EE7FDD-B24F-E811-9C1A-FA163E0C3CBE.root',
    '/store/data/Run2018A/ZeroBias/AOD/PromptReco-v1/000/315/512/00000/F4B0E008-B54F-E811-A760-FA163E77762A.root',
    '/store/data/Run2018A/ZeroBias/AOD/PromptReco-v1/000/315/512/00000/9AD3D368-B74F-E811-9042-FA163E4F8E43.root',
    '/store/data/Run2018A/ZeroBias/AOD/PromptReco-v1/000/315/512/00000/74FE2295-B44F-E811-876D-FA163E95BBB1.root',
    '/store/data/Run2018A/ZeroBias/AOD/PromptReco-v1/000/315/512/00000/3E7FD4A5-B64F-E811-8749-FA163EF5A563.root',
    '/store/data/Run2018A/ZeroBias/AOD/PromptReco-v1/000/315/512/00000/88D57301-B74F-E811-9E19-02163E01A01B.root',
    '/store/data/Run2018A/ZeroBias/AOD/PromptReco-v1/000/315/512/00000/1665E551-B94F-E811-9C27-FA163E7BA99F.root',
    '/store/data/Run2018A/ZeroBias/AOD/PromptReco-v1/000/315/512/00000/92209A92-B94F-E811-9DA7-FA163E0581F0.root',
    '/store/data/Run2018A/ZeroBias/AOD/PromptReco-v1/000/315/512/00000/14CD2CA9-B94F-E811-872F-FA163E9F4289.root',
    '/store/data/Run2018A/ZeroBias/AOD/PromptReco-v1/000/315/512/00000/46770D65-BE4F-E811-86AE-FA163E63F443.root',
    '/store/data/Run2018A/ZeroBias/AOD/PromptReco-v1/000/315/512/00000/F0E546D6-B44F-E811-88DE-02163E019F01.root',
    '/store/data/Run2018A/ZeroBias/AOD/PromptReco-v1/000/315/512/00000/9268D151-AD4F-E811-A688-02163E01A0E8.root',
    '/store/data/Run2018A/ZeroBias/AOD/PromptReco-v1/000/315/512/00000/52F6C528-AD4F-E811-B566-FA163EFCCBD6.root',
    '/store/data/Run2018A/ZeroBias/AOD/PromptReco-v1/000/315/512/00000/CEAE7DA1-B14F-E811-B365-FA163E7A4C8B.root',
    '/store/data/Run2018A/ZeroBias/AOD/PromptReco-v1/000/315/512/00000/52E511EF-B04F-E811-B837-FA163E5FB578.root',
    '/store/data/Run2018A/ZeroBias/AOD/PromptReco-v1/000/315/512/00000/A6524DF0-B24F-E811-8920-FA163E29F581.root',
    '/store/data/Run2018A/ZeroBias/AOD/PromptReco-v1/000/315/512/00000/F6668B61-B34F-E811-B4F4-FA163EEB4E2F.root',
    '/store/data/Run2018A/ZeroBias/AOD/PromptReco-v1/000/315/512/00000/36DF6B4A-B44F-E811-B8E2-FA163E8F6459.root',
    '/store/data/Run2018A/ZeroBias/AOD/PromptReco-v1/000/315/512/00000/C2DE5458-B44F-E811-BD63-FA163EC08F3E.root',
    '/store/data/Run2018A/ZeroBias/AOD/PromptReco-v1/000/315/512/00000/36EF9753-B54F-E811-AE31-FA163E2B3387.root',
    '/store/data/Run2018A/ZeroBias/AOD/PromptReco-v1/000/315/512/00000/2E065EEA-B74F-E811-BFF5-FA163E8A11D8.root',
    '/store/data/Run2018A/ZeroBias/AOD/PromptReco-v1/000/315/512/00000/0CBFA0C6-B64F-E811-8C50-FA163E455B73.root',
    '/store/data/Run2018A/ZeroBias/AOD/PromptReco-v1/000/315/512/00000/DE35E366-B84F-E811-B161-FA163E97727D.root',
    '/store/data/Run2018A/ZeroBias/AOD/PromptReco-v1/000/315/512/00000/326BF670-B74F-E811-B303-FA163E5FB578.root',
    '/store/data/Run2018A/ZeroBias/AOD/PromptReco-v1/000/315/512/00000/800D9E0F-C24F-E811-A9F4-FA163ED1B1DF.root',
    '/store/data/Run2018A/ZeroBias/AOD/PromptReco-v1/000/315/512/00000/8E6113D5-CF4F-E811-8F63-FA163E54FDC7.root',
    '/store/data/Run2018A/ZeroBias/AOD/PromptReco-v1/000/315/512/00000/909F10AB-B24F-E811-8002-FA163E756B33.root',
    '/store/data/Run2018A/ZeroBias/AOD/PromptReco-v1/000/315/512/00000/BC73304D-B44F-E811-9929-FA163EBBA909.root',
    '/store/data/Run2018A/ZeroBias/AOD/PromptReco-v1/000/315/512/00000/48A17F20-B54F-E811-96C0-FA163E9F4289.root',
    '/store/data/Run2018A/ZeroBias/AOD/PromptReco-v1/000/315/512/00000/34E72BFF-B64F-E811-A60F-FA163E9C8F11.root',
    '/store/data/Run2018A/ZeroBias/AOD/PromptReco-v1/000/315/512/00000/88B4804C-B74F-E811-AFFE-FA163EEA7BBD.root',
    '/store/data/Run2018A/ZeroBias/AOD/PromptReco-v1/000/315/512/00000/A69B5881-B94F-E811-85B9-FA163E6314D2.root'
  ),
  skipBadFiles = cms.untracked.bool(True),

)

process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32(-1)
)

# geometry definition and reco modules
process.load("RecoCTPPS.Configuration.recoCTPPS_cff")

# CTPPS DQM modules
process.load("DQM.CTPPS.ctppsDQM_cff")

process.path = cms.Path(
  process.ctppsPixelLocalReconstruction
  * process.ctppsLocalTrackLiteProducer
  * process.ctppsProtons

  * process.ctppsDQMElastic
)

process.end_path = cms.EndPath(
  process.dqmEnv +
  process.dqmSaver
)

process.schedule = cms.Schedule(
  process.path,
  process.end_path
)
