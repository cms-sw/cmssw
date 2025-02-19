import FWCore.ParameterSet.Config as cms

db_client = cms.EDAnalyzer("SiStripCommissioningOfflineDbClient",
  # general parameters
  FilePath         = cms.untracked.string('/tmp'),
  RunNumber        = cms.untracked.uint32(0),
  UseClientFile    = cms.untracked.bool(False),
  UploadHwConfig   = cms.untracked.bool(False),
  UploadAnalyses   = cms.untracked.bool(False),
  DisableDevices   = cms.untracked.bool(False),
  SaveClientFile   = cms.untracked.bool(True),
  SummaryXmlFile   = cms.untracked.FileInPath('DQM/SiStripCommissioningClients/data/summary.xml'),
  # individual parameters
  ApvTimingParameters      = cms.PSet(
    SkipFecUpdate = cms.bool(False),  # skip upload of APV PLL settings
    SkipFedUpdate = cms.bool(False),  # skip upload of FED frame finding threshold
    TargetDelay = cms.int32(-1)       # -1: latest tick (old default), otherwise target delay for all ticks' rising edge
  ),
  CalibrationParameters    = cms.PSet(),
  DaqScopeModeParameters   = cms.PSet(),
  FastFedCablingParameters = cms.PSet(),
  FedCablingParameters     = cms.PSet(),
  FedTimingParameters      = cms.PSet(),
  FineDelayParameters      = cms.PSet(
    cosmic =  cms.bool(True)
  ),
  LatencyParamameters      = cms.PSet(
    OptimizePerPartition = cms.bool(False)
  ),
  NoiseParameters          = cms.PSet(),
  OptoScanParameters       = cms.PSet(
    TargetGain = cms.double(0.863),   # target gain (0.863 ~ 690ADC for tickmark)
    SkipGainUpdate = cms.bool(False)  # wether to keep the gain the same as already on the db
  ),
  PedestalsParameters      = cms.PSet(
    DeadStripMax        = cms.double(10),    # number times the noise spread below mean noise
    NoisyStripMin       = cms.double(10),    # number times the noise spread above mean noise
    HighThreshold       = cms.double(5),    # analysis-wide high threshold for the fed zero suppression
    LowThreshold        = cms.double(2),    # analysis-wide low threshold for the fed zero suppression
    DisableBadStrips    = cms.bool(False),  # for experts! disables bad strips on the fed level 
    AddBadStrips				= cms.bool(False), #for experts! keep and add disabled bad strips. 
    KeepsStripsDisabled = cms.bool(False)   # for experts! keep strips disabled as in the db's current state
  ),
  PedsOnlyParameters       = cms.PSet(),
  PedsFullNoiseParameters  = cms.PSet(
    DeadStripMax        = cms.double(10),    # number times the noise spread below mean noise
    NoisyStripMin       = cms.double(10),    # number times the noise spread above mean noise
    HighThreshold       = cms.double(5),    # analysis-wide high threshold for the fed zero suppression
    LowThreshold        = cms.double(2),    # analysis-wide low threshold for the fed zero suppression
    KsProbCut						= cms.double(10),
    DisableBadStrips    = cms.bool(False),  # for experts! disables bad strips on the fed level 
    AddBadStrips				= cms.bool(False), 	#for experts! keep and add disabled bad strips.
    KeepsStripsDisabled = cms.bool(False)   # for experts! keeps strip disabling as in the db's current state
  ),
  SamplingParameters       = cms.PSet(),
  VpspScanParameters       = cms.PSet(),
)
