import FWCore.ParameterSet.Config as cms

client = cms.EDAnalyzer("SiStripCommissioningOfflineClient",
  # general parameters
  FilePath         = cms.untracked.string('/tmp'),
  RunNumber        = cms.untracked.uint32(0),
  UseClientFile    = cms.untracked.bool(False),
  SaveClientFile   = cms.untracked.bool(True),
  SummaryXmlFile   = cms.untracked.FileInPath('DQM/SiStripCommissioningClients/data/summary.xml'),
  # individual parameters
  ApvTimingParameters	   = cms.PSet(),
  CalibrationParameters    = cms.PSet(),
  DaqScopeModeParameters   = cms.PSet(),
  FastFedCablingParameters = cms.PSet(),
  FedCablingParameters     = cms.PSet(),
  FedTiming		   = cms.PSet(),
  FineDelayParameters	   = cms.PSet(),
  LatencyParams 	   = cms.PSet(),
  NoiseParameters	   = cms.PSet(),
  OptoScanParameters	   = cms.PSet(
    TargetGain = cms.double(0.8)    # target gain (0.8 = 640ADC for tickmark)
  ),
  PedestalsParameters	   = cms.PSet(
    DeadStripMax  = cms.double(10),  # number times the noise spread below mean noise
    NoisyStripMin = cms.double(5),  # number times the noise spread above mean noise
    HighThreshold = cms.double(5),  # analysis-wide high threshold for the fed zero suppression
    LowThreshold  = cms.double(2)   # analysis-wide low threshold for the fed zero suppression
  ),
  PedsOnlyParameters	   = cms.PSet(),
  SamplingParameters	   = cms.PSet(),
  VpspScanParameters	   = cms.PSet(),
)
