import FWCore.ParameterSet.Config as cms

process = cms.Process( "sistriplaserDQMLive" )
process.MessageLogger = cms.Service( "MessageLogger",
  cout = cms.untracked.PSet(threshold = cms.untracked.string( 'ERROR' )),
  destinations = cms.untracked.vstring( 'cout')
)
#----------------------------
# Event Source
#-----------------------------
# for live online DQM in P5
#process.load("DQM.Integration.config.inputsource_cfi")

# for testing in lxplus
process.load("DQM.Integration.config.fileinputsource_cfi")

process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32( -1 )
)

process.load( "EventFilter.SiStripRawToDigi.SiStripDigis_cfi" )
process.siStripDigis.ProductLabel = "source"#"hltCalibrationRaw"
#--------------------------
# Calibration
#--------------------------
# Condition for P5 cluster
#process.load("DQM.Integration.config.FrontierCondition_GT_cfi")
# Condition for lxplus
process.load("DQM.Integration.config.FrontierCondition_GT_Offline_cfi") 

#----------------------------
# DQM Live Environment
#-----------------------------
process.load("DQM.Integration.config.environment_cfi")
process.dqmEnv.subSystemFolder    = "SiStripLAS"
process.dqmSaver.tag = "SiStripLAS"
#----------------------------
# DQM Alignment Software
#----------------------------
process.load( "DQM.SiStripLASMonitor.LaserAlignmentProducerDQM_cfi" )
process.LaserAlignmentProducerDQM.DigiProducerList = cms.VPSet(
  cms.PSet(
    DigiLabel = cms.string( 'ZeroSuppressed' ),
    DigiType = cms.string( 'Processed' ),
    DigiProducer = cms.string( 'siStripDigis' )
  )
)
process.LaserAlignmentProducerDQM.FolderName = "SiStripLAS"
process.LaserAlignmentProducerDQM.UpperAdcThreshold = cms.uint32( 280 )

process.seqDigitization = cms.Path( process.siStripDigis )
process.DQMCommon   = cms.Sequence(process.dqmEnv*process.dqmSaver)

process.seqAnalysis = cms.Path( process.LaserAlignmentProducerDQM*process.DQMCommon)
process.siStripDigis.ProductLabel = cms.InputTag("hltTrackerCalibrationRaw")
#--------------------------------------------------
# Heavy Ion Specific Fed Raw Data Collection Label
#--------------------------------------------------

print "Running with run type = ", process.runType.getRunType()

if (process.runType.getRunType() == process.runType.hi_run):
    process.siStripDigis.ProductLabel = cms.InputTag("rawDataRepacker")


### process customizations included here
from DQM.Integration.config.online_customizations_cfi import *
process = customise(process)
