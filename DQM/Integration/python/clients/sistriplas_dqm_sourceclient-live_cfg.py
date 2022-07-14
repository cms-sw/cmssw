from __future__ import print_function
import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run3_cff import Run3
process = cms.Process( "sistriplaserDQMLive", Run3 )
process.MessageLogger = cms.Service( "MessageLogger",
  cout = cms.untracked.PSet(threshold = cms.untracked.string( 'ERROR' )),
  destinations = cms.untracked.vstring( 'cout')
)
#----------------------------
# Event Source
#-----------------------------
# for live online DQM in P5
#process.load("DQM.Integration.config.inputsource_cfi")
#from DQM.Integration.config.inputsource_cfi import options

# for testing in lxplus
process.load("DQM.Integration.config.fileinputsource_cfi")
from DQM.Integration.config.fileinputsource_cfi import options

process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32( -1 )
)

process.load( "EventFilter.SiStripRawToDigi.SiStripDigis_cfi" )
process.siStripDigis.ProductLabel = "source"#"hltCalibrationRaw"
#--------------------------
# Calibration
#--------------------------
# Condition for P5 cluster
process.load("DQM.Integration.config.FrontierCondition_GT_cfi")
# Condition for lxplus: change and possibly customise the GT
#from Configuration.AlCa.GlobalTag import GlobalTag as gtCustomise
#process.GlobalTag = gtCustomise(process.GlobalTag, 'auto:run3_data', '')

#----------------------------
# DQM Live Environment
#-----------------------------
process.load("DQM.Integration.config.environment_cfi")
process.dqmEnv.subSystemFolder    = "SiStripLAS"
process.dqmSaver.tag = "SiStripLAS"
process.dqmSaver.runNumber = options.runNumber
process.dqmSaverPB.tag = "SiStripLAS"
process.dqmSaverPB.runNumber = options.runNumber
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
process.LaserAlignmentProducerDQM.UpperAdcThreshold = 280 

process.seqDigitization = cms.Path( process.siStripDigis )
process.DQMCommon   = cms.Sequence(process.dqmEnv*process.dqmSaver*process.dqmSaverPB)

process.seqAnalysis = cms.Path( process.LaserAlignmentProducerDQM*process.DQMCommon)
process.siStripDigis.ProductLabel = "hltTrackerCalibrationRaw"
#--------------------------------------------------
# Heavy Ion Specific Fed Raw Data Collection Label
#--------------------------------------------------

print("Running with run type = ", process.runType.getRunType())

if (process.runType.getRunType() == process.runType.hi_run):
    process.siStripDigis.ProductLabel = "rawDataRepacker"


### process customizations included here
from DQM.Integration.config.online_customizations_cfi import *
process = customise(process)
