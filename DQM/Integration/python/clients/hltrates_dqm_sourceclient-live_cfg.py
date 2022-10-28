import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run3_cff import Run3
process = cms.Process("DQM", Run3)
process.options = cms.untracked.PSet(
  SkipEvent = cms.untracked.vstring('ProductNotFound') 
)

process.load("DQMServices.Core.DQM_cfg")

#### leave the following few lines uncommented for online running
process.load("DQM.Integration.config.inputsource_cfi")
from DQM.Integration.config.inputsource_cfi import options
process.load("DQM.Integration.config.environment_cfi")
#process.DQMEventStreamHttpReader.SelectHLTOutput = cms.untracked.string('hltOutputHLTDQMResults')

#### end first online running section

############ Test offline running


# process.source = cms.Source("PoolSource",
#                                 fileNames = cms.untracked.vstring('file:/data/ndpc0/b/slaunwhj/rawData/June/0091D91D-D19B-E011-BDCE-001D09F2512C.root')
#                             )

# process.maxEvents = cms.untracked.PSet(
#         input = cms.untracked.int32(-1)
#         )

##############################


# old, not used
#process.DQMEventStreamHttpReader.sourceURL = cms.string('http://srv-c2c07-13.cms:11100/urn:xdaq-application:lid=50')

# old, not used

process.dqmSaver.tag = "HLTRates"
process.dqmSaver.runNumber = options.runNumber
process.dqmSaverPB.tag = 'HLTRates'
process.dqmSaverPB.runNumber = options.runNumber

#process.load("Configuration.StandardSequences.GeometryPilot2_cff")
#process.load("Configuration.StandardSequences.MagneticField_cff")
#process.GlobalTrackingGeometryESProducer = cms.ESProducer( "GlobalTrackingGeometryESProducer" ) # for muon hlt dqm
#SiStrip Local Reco
#process.load("CalibTracker.SiStripCommon.TkDetMapESProducer_cfi")

#---- for P5 (online) DB access
process.load("DQM.Integration.config.FrontierCondition_GT_cfi")

#---- for offline DB access: change and possibly customise the GT
#from Configuration.AlCa.GlobalTag import GlobalTag as gtCustomise
#process.GlobalTag = gtCustomise(process.GlobalTag, "74X_dataRun2_Express_v3", "")



################################
#
# Need to do raw to digi
# in order to use PS providers
#
# This is a hassle
# but I will try it
# following lines are only for
# running the silly RawToDigi
#
################################

# JMS Aug 16 2011 
# Remove these
# We don't need to run raw to digi
#

#process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
#process.load('Configuration/StandardSequences/RawToDigi_Data_cff')

#process.load("CalibTracker.SiStripCommon.TkDetMapESProducer_cfi")

####### JMS Aug 16 2011 you do need to prescale
process.hltPreTrigResRateMon = cms.EDFilter ("HLTPrescaler",
                                             L1GtReadoutRecordTag = cms.InputTag("NONE"),
                                             offset = cms.uint32(0)
                                             )

process.PrescaleService = cms.Service( "PrescaleService",
    lvl1DefaultLabel = cms.string( "PSTrigRates" ),
    lvl1Labels = cms.vstring( 'PS'),
    prescaleTable = cms.VPSet(
    cms.PSet(  pathName = cms.string( "rateMon" ),
        prescales = cms.vuint32(6)
      ),
     )
    )


process.load("DQM.HLTEvF.TrigResRateMon_cfi")

# run on 1 out of 8 SM, LSSize 23 -> 23/8 = 2.875
# stream is prescaled by 10, to correct change LSSize 23 -> 23/10 = 2.3
process.trRateMon.LuminositySegmentSize = 2.3


# Add RawToDigi
process.rateMon = cms.EndPath(process.hltPreTrigResRateMon *process.trRateMon)


process.pp = cms.Path(process.dqmEnv+process.dqmSaver+process.dqmSaverPB)

process.dqmEnv.subSystemFolder = 'HLT/TrigResults'
#process.hltResults.plotAll = True


### process customizations included here
from DQM.Integration.config.online_customizations_cfi import *
process = customise(process)
