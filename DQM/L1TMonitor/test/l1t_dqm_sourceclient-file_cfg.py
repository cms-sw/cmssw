from __future__ import print_function
#
# cfg file to run online L1 Trigger DQM
#     the user can choose the environment (live, playback, file-P5, file)
#     for 'file, one can also choose the global tag type and the actual tag
#
# V M Ghete 2010-07-09


import FWCore.ParameterSet.Config as cms
import sys

# choose the environment you run
#l1DqmEnv = 'live'
#l1DqmEnv = 'playback'
#l1DqmEnv = 'file-P5'
l1DqmEnv = 'file'

l1Condition = 'legacy'
#l1Condition = 'stage1'

# for 'file' choose also the type of the global tag and (edit) the actual global tag
if l1DqmEnv == 'file' :

    globalTagType = 'HLT'
    #globalTagType = 'P'
    #globalTagType = 'E'
    #globalTagType = 'R'

    if globalTagType == 'HLT' :
        globalTagValue = 'GR_H_V44'
    elif globalTagType == 'P' :
        globalTagValue = 'GR_P_V29'
    elif globalTagType == 'E' :
        globalTagValue = 'GR_E_V23'
    elif globalTagType == 'R' :
        globalTagValue = 'GR_R_52_V4'
    else :
        print('No valid global tag type', globalTagType)
        print('Valid types: HLT, P, E, R')
        sys.exit()


process = cms.Process("DQM")

# check that a valid choice for environment exists

if not ((l1DqmEnv == 'live') or l1DqmEnv == 'playback' or l1DqmEnv == 'file-P5' or l1DqmEnv == 'file' ) :
    print('No valid input source was chosen. Your value for l1DqmEnv input parameter is:')
    print('l1DqmEnv = ', l1DqmEnv)
    print('Available options: "live", "playback", "file-P5", "file" ')
    sys.exit()

#----------------------------
# Event Source
#

if l1DqmEnv == 'live' :
    process.load("DQM.Integration.test.inputsource_cfi")
    process.EventStreamHttpReader.SelectEvents = cms.untracked.PSet(
            SelectEvents = cms.vstring("*")
            )
    process.EventStreamHttpReader.consumerName = 'L1T DQM Consumer'
    process.EventStreamHttpReader.maxEventRequestRate = cms.untracked.double(25.0)

elif l1DqmEnv == 'playback' :
    print('FIXME')
    sys.exit()

else :
    # running on a file
    process.load("DQM.L1TMonitor.inputsource_file_cfi")


#----------------------------
# DQM Environment
#

#process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")
process.dqmEnv.subSystemFolder = 'L1T'

if l1DqmEnv == 'live' :
    process.load("DQM.Integration.test.environment_cfi")

    #
    # load and configure modules via Global Tag
    # https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideFrontierConditions
    process.load("DQM.Integration.test.FrontierCondition_GT_cfi")
    es_prefer_GlobalTag = cms.ESPrefer('GlobalTag')
    process.GlobalTag.RefreshEachRun = cms.untracked.bool(True)

elif l1DqmEnv == 'playback' :
    print('FIXME')

elif l1DqmEnv == 'file-P5' :
    process.load("DQM.Integration.test.FrontierCondition_GT_cfi")
    es_prefer_GlobalTag = cms.ESPrefer('GlobalTag')
    process.GlobalTag.RefreshEachRun = cms.untracked.bool(True)

else :
    # running on a file, on lxplus (not on .cms)
    process.load("DQM.L1TMonitor.environment_file_cff")

    process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

    if globalTagType == 'HLT' :
         process.GlobalTag.connect = 'frontier://FrontierProd/CMS_COND_31X_GLOBALTAG'
         process.GlobalTag.pfnPrefix = cms.untracked.string('frontier://FrontierProd/')

    process.GlobalTag.globaltag = globalTagValue+'::All'
    es_prefer_GlobalTag = cms.ESPrefer('GlobalTag')


#process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")

#-------------------------------------
# sequences needed for L1 trigger DQM
#

# standard unpacking sequence
process.load("Configuration.StandardSequences.RawToDigi_Data_cff")

# L1 Trigger sequences

# l1tMonitor and l1tMonitorEndPathSeq
process.load("DQM.L1TMonitor.L1TMonitor_cff")

# L1 trigger synchronization module - it uses also HltHighLevel filter
process.load("DQM.L1TMonitor.L1TSync_cff")

# l1tMonitorClient and l1tMonitorClientEndPathSeq
process.load("DQM.L1TMonitorClient.L1TMonitorClient_cff")

# change the DB connections when not at P5 - works on lxplus only...
if ( l1DqmEnv == 'file' ) :
    process.l1tSync.oracleDB = cms.string("oracle://cms_orcon_adg/CMS_COND_31X_L1T")
    process.l1tSync.pathCondDB = cms.string("/afs/cern.ch/cms/DB/conddb/ADG")
    #
    process.l1tRate.oracleDB = cms.string("oracle://cms_orcon_adg/CMS_COND_31X_L1T")
    process.l1tRate.pathCondDB = cms.string("/afs/cern.ch/cms/DB/conddb/ADG")


#-------------------------------------
# paths & schedule for L1 Trigger DQM
#

# TODO define a L1 trigger L1TriggerRawToDigi in the standard sequence
# to avoid all these remove
process.rawToDigiPath = cms.Path(process.RawToDigi)
#
process.RawToDigi.remove("siPixelDigis")
process.RawToDigi.remove("siStripDigis")
process.RawToDigi.remove("scalersRawToDigi")
process.RawToDigi.remove("castorDigis")

# for GCT, unpack all five samples
process.gctDigis.numberOfGctSamplesToUnpack = cms.uint32(5)

if l1Condition == 'stage1':
    process.l1tMonitorPath = cms.Path(process.l1tMonitorStage1Online)
    process.l1tMonitorClientPath = cms.Path(process.l1tMonitorStage1Client)
    # Update HfRing thresholds to accomodate di-iso tau trigger thresholds
    from L1TriggerConfig.L1ScalesProducers.l1CaloScales_cfi import l1CaloScales
    l1CaloScales.L1HfRingThresholds = cms.vdouble(0.0, 24.0, 28.0, 32.0, 36.0, 40.0, 44.0, 48.0)
    l1CaloScales.L1HtMissThresholds = cms.vdouble(
      0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
      0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19,
      0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29,
      0.30, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39,
      0.40, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49,
      0.50, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59,
      0.60, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69,
      0.70, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79,
      0.80, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89,
      0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99,
      1.00, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09,
      1.10, 1.11, 1.12, 1.13, 1.14, 1.15, 1.16, 1.17, 1.18, 1.19,
      1.20, 1.21, 1.22, 1.23, 1.24, 1.25, 1.26, 1.27
    )
else :
    process.l1tMonitorPath = cms.Path(process.l1tMonitorOnline)
    process.l1tMonitorClientPath = cms.Path(process.l1tMonitorClient)

# separate L1TSync path due to the use of the HltHighLevel filter
process.l1tSyncPath = cms.Path(process.l1tSyncHltFilter+process.l1tSync)

#
process.l1tMonitorEndPath = cms.EndPath(process.l1tMonitorEndPathSeq)

#
process.l1tMonitorClientEndPath = cms.EndPath(process.l1tMonitorClientEndPathSeq)

#
process.dqmEndPath = cms.EndPath(
                                 process.dqmEnv *
                                 process.dqmSaver
                                 )


process.output = cms.OutputModule(
    "PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
    outputCommands = cms.untracked.vstring('keep *',),
    # outputCommands = cms.untracked.vstring('drop *',
    #                                        'keep *_*_*_L1TEMULATION'),
    fileName = cms.untracked.string('stage1_debug.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string(''),
        dataTier = cms.untracked.string('')
        )
    )

process.output_step = cms.EndPath(process.output)


#
process.schedule = cms.Schedule(process.rawToDigiPath,
                                process.l1tMonitorPath,
                                process.l1tSyncPath,
                                process.l1tMonitorClientPath,
                                process.output_step,
                                process.l1tMonitorEndPath,
                                process.l1tMonitorClientEndPath,
                                process.dqmEndPath
                                )

#---------------------------------------------

# examples for quick fixes in case of troubles
#    please do not modify the commented lines
#


#
# turn on verbosity in L1TEventInfoClient
#
# process.l1tEventInfoClient.verbose = cms.untracked.bool(True)


# remove module(s) or system sequence from l1tMonitorPath
#        quality test disabled also
#
#process.l1tMonitorOnline.remove(process.bxTiming)

process.l1tMonitorOnline.remove(process.l1tBPTX)

#process.l1tMonitorOnline.remove(process.l1Dttf)

#process.l1tMonitorOnline.remove(process.l1tCsctf)

#process.l1tMonitorOnline.remove(process.l1tRpctf)

#process.l1tMonitorOnline.remove(process.l1tGmt)

#process.l1tMonitorOnline.remove(process.l1tGt)

#process.l1tMonitorOnline.remove(process.l1ExtraDqmSeq)

process.l1tMonitorOnline.remove(process.l1tRate)

#process.l1tMonitorOnline.remove(process.l1tRctSeq)

#process.l1tMonitorOnline.remove(process.l1tGctSeq)


# remove module(s) or system sequence from l1tMonitorEndPath
#
#process.l1tMonitorEndPathSeq.remove(process.l1s)
#process.l1tMonitorEndPathSeq.remove(process.l1tscalers)

#
process.schedule.remove(process.l1tSyncPath)


#
# un-comment next lines in case you use the file for private tests using data from the L1T server
#
#process.dqmSaver.dirName = '.'
#process.dqmSaver.saveByRun = 1
#process.dqmSaver.saveAtJobEnd = True


# Message Logger
process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.debugModules = ['l1tGt']

process.MessageLogger.files.L1TDQM_errors = cms.untracked.PSet(
        threshold = cms.untracked.string('ERROR'),
        ERROR = cms.untracked.PSet( limit = cms.untracked.int32(-1) )
       )

process.MessageLogger.files.L1TDQM_warnings = cms.untracked.PSet(
        threshold = cms.untracked.string('WARNING'),
        WARNING = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        ERROR = cms.untracked.PSet( limit = cms.untracked.int32(0) )
        )

process.MessageLogger.files.L1TDQM_info = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO'),
        INFO = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        WARNING = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        ERROR = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        L1TGT = cms.untracked.PSet( limit = cms.untracked.int32(-1) )
        )

process.MessageLogger.files.L1TDQM_debug = cms.untracked.PSet(
        threshold = cms.untracked.string('DEBUG'),
        DEBUG = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        INFO = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        WARNING = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        ERROR = cms.untracked.PSet( limit = cms.untracked.int32(0) )
        )

#--------------------------------------------------
# Heavy Ion Specific Fed Raw Data Collection Label
#--------------------------------------------------

print("Running with run type = ", process.runType.getRunType())
process.castorDigis.InputLabel = cms.InputTag("rawDataCollector")
process.csctfDigis.producer = cms.InputTag("rawDataCollector")
process.dttfDigis.DTTF_FED_Source = cms.InputTag("rawDataCollector")
process.ecalDigis.InputLabel = cms.InputTag("rawDataCollector")
process.ecalPreshowerDigis.sourceTag = cms.InputTag("rawDataCollector")
process.rctDigis.inputLabel = cms.InputTag("rawDataCollector")
process.gctDigis.inputLabel = cms.InputTag("rawDataCollector")
process.gtDigis.DaqGtInputTag = cms.InputTag("rawDataCollector")
process.gtEvmDigis.EvmGtInputTag = cms.InputTag("rawDataCollector")
process.hcalDigis.InputLabel = cms.InputTag("rawDataCollector")
process.muonCSCDigis.InputObjects = cms.InputTag("rawDataCollector")
process.muonDTDigis.inputLabel = cms.InputTag("rawDataCollector")
process.muonRPCDigis.InputLabel = cms.InputTag("rawDataCollector")
process.scalersRawToDigi.scalersInputTag = cms.InputTag("rawDataCollector")
process.siPixelDigis.InputLabel = cms.InputTag("rawDataCollector")
process.siStripDigis.ProductLabel = cms.InputTag("rawDataCollector")
process.bxTiming.FedSource = cms.untracked.InputTag("rawDataCollector")
process.l1s.fedRawData = cms.InputTag("rawDataCollector")

if (process.runType.getRunType() == process.runType.hi_run):
    process.castorDigis.InputLabel = cms.InputTag("rawDataRepacker")
    process.csctfDigis.producer = cms.InputTag("rawDataRepacker")
    process.dttfDigis.DTTF_FED_Source = cms.InputTag("rawDataRepacker")
    process.ecalDigis.InputLabel = cms.InputTag("rawDataRepacker")
    process.ecalPreshowerDigis.sourceTag = cms.InputTag("rawDataRepacker")
    process.rctDigis.inputLabel = cms.InputTag("rawDataRepacker")
    process.gctDigis.inputLabel = cms.InputTag("rawDataRepacker")
    process.gtDigis.DaqGtInputTag = cms.InputTag("rawDataRepacker")
    process.gtEvmDigis.EvmGtInputTag = cms.InputTag("rawDataRepacker")
    process.hcalDigis.InputLabel = cms.InputTag("rawDataRepacker")
    process.muonCSCDigis.InputObjects = cms.InputTag("rawDataRepacker")
    process.muonDTDigis.inputLabel = cms.InputTag("rawDataRepacker")
    process.muonRPCDigis.InputLabel = cms.InputTag("rawDataRepacker")
    process.scalersRawToDigi.scalersInputTag = cms.InputTag("rawDataRepacker")
    process.siPixelDigis.InputLabel = cms.InputTag("rawDataRepacker")
    process.siStripDigis.ProductLabel = cms.InputTag("rawDataRepacker")
    process.bxTiming.FedSource = cms.untracked.InputTag("rawDataRepacker")
    process.l1s.fedRawData = cms.InputTag("rawDataRepacker")


from SLHCUpgradeSimulations.Configuration.postLS1Customs import customise_DQM
#call to customisation function customise_DQM imported from SLHCUpgradeSimulations.Configuration.postLS1Customs
process = customise_DQM(process)
