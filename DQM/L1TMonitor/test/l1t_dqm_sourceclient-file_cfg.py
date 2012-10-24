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

# for 'file' choose also the type of the global tag and (edit) the actual global tag
if l1DqmEnv == 'file' :
    
    globalTagType = 'HLT'
    #globalTagType = 'P'
    #globalTagType = 'E'
    #globalTagType = 'R'
    
    if globalTagType == 'HLT' :
        globalTagValue = 'GR_H_V26'
    elif globalTagType == 'P' :
        globalTagValue = 'GR_P_V29'
    elif globalTagType == 'E' :
        globalTagValue = 'GR_E_V23'
    elif globalTagType == 'R' :
        globalTagValue = 'GR_R_52_V4'
    else :
        print 'No valid global tag type', globalTagType
        print 'Valid types: HLT, P, E, R'
        sys.exit()


process = cms.Process("DQM")

# check that a valid choice for environment exists

if not ((l1DqmEnv == 'live') or l1DqmEnv == 'playback' or l1DqmEnv == 'file-P5' or l1DqmEnv == 'file' ) : 
    print 'No valid input source was chosen. Your value for l1DqmEnv input parameter is:'  
    print 'l1DqmEnv = ', l1DqmEnv
    print 'Available options: "live", "playback", "file-P5", "file" '
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
    print 'FIXME'
    sys.exit()
    
else : 
    # running on a file
    process.load("DQM.L1TMonitor.inputsource_file_cfi")
    
      
#----------------------------
# DQM Environment
#

process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")
process.dqmEnv.subSystemFolder = 'L1T'

if l1DqmEnv == 'live' :
    process.load("DQM.Integration.test.environment_cfi")
    process.DQMStore.referenceFileName = "/dqmdata/dqm/reference/l1t_reference.root"

    #
    # load and configure modules via Global Tag
    # https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideFrontierConditions
    process.load("DQM.Integration.test.FrontierCondition_GT_cfi")
    es_prefer_GlobalTag = cms.ESPrefer('GlobalTag')
    process.GlobalTag.RefreshEachRun = cms.untracked.bool(True)

elif l1DqmEnv == 'playback' :
    print 'FIXME'
    
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


#process.load("Configuration.StandardSequences.Geometry_cff")
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
    process.l1tSync.oracleDB   = cms.string("oracle://cms_orcon_adg/CMS_COND_31X_L1T")
    process.l1tRate.pathCondDB = cms.string("/afs/cern.ch/cms/DB/conddb")        #

    process.l1tRate.oracleDB   = cms.string("oracle://cms_orcon_adg/CMS_COND_31X_L1T")
    process.l1tRate.pathCondDB = cms.string("/afs/cern.ch/cms/DB/conddb")    

    process.l1tBPTX.oracleDB   = cms.string("oracle://cms_orcon_adg/CMS_COND_31X_L1T")
    process.l1tBPTX.pathCondDB = cms.string("/afs/cern.ch/cms/DB/conddb")

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

# 
process.l1tMonitorPath = cms.Path(process.l1tMonitorOnline)

# separate L1TSync path due to the use of the HltHighLevel filter
process.l1tSyncPath = cms.Path(process.l1tSyncHltFilter+process.l1tSync)

#
process.l1tMonitorClientPath = cms.Path(process.l1tMonitorClient)

#
process.l1tMonitorEndPath = cms.EndPath(process.l1tMonitorEndPathSeq)

#
process.l1tMonitorClientEndPath = cms.EndPath(process.l1tMonitorClientEndPathSeq)

#
process.dqmEndPath = cms.EndPath(
                                 process.dqmEnv *
                                 process.dqmSaver
                                 )

#
process.schedule = cms.Schedule(process.rawToDigiPath,
                                process.l1tMonitorPath,
                                process.l1tSyncPath,
                                process.l1tMonitorClientPath,
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

#process.l1tMonitorOnline.remove(process.l1Dttf)

#process.l1tMonitorOnline.remove(process.l1tCsctf) 

#process.l1tMonitorOnline.remove(process.l1tRpctf)

#process.l1tMonitorOnline.remove(process.l1tGmt)

#process.l1tMonitorOnline.remove(process.l1tGt) 

#process.l1tMonitorOnline.remove(process.l1ExtraDqmSeq)

#process.l1tMonitorOnline.remove(process.l1tRate)

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
process.MessageLogger.categories.append('L1TGT')
process.MessageLogger.destinations = ['L1TDQM_errors', 
                                      'L1TDQM_warnings', 
                                      'L1TDQM_info', 
                                      'L1TDQM_debug'
                                      ]

process.MessageLogger.L1TDQM_errors = cms.untracked.PSet( 
        threshold = cms.untracked.string('ERROR'),
        ERROR = cms.untracked.PSet( limit = cms.untracked.int32(-1) )
       )

process.MessageLogger.L1TDQM_warnings = cms.untracked.PSet( 
        threshold = cms.untracked.string('WARNING'),
        WARNING = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        ERROR = cms.untracked.PSet( limit = cms.untracked.int32(0) )
        )

process.MessageLogger.L1TDQM_info = cms.untracked.PSet( 
        threshold = cms.untracked.string('INFO'),
        INFO = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        WARNING = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        ERROR = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        L1TGT = cms.untracked.PSet( limit = cms.untracked.int32(-1) )
        )

process.MessageLogger.L1TDQM_debug = cms.untracked.PSet( 
        threshold = cms.untracked.string('DEBUG'),
        DEBUG = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        INFO = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        WARNING = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        ERROR = cms.untracked.PSet( limit = cms.untracked.int32(0) )
        )

#--------------------------------------------------
# Heavy Ion Specific Fed Raw Data Collection Label
#--------------------------------------------------

print "Running with run type = ", process.runType.getRunType()
process.castorDigis.InputLabel           = cms.InputTag("rawDataCollector")
process.csctfDigis.producer              = cms.InputTag("rawDataCollector")
process.dttfDigis.DTTF_FED_Source        = cms.InputTag("rawDataCollector")
process.ecalDigis.InputLabel             = cms.InputTag("rawDataCollector")
process.ecalPreshowerDigis.sourceTag     = cms.InputTag("rawDataCollector")
process.gctDigis.inputLabel              = cms.InputTag("rawDataCollector")
process.gtDigis.DaqGtInputTag            = cms.InputTag("rawDataCollector")
process.gtEvmDigis.EvmGtInputTag         = cms.InputTag("rawDataCollector")
process.hcalDigis.InputLabel             = cms.InputTag("rawDataCollector")
process.muonCSCDigis.InputObjects        = cms.InputTag("rawDataCollector")
process.muonDTDigis.inputLabel           = cms.InputTag("rawDataCollector")
process.muonRPCDigis.InputLabel          = cms.InputTag("rawDataCollector")
process.scalersRawToDigi.scalersInputTag = cms.InputTag("rawDataCollector")
process.siPixelDigis.InputLabel          = cms.InputTag("rawDataCollector")
process.siStripDigis.ProductLabel        = cms.InputTag("rawDataCollector")
process.bxTiming.FedSource               = cms.untracked.InputTag("rawDataCollector")
process.l1s.fedRawData                   = cms.InputTag("rawDataCollector")
    
if (process.runType.getRunType() == process.runType.hi_run):
    process.castorDigis.InputLabel           = cms.InputTag("rawDataRepacker")
    process.csctfDigis.producer              = cms.InputTag("rawDataRepacker")
    process.dttfDigis.DTTF_FED_Source        = cms.InputTag("rawDataRepacker")
    process.ecalDigis.InputLabel             = cms.InputTag("rawDataRepacker")
    process.ecalPreshowerDigis.sourceTag     = cms.InputTag("rawDataRepacker")
    process.gctDigis.inputLabel              = cms.InputTag("rawDataRepacker")
    process.gtDigis.DaqGtInputTag            = cms.InputTag("rawDataRepacker")
    process.gtEvmDigis.EvmGtInputTag         = cms.InputTag("rawDataRepacker")
    process.hcalDigis.InputLabel             = cms.InputTag("rawDataRepacker")
    process.muonCSCDigis.InputObjects        = cms.InputTag("rawDataRepacker")
    process.muonDTDigis.inputLabel           = cms.InputTag("rawDataRepacker")
    process.muonRPCDigis.InputLabel          = cms.InputTag("rawDataRepacker")
    process.scalersRawToDigi.scalersInputTag = cms.InputTag("rawDataRepacker")
    process.siPixelDigis.InputLabel          = cms.InputTag("rawDataRepacker")
    process.siStripDigis.ProductLabel        = cms.InputTag("rawDataRepacker")
    process.bxTiming.FedSource               = cms.untracked.InputTag("rawDataRepacker")
    process.l1s.fedRawData                   = cms.InputTag("rawDataRepacker")


