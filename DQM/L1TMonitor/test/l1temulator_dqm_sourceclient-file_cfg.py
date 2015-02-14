#
# cfg file to run online L1 Trigger emulator DQM
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


process = cms.Process("L1TEmuDQM")

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
    process.EventStreamHttpReader.consumerName = 'L1TEMU DQM Consumer'
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
process.dqmEnv.subSystemFolder = 'L1TEMU'

if l1DqmEnv == 'live' :
    process.load("DQM.Integration.test.environment_cfi")
    # no references needed

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


#process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")

#-------------------------------------
# sequences needed for L1 emulator DQM
#

# standard unpacking sequence 
process.load("Configuration.StandardSequences.RawToDigi_Data_cff")    

# L1 data - emulator sequences 
process.load("DQM.L1TMonitor.L1TEmulatorMonitor_cff")    
process.load("DQM.L1TMonitorClient.L1TEMUMonitorClient_cff")    

#-------------------------------------
# paths & schedule for L1 emulator DQM
#

# TODO define a L1 trigger L1TriggerRawToDigi in the standard sequence 
# to avoid all these remove
process.rawToDigiPath = cms.Path(process.RawToDigi)
#
process.RawToDigi.remove("siPixelDigis")
process.RawToDigi.remove("siStripDigis")
process.RawToDigi.remove("scalersRawToDigi")
process.RawToDigi.remove("castorDigis")

# L1HvVal + emulator monitoring path
process.l1HwValEmulatorMonitorPath = cms.Path(process.l1HwValEmulatorMonitor)

# for RCT at P5, read FED vector from OMDS
if ( l1DqmEnv != 'file' ) : 
    process.load("L1TriggerConfig.RCTConfigProducers.l1RCTOmdsFedVectorProducer_cfi")
    process.valRctDigis.getFedsFromOmds = cms.bool(True)
 
#
process.l1EmulatorMonitorClientPath = cms.Path(process.l1EmulatorMonitorClient)

#
process.l1EmulatorMonitorEndPath = cms.EndPath(process.dqmEnv*process.dqmSaver)

#

#
process.schedule = cms.Schedule(process.rawToDigiPath,
                                process.l1HwValEmulatorMonitorPath,
                                process.l1EmulatorMonitorClientPath,
                                process.l1EmulatorMonitorEndPath)

#---------------------------------------------

# examples for quick fixes in case of troubles 
#    please do not modify the commented lines
#
# remove a module from hardware validation
# cff file: L1Trigger.HardwareValidation.L1HardwareValidation_cff
#
# process.L1HardwareValidation.remove("deCsctf")
#
process.L1HardwareValidation.remove(process.deDt)


#
# remove a L1 trigger system from the comparator integrated in hardware validation
# cfi file: L1Trigger.HardwareValidation.L1Comparator_cfi
# remove (consistently) the same systems from L1TDEMON
# cfi file: DQM.L1TMonitor.L1TDEMON_cfi
#
# 
# process.l1compare.COMPARE_COLLS = [
#        0,  0,  1,  1,   0,  1,  0,  0,  1,  0,  1, 0
#        ]
#    # ETP,HTP,RCT,GCT, DTP,DTF,CTP,CTF,RPC,LTC,GMT,GT
#
# process.l1demon.COMPARE_COLLS = [
#        0,  0,  1,  1,   0,  1,  0,  0,  1,  0,  1, 0
#        ]
#    # ETP,HTP,RCT,GCT, DTP,DTF,CTP,CTF,RPC,LTC,GMT,GT


#
# remove an expert module for L1 trigger system
# cff file: DQM.L1TMonitor.L1TEmulatorMonitor_cff
#
# process.l1ExpertDataVsEmulator.remove(process.l1GtHwValidation)
#

#process.l1ExpertDataVsEmulator.remove(process.l1TdeCSCTF)

#
# remove a module / sequence from l1EmulatorMonitorClient
# cff file: DQM.L1TMonitorClient.L1TEmulatorMonitorClient_cff
#
# process.l1EmulatorMonitorClient.remove(process.l1EmulatorErrorFlagClient)
#


#
# fast over-mask a system in L1TEMUEventInfoClient: 
#   if the name of the system is in the list, the system will be masked
#   (the default mask value is given in L1Systems VPSet)             
#
# names are case sensitive, order is irrelevant
# "ECAL", "HCAL", "RCT", "GCT", "DTTF", "DTTPG", "CSCTF", "CSCTPG", "RPC", "GMT", "GT"
#
# process.l1temuEventInfoClient.MaskL1Systems = cms.vstring("ECAL")
#


#
# fast over-mask an object in L1TEMUEventInfoClient:
#   if the name of the object is in the list, the object will be masked
#   (the default mask value is given in L1Objects VPSet)             
#
# names are case sensitive, order is irrelevant
# 
# "Mu", "NoIsoEG", "IsoEG", "CenJet", "ForJet", "TauJet", "ETM", "ETT", "HTT", "HTM", 
# "HfBitCounts", "HfRingEtSums", "TechTrig", "GtExternal
#
# process.l1temuEventInfoClient.MaskL1Objects =  cms.vstring("ETM")   
#


#
# turn on verbosity in L1TEMUEventInfoClient
#
# process.l1EmulatorEventInfoClient.verbose = cms.untracked.bool(True)


print "Running with run type = ", process.runType.getRunType()
process.castorDigis.InputLabel = cms.InputTag("rawDataCollector")
process.csctfDigis.producer = cms.InputTag("rawDataCollector")
process.dttfDigis.DTTF_FED_Source = cms.InputTag("rawDataCollector")
process.ecalDigis.InputLabel = cms.InputTag("rawDataCollector")
process.ecalPreshowerDigis.sourceTag = cms.InputTag("rawDataCollector")
process.gctDigis.inputLabel = cms.InputTag("rawDataCollector")
process.gtDigis.DaqGtInputTag = cms.InputTag("rawDataCollector")
process.gtEvmDigis.EvmGtInputTag = cms.InputTag("rawDataCollector")
process.hcalDigis.InputLabel = cms.InputTag("rawDataCollector")
process.l1compare.FEDsourceEmul = cms.untracked.InputTag("rawDataCollector")
process.l1compare.FEDsourceData = cms.untracked.InputTag("rawDataCollector")
process.muonCSCDigis.InputObjects = cms.InputTag("rawDataCollector")
process.muonDTDigis.inputLabel = cms.InputTag("rawDataCollector")
process.muonRPCDigis.InputLabel = cms.InputTag("rawDataCollector")
process.scalersRawToDigi.scalersInputTag = cms.InputTag("rawDataCollector")
process.siPixelDigis.InputLabel = cms.InputTag("rawDataCollector")
process.siStripDigis.ProductLabel = cms.InputTag("rawDataCollector")

#--------------------------------------------------
# Heavy Ion Specific Fed Raw Data Collection Label
#--------------------------------------------------
if (process.runType.getRunType() == process.runType.hi_run):
    process.castorDigis.InputLabel = cms.InputTag("rawDataRepacker")
    process.csctfDigis.producer = cms.InputTag("rawDataRepacker")
    process.dttfDigis.DTTF_FED_Source = cms.InputTag("rawDataRepacker")
    process.ecalDigis.InputLabel = cms.InputTag("rawDataRepacker")
    process.ecalPreshowerDigis.sourceTag = cms.InputTag("rawDataRepacker")
    process.gctDigis.inputLabel = cms.InputTag("rawDataRepacker")
    process.gtDigis.DaqGtInputTag = cms.InputTag("rawDataRepacker")
    process.gtEvmDigis.EvmGtInputTag = cms.InputTag("rawDataRepacker")
    process.hcalDigis.InputLabel = cms.InputTag("rawDataRepacker")
    process.l1compare.FEDsourceEmul = cms.untracked.InputTag("rawDataRepacker")
    process.l1compare.FEDsourceData = cms.untracked.InputTag("rawDataRepacker")
    process.muonCSCDigis.InputObjects = cms.InputTag("rawDataRepacker")
    process.muonDTDigis.inputLabel = cms.InputTag("rawDataRepacker")
    process.muonRPCDigis.InputLabel = cms.InputTag("rawDataRepacker")
    process.scalersRawToDigi.scalersInputTag = cms.InputTag("rawDataRepacker")
    process.siPixelDigis.InputLabel = cms.InputTag("rawDataRepacker")
    process.siStripDigis.ProductLabel = cms.InputTag("rawDataRepacker")



