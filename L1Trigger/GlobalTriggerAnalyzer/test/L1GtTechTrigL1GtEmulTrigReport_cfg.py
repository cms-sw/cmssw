from __future__ import print_function
#
# cfg file to run some technical trigger emulators/simulation, L1 Global Trigger emulator 
# on a file containing the output of the  GCT system and GMT system ("digi"), followed by 
# the L1 trigger report
#
# V M Ghete 2009-03-04


import FWCore.ParameterSet.Config as cms

# process
process = cms.Process('TestL1Gt')

###################### user choices ######################

# choose the type of sample used (True for RelVal, False for data)
useRelValSample = True 
#useRelValSample=False 

# release to be used
useRelease = 'CMSSW_3_4_X'


if useRelValSample == True :
    
    if useRelease == 'CMSSW_2_2_12' :
        
        useGlobalTag = 'IDEAL_V12'
        #useGlobalTag='STARTUP_V11'
    
    elif useRelease == 'CMSSW_3_1_1' :
         useGlobalTag = 'MC_31X_V2'
         #useGlobalTag = 'STARTUP31X_V1'

    elif useRelease == 'CMSSW_3_4_X' :
         useGlobalTag = 'MC_3XY_V10'
         #useGlobalTag = 'STARTUP3X_V9'

    # RelVals 
    useSample = 'RelValTTbar'
    #useSample = 'RelValZTT'
    
else :
    # global tag for data
    
    if useRelease == 'CMSSW_2_2_12' :
        useGlobalTag = 'CRAFT_ALL_V11'
        
    elif useRelease == 'CMSSW_3_1_1' :
        #useGlobalTag = 'CRAFT0831X_V1'
        useGlobalTag = 'GR09_31X_V1P'


# explicit choice of the L1 menu - available choices:
# use menu from global tag
l1Menu =''
# use selected menu
#l1Menu = 'L1Menu_Commissioning2009_v5'
#l1Menu = 'L1Menu_MC2009_v0'
#l1Menu = 'L1Menu_startup2_v4'
#l1Menu = 'L1Menu_2008MC_2E30'
#l1Menu = 'myMenu'

# change to True to use local files
#     the type of file should match the choice of useRelValSample and useGlobalTag
useLocalFiles = False 

###################### end user choices ###################


# number of events to be processed and source file
process.maxEvents = cms.untracked.PSet(
    input=cms.untracked.int32(2000)
)

readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring() 
process.source = cms.Source ('PoolSource', fileNames=readFiles, secondaryFileNames=secFiles)

# type of sample used (True for RelVal, False for data)

if useRelValSample == True :
    if useGlobalTag.count('IDEAL') or useGlobalTag.count('MC') :
        
        if (useRelease == 'CMSSW_3_4_X') and (useSample == 'RelValTTbar') :

            dataset = cms.untracked.vstring('/RelValTTbar/CMSSW_3_4_0_pre2-MC_3XY_V10-v1/GEN-SIM-DIGI-RAW-HLTDEBUG')       
            readFiles.extend([
                '/store/relval/CMSSW_3_4_0_pre2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V10-v1/0003/F4AC0278-96BD-DE11-8687-00261894392D.root',
                '/store/relval/CMSSW_3_4_0_pre2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V10-v1/0003/E2E078D8-C0BD-DE11-A77B-0026189438E0.root',
                '/store/relval/CMSSW_3_4_0_pre2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V10-v1/0003/E0D7EB2D-90BD-DE11-9BB6-0017312B5651.root'
                ]);


            secFiles.extend([
                ])

        else : 
            print('Error: no file list defined for release ', useRelease, ' global tag ', useGlobalTag, ' data sample ',  useSample)   
            sys.exit()
            
    elif useGlobalTag.count('STARTUP') :

        if (useRelease == 'CMSSW_3_4_X') and (useSample == 'RelValBeamHalo') :

            dataset = cms.untracked.vstring('/RelValBeamHalo/CMSSW_3_4_0_pre2-STARTUP3XY_V9-v1/GEN-SIM-DIGI-RAW-HLTDEBUG')
            readFiles.extend( [
                '/store/relval/CMSSW_3_4_0_pre2/RelValBeamHalo/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3XY_V9-v1/0003/BC5F1CA5-C0BD-DE11-B4BD-002618943982.root',
                '/store/relval/CMSSW_3_4_0_pre2/RelValBeamHalo/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3XY_V9-v1/0003/663EED3C-89BD-DE11-97CF-001A92971B8E.root'
                ]);


            secFiles.extend([
                ])
        else :
            print('Error: no file list defined for release ', useRelease, ' global tag ', useGlobalTag, ' data sample ',  useSample)   
            sys.exit()

else : 

    # CRAFT data FIXME
    print('Error: no file list defined for release ', useRelease, ' global tag ', useGlobalTag, ' data sample ',  useSample)   
    sys.exit()
    
    dataset = ''
    
    readFiles.extend([
        ]);

    secFiles.extend([
        ])

if useLocalFiles :
    readFiles = 'file:/afs/cern.ch/user/g/ghete/scratch0/CmsswTestFiles/testGt_L1GtTrigReport_source.root'

# technical triggers to be run and collected by GT

# producers for Technical Trigger
#
# BSC Trigger
import L1TriggerOffline.L1Analyzer.bscTrigger_cfi
process.simBscDigis = L1TriggerOffline.L1Analyzer.bscTrigger_cfi.bscTrigger.clone()

# RPC Technical Trigger
import L1Trigger.RPCTechnicalTrigger.rpcTechnicalTrigger_cfi
process.simRpcTechTrigDigis = L1Trigger.RPCTechnicalTrigger.rpcTechnicalTrigger_cfi.rpcTechnicalTrigger.clone()
process.simRpcTechTrigDigis.RPCDigiLabel = 'simMuonRPCDigis'

# Global Trigger emulator
import L1Trigger.GlobalTrigger.gtDigis_cfi
process.l1GtEmulDigis = L1Trigger.GlobalTrigger.gtDigis_cfi.gtDigis.clone()


# input tag for GMT readout collection: 
#     gmtDigis = GMT emulator (default)
#     l1GtUnpack     = GT unpacker (common GT/GMT unpacker)
process.l1GtEmulDigis.GmtInputTag = 'simGmtDigis'

# input tag for GCT readout collections: 
#     gctDigis = GCT emulator (default) 
process.l1GtEmulDigis.GctInputTag = 'simGctDigis'

# input tag for CASTOR record 
#     castorL1Digis =  CASTOR
#process.l1GtEmulDigis.CastorInputTag = cms.InputTag("castorL1Digis")
    
# technical triggers: a vector of input tags, one tag per each technical 
# trigger producer 
# 
# by default: empty vector
    
# Example:
# TechnicalTriggersInputTags = cms.VInputTag(cms.InputTag('aTechTrigDigis'), 
#                                            cms.InputTag('anotherTechTriggerDigis')),
process.l1GtEmulDigis.TechnicalTriggersInputTags = cms.VInputTag(cms.InputTag('simBscDigis'), 
                                                                 cms.InputTag('simRpcTechTrigDigis'))

# logical flag to produce the L1 GT DAQ readout record
#     if true, produce the record (default)
#process.l1GtEmulDigis.ProduceL1GtDaqRecord = False
    
# logical flag to produce the L1 GT EVM readout record
#     if true, produce the record (default)
#process.l1GtEmulDigis.ProduceL1GtEvmRecord = False

# logical flag to produce the L1 GT object map record
#     if true, produce the record (default)
#process.l1GtEmulDigis.ProduceL1GtObjectMapRecord = False

# logical flag to write the PSB content in the  L1 GT DAQ record
#     if true, write the PSB content in the record (default)
#process.l1GtEmulDigis.WritePsbL1GtDaqRecord = False

# logical flag to read the technical trigger records
#     if true, it will read via getMany the available records (default)
#process.l1GtEmulDigis.ReadTechnicalTriggerRecords = False

# number of "bunch crossing in the event" (BxInEvent) to be emulated
# symmetric around L1Accept (BxInEvent = 0):
#    1 (BxInEvent = 0); 3 (F 0 1) (standard record); 5 (E F 0 1 2) (debug record)
# even numbers (except 0) "rounded" to the nearest lower odd number
# negative value: emulate TotalBxInEvent as given in EventSetup  
#process.l1GtEmulDigis.EmulateBxInEvent = 3

# length of BST record (in bytes) from parameter set
# negative value: take the value from EventSetup      
#process.l1GtEmulDigis.BstLengthBytes = 52

# load and configure modules via Global Tag
# https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideFrontierConditions

process.load("Configuration.StandardSequences.GeometryDB_cff")
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.GlobalTag.globaltag = useGlobalTag + '::All'


# explicit choice of the L1 menu

if l1Menu == '' :
    print('Using menu from global tag', useGlobalTag)

elif l1Menu == 'L1Menu_Commissioning2009_v5' :
    process.load('L1Trigger.Configuration.L1StartupConfig_cff')
    process.load('L1TriggerConfig.L1GtConfigProducers.Luminosity.startup.L1Menu_Commissioning2009_v5_L1T_Scales_20080926_startup_Imp0_Unprescaled_cff')
    
elif l1Menu == 'L1Menu_startup2_v4' :
    process.load('L1Trigger.Configuration.L1StartupConfig_cff')
    process.load('L1TriggerConfig.L1GtConfigProducers.Luminosity.startup.L1Menu_startup2_v4_L1T_Scales_20080926_startup_Imp0_Unprescaled_cff')
    
elif l1Menu == 'L1Menu_MC2009_v0' :
    process.load('L1TriggerConfig.L1GtConfigProducers.Luminosity.lumi1031.L1Menu_MC2009_v0_L1T_Scales_20080922_Imp0_Unprescaled_cff')

elif l1Menu == 'L1Menu_2008MC_2E30' :
    process.load("L1TriggerConfig.L1GtConfigProducers.Luminosity.lumi1030.L1Menu_2008MC_2E30_Unprescaled_cff")

elif l1Menu == 'myMenu' :
    #process.load("L1TriggerConfig.L1GtConfigProducers.Luminosity.startup.L1Menu_startup_v3_Unprescaled_cff")
    #process.load("L1TriggerConfig.L1GtConfigProducers.Luminosity.startup.L1Menu_startup_v4_Unprescaled_cff")
    #process.load("L1TriggerConfig.L1GtConfigProducers.Luminosity.startup.L1Menu_startup_v5_Unprescaled_cff")
    #process.load("L1TriggerConfig.L1GtConfigProducers.Luminosity.startup.L1Menu_startup2_v1_Unprescaled_cff")
    #process.load("L1TriggerConfig.L1GtConfigProducers.Luminosity.startup.L1Menu_startup2_v2_Unprescaled_cff")
    process.load("L1TriggerConfig.L1GtConfigProducers.Luminosity.startup.L1Menu_startup2_v3_Unprescaled_cff")
else :
    print('No such L1 menu: ', l1Menu)  
      

#
# l1GtTrigReport module
#

process.load("L1Trigger.GlobalTriggerAnalyzer.l1GtTrigReport_cfi")

# boolean flag to select the input record
# if true, it will use L1GlobalTriggerRecord 
#process.l1GtTrigReport.UseL1GlobalTriggerRecord = True

# input tag for GT record: 
#   GT emulator:    gtDigis (DAQ record)
#   GT unpacker:    gtDigis (DAQ record)
#   GT lite record: l1GtRecord 
process.l1GtTrigReport.L1GtRecordInputTag = "l1GtEmulDigis"

#process.l1GtTrigReport.PrintVerbosity = 10
#process.l1GtTrigReport.PrintOutput = 1

# path to be run
process.p = cms.Path(process.simBscDigis 
                     + process.simRpcTechTrigDigis 
                     + process.l1GtEmulDigis 
                     + process.l1GtTrigReport)

# Message Logger
process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.debugModules = ['l1GtEmulDigis', 'l1GtTrigReport']

process.MessageLogger.cerr.threshold = 'DEBUG'
#process.MessageLogger.cerr.threshold = 'INFO'
#process.MessageLogger.cerr.threshold = 'WARNING'
#process.MessageLogger.cerr.threshold = 'ERROR'

process.MessageLogger.cerr.DEBUG = cms.untracked.PSet( limit = cms.untracked.int32(0) )
process.MessageLogger.cerr.INFO = cms.untracked.PSet( limit = cms.untracked.int32(0) )
process.MessageLogger.cerr.WARNING = cms.untracked.PSet( limit = cms.untracked.int32(0) )
process.MessageLogger.cerr.ERROR = cms.untracked.PSet( limit = cms.untracked.int32(0) )

process.MessageLogger.cerr.L1GtTrigReport = cms.untracked.PSet( limit = cms.untracked.int32(-1) )

# summary
process.options = cms.untracked.PSet(
    wantSummary=cms.untracked.bool(True)
)

# output 

# keep GMT, GCT and GT output records
# update labels if InputTags are replaced above

process.outputL1GlobalTrigger = cms.OutputModule("PoolOutputModule",
    fileName=cms.untracked.string('L1GtTechTrigL1GtEmulTrigReport.root'),
    outputCommands=cms.untracked.vstring('drop *',
        'keep *_simGmtDigis_*_*',
        'keep *_simGctDigis_*_*',
        'keep *_l1GtEmulDigis_*_*')
)

process.outpath = cms.EndPath(process.outputL1GlobalTrigger)
