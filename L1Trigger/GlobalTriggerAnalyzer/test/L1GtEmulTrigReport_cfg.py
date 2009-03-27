#
# cfg file to run L1 Global Trigger emulator on a file containing the output of the 
# GCT system and GMT system ("digi"), followed by the L1 trigger report
#
# V M Ghete 2009-03-03

###################### user choices ######################


# choose the type of sample used (True for RelVal, False for data)
useRelValSample = True 
#useRelValSample=False 

# choose the global tag type for RelVal; 
#     actual GlobalTag must be replaced in the "if" below 
useGlobalTag = 'IDEAL'
#useGlobalTag='STARTUP'

# explicit choice of the L1 menu - available choices:
#l1Menu = 'L1Menu_Commissioning2009_v0'
l1Menu = 'L1Menu_MC2009_v0'
#l1Menu = 'L1Menu_startup2_v4'
#l1Menu = 'L1Menu_2008MC_2E30'
#l1Menu = 'myMenu'

# change to True to use local files
#     the type of file should match the choice of useRelValSample and useGlobalTag
useLocalFiles = False 

###################### end user choices ###################


import FWCore.ParameterSet.Config as cms

# process
process = cms.Process('TestL1Gt')

# number of events to be processed and source file
process.maxEvents = cms.untracked.PSet(
    input=cms.untracked.int32(200)
)

readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring() 
process.source = cms.Source ('PoolSource', fileNames=readFiles, secondaryFileNames=secFiles)

# type of sample used (True for RelVal, False for data)

if useRelValSample == True :
    if useGlobalTag == 'IDEAL':

        #/RelValTTbar/CMSSW_2_2_4_IDEAL_V11_v1/GEN-SIM-DIGI-RAW-HLTDEBUG
        dataset = cms.untracked.vstring('RelValTTbar_CMSSW_2_2_4_IDEAL_V11_v1')
        
        readFiles.extend([
            '/store/relval/CMSSW_2_2_4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V11_v1/0000/02697009-5CF3-DD11-A862-001D09F2423B.root',
            '/store/relval/CMSSW_2_2_4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V11_v1/0000/064657A8-59F3-DD11-ACA5-000423D991F0.root',
            '/store/relval/CMSSW_2_2_4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V11_v1/0000/0817F6DE-5BF3-DD11-880D-0019DB29C5FC.root',
            '/store/relval/CMSSW_2_2_4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V11_v1/0000/0899697C-5AF3-DD11-9D21-001617DBD472.root'
            ]);


        secFiles.extend([
            ])

    elif useGlobalTag == 'STARTUP':

        #/RelValTTbar/CMSSW_2_2_4_STARTUP_V8_v1/GEN-SIM-DIGI-RAW-HLTDEBUG
        dataset = cms.untracked.vstring('RelValTTbar_CMSSW_2_2_4_STARTUP_V8_v1')
        
        readFiles.extend([
            '/store/relval/CMSSW_2_2_4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V8_v1/0000/069AA022-5BF3-DD11-9A56-001617E30D12.root',
            '/store/relval/CMSSW_2_2_4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V8_v1/0000/08DA99A6-5AF3-DD11-AAC1-001D09F24493.root',
            '/store/relval/CMSSW_2_2_4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V8_v1/0000/0A725E15-5BF3-DD11-8B4B-000423D99CEE.root',
            '/store/relval/CMSSW_2_2_4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V8_v1/0000/0AF5B676-5AF3-DD11-A22F-001617DBCF1E.root'
            ]);


        secFiles.extend([
            ])
    else :
        print 'Error: Global Tag ', useGlobalTag, ' not defined.'    

else : 

    # CRAFT data FIXME
    dataset = ''
    
    readFiles.extend([
        ]);

    secFiles.extend([
        ])

if useLocalFiles :
    readFiles = 'file:/afs/cern.ch/user/g/ghete/scratch0/CmsswTestFiles/testGt_L1GtTrigReport_source.root'


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
process.l1GtEmulDigis.TechnicalTriggersInputTags = cms.VInputTag(cms.InputTag('bscTrigger'))

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

process.load('Configuration.StandardSequences.Geometry_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

if useRelValSample == True :
    if useGlobalTag == 'IDEAL':
        process.GlobalTag.globaltag = 'IDEAL_V11::All'

    elif useGlobalTag == 'STARTUP':
        process.GlobalTag.globaltag = 'STARTUP_V8::All'

    else :
        print 'Error: Global Tag ', useGlobalTag, ' not defined.'    

else :
    process.GlobalTag.globaltag = 'CRAFT_ALL_V8::All'


# explicit choice of the L1 menu

if l1Menu == 'L1Menu_Commissioning2009_v0' :
    process.load('L1Trigger.Configuration.L1StartupConfig_cff')
    process.load('L1TriggerConfig.L1GtConfigProducers.Luminosity.startup.L1Menu_Commissioning2009_v0_L1T_Scales_20080926_startup_Imp0_Unprescaled_cff')
    
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
    print 'No such L1 menu: ', l1Menu  
      

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
process.p = cms.Path(process.l1GtEmulDigis*process.l1GtTrigReport)


# Message Logger
process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.debugModules = ['l1GtEmulDigis', 'l1GtTrigReport']
process.MessageLogger.categories.append('L1GlobalTrigger')
process.MessageLogger.destinations = ['L1GtEmulTrigReport']
process.MessageLogger.cout = cms.untracked.PSet(
    #threshold=cms.untracked.string('DEBUG'),
    #threshold = cms.untracked.string('INFO'),
    threshold = cms.untracked.string('ERROR'),
    DEBUG=cms.untracked.PSet(
        limit=cms.untracked.int32(-1)
    ),
    INFO=cms.untracked.PSet(
        limit=cms.untracked.int32(-1)
    ),
    WARNING=cms.untracked.PSet(
        limit=cms.untracked.int32(-1)
    ),
    ERROR=cms.untracked.PSet(
        limit=cms.untracked.int32(-1)
    ),
    default = cms.untracked.PSet( 
        limit=cms.untracked.int32(0)  
    ),
    L1GlobalTrigger = cms.untracked.PSet( 
        limit=cms.untracked.int32(-1)  
    )
)

# summary
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)

# output 

# keep GMT, GCT and GT output records
# update labels if InputTags are replaced above

process.outputL1GlobalTrigger = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testGt_Emulator_GctGmtFile_output.root'),
    outputCommands = cms.untracked.vstring('drop *', 
        'keep *_simGmtDigis_*_*', 
        'keep *_simGctDigis_*_*', 
        'keep *_l1GtEmulDigis_*_*')
)

process.outpath = cms.EndPath(process.outputL1GlobalTrigger)
