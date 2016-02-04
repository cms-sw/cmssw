#
# cfg file to run L1 Global Trigger emulator on a file containing the output of the 
# GCT system and GMT system ("digi"), followed by the L1 trigger report
#
# V M Ghete 2009-03-03

import FWCore.ParameterSet.Config as cms

# process
process = cms.Process('TestL1Gt')


###################### user choices ######################


# choose the type of sample used (True for RelVal, False for data)
useRelValSample = True 
#useRelValSample=False 

if useRelValSample == True :
    useGlobalTag = 'IDEAL_30X'
    #useGlobalTag='STARTUP_30X'
else :
    useGlobalTag = 'CRAFT_ALL_V11'


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


# number of events to be processed and source file
process.maxEvents = cms.untracked.PSet(
    input=cms.untracked.int32(10)
)

readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring() 
process.source = cms.Source ('PoolSource', fileNames=readFiles, secondaryFileNames=secFiles)

# type of sample used (True for RelVal, False for data)

if useRelValSample == True :
    if useGlobalTag.count('IDEAL') :


        # /RelValTTbar/CMSSW_3_1_0_pre4_IDEAL_30X_v1/GEN-SIM-DIGI-RAW-HLTDEBUG
        dataset = cms.untracked.vstring('RelValTTbar_CMSSW_3_1_0_pre4_IDEAL_30X_v1')
        readFiles.extend([
            '/store/relval/CMSSW_3_1_0_pre4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0003/3AA6EEA4-3B16-DE11-B35F-001617C3B654.root',
            '/store/relval/CMSSW_3_1_0_pre4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0003/4250F67F-4C16-DE11-95D4-000423D98DC4.root',
            '/store/relval/CMSSW_3_1_0_pre4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0003/44601F6F-4A16-DE11-B830-001617E30D00.root',
            '/store/relval/CMSSW_3_1_0_pre4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0003/52C2A955-3716-DE11-87D2-000423D99A8E.root'
        ]);

        #/RelValTTbar/CMSSW_2_2_4_IDEAL_V11_v1/GEN-SIM-DIGI-RAW-HLTDEBUG
        #dataset = cms.untracked.vstring('RelValTTbar_CMSSW_2_2_4_IDEAL_V11_v1')

        #readFiles.extend([
        #    '/store/relval/CMSSW_2_2_4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V11_v1/0000/02697009-5CF3-DD11-A862-001D09F2423B.root',
        #    '/store/relval/CMSSW_2_2_4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V11_v1/0000/064657A8-59F3-DD11-ACA5-000423D991F0.root',
        #    '/store/relval/CMSSW_2_2_4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V11_v1/0000/0817F6DE-5BF3-DD11-880D-0019DB29C5FC.root',
        #    '/store/relval/CMSSW_2_2_4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V11_v1/0000/0899697C-5AF3-DD11-9D21-001617DBD472.root'
        #    ]);


        secFiles.extend([
            ])

    elif useGlobalTag.count('STARTUP') :

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
 
# number of BXs in the event corresponding to alternative 0 and 1 in altNrBxBoard()
# EmulateBxInEvent >= max(RecordLength[0], RecordLength[1])
# negative values: take the numbers from event setup, from L1GtParameters
process.l1GtEmulDigis.RecordLength = cms.vint32(3, 5)

# alternative for number of BX per active board in GT DAQ record: 0 or 1
# the position is identical with the active board bit
process.l1GtEmulDigis.AlternativeNrBxBoardDaq = cms.uint32(0x000F)  

# alternative for number of BX per active board in GT EVM record: 0 or 1
# the position is identical with the active board bit
#process.l1GtEmulDigis.AlternativeNrBxBoardEvm = cms.uint32(0x0000)  

# length of BST record (in bytes) from parameter set
# negative value: take the value from EventSetup      
#process.l1GtEmulDigis.BstLengthBytes = 52

# 
process.l1GtEmulDigis.Verbosity = cms.untracked.int32(1)

# load and configure modules via Global Tag
# https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideFrontierConditions

process.load('Configuration.StandardSequences.Geometry_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.GlobalTag.globaltag = useGlobalTag+'::All'

# TEMPORARY remove it after GTag OK

from CondTools.L1Trigger.L1CondDBSource_cff import initCondDBSource
initCondDBSource( process,
    inputDBConnect = 'sqlite_file:/afs/cern.ch/user/w/wsun/public/conddb/l1config31XV2.db',
    tagBase = 'IDEAL')

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
process.MessageLogger.L1GtEmulTrigReport = cms.untracked.PSet(
    threshold=cms.untracked.string('DEBUG'),
    #threshold = cms.untracked.string('INFO'),
    #threshold = cms.untracked.string('ERROR'),
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
        limit=cms.untracked.int32(-1)  
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
