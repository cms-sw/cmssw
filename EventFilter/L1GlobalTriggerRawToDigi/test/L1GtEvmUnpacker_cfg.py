#
# cfg file to unpack RAW L1 GT EVM data
 
# V M Ghete 2009-04-03

import FWCore.ParameterSet.Config as cms
import sys

# process
process = cms.Process('TestL1GtEvmUnpacker')

###################### user choices ######################

# choose (pre)release
useRelease = 'CMSSW_3_3_6'

# choose the type of sample used (True for RelVal, False for data)
#useRelValSample = True 
useRelValSample=False 

# actual GlobalTag must be appropriate for the sample use

if useRelValSample == True :
    
    if useRelease == 'CMSSW_2_2_12' :
        
        useGlobalTag = 'IDEAL_V12'
        #useGlobalTag='STARTUP_V11'
    
    elif useRelease == 'CMSSW_3_1_1' :
         #useGlobalTag = 'MC_31X_V2'
         useGlobalTag = 'STARTUP31X_V1'

    elif useRelease == 'CMSSW_3_3_6' :
        useGlobalTag = 'MC_3XY_V9A'
        #useGlobalTag='STARTUP3X_V8I'
    else :
        print 'Error: release ', useRelease, ' not defined.'    
        sys.exit()
 
    # RelVals 
    useSample = 'RelValTTbar'
       
else :
    # 22X
    #useGlobalTag = 'CRAFT_ALL_V12'
    
    # 31X
    #useGlobalTag = 'CRAFT0831X_V1'
 
    # 333
    runNumber = 123596
    #runNumber == 116035

    useGlobalTag = 'GR09_P_V7'
     

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
    if useGlobalTag.count('IDEAL') or useGlobalTag.count('MC') :

        if (useRelease == 'CMSSW_3_3_6') and (useSample == 'RelValTTbar') :

            dataset = cms.untracked.vstring('/RelValTTbar/CMSSW_3_3_6-MC_3XY_V9A-v1/GEN-SIM-DIGI-RAW-HLTDEBUG')
        
            readFiles.extend([
                '/store/relval/CMSSW_3_3_6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V9A-v1/0009/F6C6F406-3CE4-DE11-8F12-00304867BEE4.root',
                '/store/relval/CMSSW_3_3_6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V9A-v1/0009/B2898985-3BE4-DE11-98B2-00261894396A.root',
                '/store/relval/CMSSW_3_3_6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V9A-v1/0009/ACB9360D-3CE4-DE11-904D-00261894391D.root',
                '/store/relval/CMSSW_3_3_6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V9A-v1/0009/A6ACEC91-3CE4-DE11-A6FB-00261894390E.root'
               ]);

        elif (useRelease == 'CMSSW_2_2_12') and (useSample == 'RelValTTbar') :

            dataset = cms.untracked.vstring('/RelValTTbar/CMSSW_2_2_4_IDEAL_V11_v1/GEN-SIM-DIGI-RAW-HLTDEBUG')
        
            readFiles.extend([
                '/store/relval/CMSSW_2_2_4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V11_v1/0000/02697009-5CF3-DD11-A862-001D09F2423B.root',
                '/store/relval/CMSSW_2_2_4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V11_v1/0000/064657A8-59F3-DD11-ACA5-000423D991F0.root',
                '/store/relval/CMSSW_2_2_4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V11_v1/0000/0817F6DE-5BF3-DD11-880D-0019DB29C5FC.root',
                '/store/relval/CMSSW_2_2_4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V11_v1/0000/0899697C-5AF3-DD11-9D21-001617DBD472.root'
                ]);

        else :
            print 'Error: no files for sample ', useSample, ', (pre)release ', useRelease, ' and global tag ', useGlobalTag, ' defined.'    
            sys.exit()

        secFiles.extend([
            ])

    elif useGlobalTag.count('STARTUP') :

        if (useRelease == 'CMSSW_3_3_6') and (useSample == 'RelValTTbar') :

            dataset = cms.untracked.vstring('/RelValTTbar/CMSSW_3_3_6-STARTUP3X_V8H-v1/GEN-SIM-DIGI-RAW-HLTDEBUG')
        
            readFiles.extend([
                '/store/relval/CMSSW_3_3_6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V8H-v1/0009/E44B9490-3BE4-DE11-962B-0026189437FD.root',
                '/store/relval/CMSSW_3_3_6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V8H-v1/0009/E0BA5492-3BE4-DE11-9417-002618943926.root',
                '/store/relval/CMSSW_3_3_6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V8H-v1/0009/6827DCDF-9EE4-DE11-8A58-002618943920.root'
               ]);

        elif (useRelease == 'CMSSW_2_2_12') and (useSample == 'RelValTTbar') :
            
            dataset = cms.untracked.vstring('/RelValTTbar/CMSSW_2_2_4_STARTUP_V8_v1/GEN-SIM-DIGI-RAW-HLTDEBUG')
        
            readFiles.extend([
                '/store/relval/CMSSW_2_2_4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V8_v1/0000/069AA022-5BF3-DD11-9A56-001617E30D12.root',
                '/store/relval/CMSSW_2_2_4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V8_v1/0000/08DA99A6-5AF3-DD11-AAC1-001D09F24493.root',
                '/store/relval/CMSSW_2_2_4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V8_v1/0000/0A725E15-5BF3-DD11-8B4B-000423D99CEE.root',
                '/store/relval/CMSSW_2_2_4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V8_v1/0000/0AF5B676-5AF3-DD11-A22F-001617DBCF1E.root'
                ]);

        else :
            print 'Error: no files for sample ', useSample, ', (pre)release ', useRelease, ' and global tag ', useGlobalTag, ' defined.'    
            sys.exit()

        secFiles.extend([
            ])
    else :
        print 'Error: Global Tag ', useGlobalTag, ' not defined.'    
        sys.exit()

else : 

    # data

    if runNumber == 123596 :
        dataset = '/Cosmics/BeamCommissioning09-v1/RAW'
    
        readFiles.extend( [
            '/store/data/BeamCommissioning09/Cosmics/RAW/v1/000/123/596/8E21B4C8-74E2-DE11-ABAA-000423D999CA.root' 
            ] );

        secFiles.extend([
            ])

    
    elif runNumber == 116035 :
        dataset = '/Cosmics/Commissioning09-v3/RAW'
        print '   Running on set: '+ dataset + ' with global tag ' +  useGlobalTag 
    
        readFiles.extend( [                        
            '/store/data/Commissioning09/Cosmics/RAW/v3/000/116/035/34A8317D-76AF-DE11-91DB-000423D98DC4.root'
            ]);                                                                                               

        secFiles.extend([
            ])
        
    else :
        print 'Error: run ', runNumber, ' not defined.'    
        sys.exit()
        

if useLocalFiles :
    readFiles = 'file:/afs/cern.ch/user/g/ghete/scratch0/CmsswTestFiles/testGt_L1GtEvmUnpacker_source.root'


# load and configure modules via Global Tag
# https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideFrontierConditions

process.load('Configuration.StandardSequences.Geometry_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.GlobalTag.globaltag = useGlobalTag+'::All'

# L1 GT/GMT EvmUnpack
process.load("EventFilter.L1GlobalTriggerRawToDigi.l1GtEvmUnpack_cfi")

# input tag for GT readout collection: 
#     source        = hardware record

if useRelValSample == True :
    evmGtInputTag = 'rawDataCollector'
else :
    evmGtInputTag = 'source'

process.l1GtEvmUnpack.EvmGtInputTag = evmGtInputTag
#process.l1GtEvmUnpack.EvmGtInputTag = 'l1GtTextToRaw'

# Active Boards Mask

# no board masked (default)
#process.l1GtEvmUnpack.ActiveBoardsMask = 0xFFFF
    
# GTFE only in the record
#process.l1GtEvmUnpack.ActiveBoardsMask = 0x0000
     

# BxInEvent to be EvmUnpacked
# all available BxInEvent (default)
#process.l1GtEvmUnpack.UnpackBxInEvent = -1 

# BxInEvent = 0 (L1A)
#process.l1GtEvmUnpack.UnpackBxInEvent = 1 

# 3 BxInEvent (F, 0, 1)  
#process.l1GtEvmUnpack.UnpackBxInEvent = 3 

# length of BST message (in bytes)
# if negative, take it from event setup
#process.l1GtEvmUnpack.BstLengthBytes = 52

# set it to verbose
process.l1GtEvmUnpack.Verbosity = cms.untracked.int32(1)

# path to be run
process.p = cms.Path(process.l1GtEvmUnpack)

# Message Logger
process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.debugModules = ['l1GtEvmUnpack']
process.MessageLogger.destinations = ['L1GtEvmUnpacker']
process.MessageLogger.L1GtEvmUnpacker = cms.untracked.PSet(
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
    )
)

# summary
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)

# output 

process.outputL1GtEvmUnpack = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('L1GtEvmUnpacker.root'),
    # keep only unpacked data in the ROOT file
    outputCommands = cms.untracked.vstring('drop *', 
        'keep *_l1GtEvmUnpack_*_*')
)

process.outpath = cms.EndPath(process.outputL1GtEvmUnpack)
