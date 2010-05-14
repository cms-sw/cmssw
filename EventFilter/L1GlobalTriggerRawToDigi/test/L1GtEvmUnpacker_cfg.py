#
# cfg file to unpack RAW L1 GT EVM data
 
# V M Ghete 2009-04-03

import FWCore.ParameterSet.Config as cms

# process
process = cms.Process('TestL1GtEvmUnpacker')

###################### user choices ######################

# (pre)release (cycle) used to produce the event samples
#cmsRunRelease = 'CMSSW_3_6_X'
cmsRunRelease = 'CMSSW_3_5_X'

# choose (pre)release used to produce the event samples
sampleFromRelease = 'CMSSW_3_5_2'
#sampleFromRelease = 'CMSSW_3_3_6'
#sampleFromRelease = 'CMSSW_2_2_12'

# choose the type of sample used:
#   True for RelVal
#   False for data

# default value
useRelValSample = True
#
# comment/uncomment the next line to choose sample type 
# (un-commented selects data)
useRelValSample=False 

# change to True to use local files
#     the type of file should match the choice of useRelValSample or data
#     useGlobalTag must be defined here

useLocalFiles = False 
#useLocalFiles = True 

if useRelValSample == True :
    
    globalTag = 'MC'
    #globalTag = 'START'
    
    # RelVals 
    useSample = 'RelValTTbar'
    
else :

    #runNumber = 123596
    #runNumber = 116035
    #runNumber = 121560
    runNumber = 127715

# change to True to use local files
#     the type of file should match the choice of useRelValSample
#     useGlobalTag must be defined here

useLocalFiles = False 
#useLocalFiles = True 

if (useLocalFiles == True) :
    useGlobalTag = 'GR09_P_V8_34X'
    dataType = 'RECO'
    
# number of events to be run (-1 for all)
maxNumberEvents = 10
#maxNumberEvents = -1

###################### end user choices ###################




# global tags for the release used to run
if (useRelValSample == True) and (useLocalFiles == False) :
    
    if cmsRunRelease == 'CMSSW_3_5_X' :
        if globalTag == 'MC' :
            useGlobalTag = 'MC_3XY_V21'
        else :
            useGlobalTag = 'START3X_V21'
    elif cmsRunRelease == 'CMSSW_3_4_1' :
        if globalTag == 'MC' :
            useGlobalTag = 'MC_3XY_V15'
        else :
            useGlobalTag = 'STARTUP3X_V15'
    elif cmsRunRelease == 'CMSSW_3_3_6' :
        if globalTag == 'MC' :
            useGlobalTag = 'MC_3XY_V9B'
        else :
            useGlobalTag = 'STARTUP3X_V8M'
    else :
        print 'Error: no global tag defined for release ', cmsRunRelease, ' used with RelVal sample'
        sys.exit()
   
elif (useRelValSample == False) and (useLocalFiles == False) :
    # global tag
    
    if cmsRunRelease == 'CMSSW_3_5_X' :
        useGlobalTag = 'GR10_P_V2'
    elif cmsRunRelease == 'CMSSW_3_4_1' :
        useGlobalTag = 'GR09_P_V8_34X'
    elif cmsRunRelease == 'CMSSW_3_3_6' :
        useGlobalTag = 'GR09_P_V8'
    else :
        print 'Error: no global tag defined for release ', cmsRunRelease, ' used with data sample'
        sys.exit()
else :
       print 'Using local file(s) with global tag ',  useGlobalTag, ' and release ', cmsRunRelease
     

# number of events to be processed and source file
process.maxEvents = cms.untracked.PSet(
    input=cms.untracked.int32(maxNumberEvents)
)

readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring() 
process.source = cms.Source ('PoolSource', fileNames=readFiles, secondaryFileNames=secFiles)

# type of sample used (True for RelVal, False for data)

if (useRelValSample == True) and (useLocalFiles == False) :
    if useGlobalTag.count('IDEAL') or useGlobalTag.count('MC') :

        if (sampleFromRelease == 'CMSSW_3_5_2') and (useSample == 'RelValTTbar') :

            dataset = cms.untracked.vstring('/RelValTTbar/CMSSW_3_5_2-MC_3XY_V21-v1/GEN-SIM-DIGI-RAW-HLTDEBUG')
            print '   Running on ', useSample, ' sample produced with ', sampleFromRelease, '. Global tag used to run: ', useGlobalTag  
        
            readFiles.extend( [
                '/store/relval/CMSSW_3_5_2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V21-v1/0016/FAA58A57-3D1E-DF11-87A5-001731A283DF.root',
                '/store/relval/CMSSW_3_5_2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V21-v1/0016/E64BA05D-3A1E-DF11-8861-00261894380D.root',
                '/store/relval/CMSSW_3_5_2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V21-v1/0016/D80D18C7-311E-DF11-93E9-0018F3D09676.root',
                '/store/relval/CMSSW_3_5_2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V21-v1/0016/D27E27EA-391E-DF11-852C-0017319EB92B.root',
                '/store/relval/CMSSW_3_5_2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V21-v1/0016/CC8E2952-391E-DF11-8EE5-0018F3D096D8.root',
                '/store/relval/CMSSW_3_5_2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V21-v1/0016/CACE2BE1-371E-DF11-906C-001731AF685D.root',
                '/store/relval/CMSSW_3_5_2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V21-v1/0016/B6A6D3D4-3C1E-DF11-BCBC-001731AF68B9.root'
                ]);

        elif (sampleFromRelease == 'CMSSW_3_3_6') and (useSample == 'RelValTTbar') :

            dataset = cms.untracked.vstring('/RelValTTbar/CMSSW_3_3_6-MC_3XY_V9A-v1/GEN-SIM-DIGI-RAW-HLTDEBUG')
            print '   Running on ', useSample, ' sample produced with ', sampleFromRelease, '. Global tag used to run: ', useGlobalTag  
        
            readFiles.extend([
                '/store/relval/CMSSW_3_3_6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V9A-v1/0009/F6C6F406-3CE4-DE11-8F12-00304867BEE4.root',
                '/store/relval/CMSSW_3_3_6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V9A-v1/0009/B2898985-3BE4-DE11-98B2-00261894396A.root',
                '/store/relval/CMSSW_3_3_6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V9A-v1/0009/ACB9360D-3CE4-DE11-904D-00261894391D.root',
                '/store/relval/CMSSW_3_3_6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V9A-v1/0009/A6ACEC91-3CE4-DE11-A6FB-00261894390E.root'
               ]);

        elif (sampleFromRelease == 'CMSSW_2_2_12') and (useSample == 'RelValTTbar') :

            dataset = cms.untracked.vstring('/RelValTTbar/CMSSW_2_2_4_IDEAL_V11_v1/GEN-SIM-DIGI-RAW-HLTDEBUG')
            print '   Running on ', useSample, ' sample produced with ', sampleFromRelease, '. Global tag used to run: ', useGlobalTag  
        
            readFiles.extend([
                '/store/relval/CMSSW_2_2_4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V11_v1/0000/02697009-5CF3-DD11-A862-001D09F2423B.root',
                '/store/relval/CMSSW_2_2_4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V11_v1/0000/064657A8-59F3-DD11-ACA5-000423D991F0.root',
                '/store/relval/CMSSW_2_2_4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V11_v1/0000/0817F6DE-5BF3-DD11-880D-0019DB29C5FC.root',
                '/store/relval/CMSSW_2_2_4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V11_v1/0000/0899697C-5AF3-DD11-9D21-001617DBD472.root'
                ]);

        else :
            print 'Error: no files for ', useSample, ' sample produced with ', sampleFromRelease, ' and global tag ', useGlobalTag, ' defined.'    
            sys.exit()

        secFiles.extend([
            ])

    elif useGlobalTag.count('START') :

        if (sampleFromRelease == 'CMSSW_3_5_2') and (useSample == 'RelValTTbar') :

            dataset = cms.untracked.vstring('/RelValTTbar/CMSSW_3_5_2-START3X_V21-v1/GEN-SIM-DIGI-RAW-HLTDEBUG')
            print '   Running on ', useSample, ' sample produced with ', sampleFromRelease, '. Global tag used to run: ', useGlobalTag  
        
            readFiles.extend( [
                '/store/relval/CMSSW_3_5_2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V21-v1/0016/FE2573D6-381E-DF11-9B55-001731AF678D.root',
                '/store/relval/CMSSW_3_5_2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V21-v1/0016/F687B04E-331E-DF11-B1C3-0018F3D0960E.root',
                '/store/relval/CMSSW_3_5_2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V21-v1/0016/D2E9E5C7-2D1E-DF11-AA1D-003048679296.root',
                '/store/relval/CMSSW_3_5_2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V21-v1/0016/D0B6CCE8-321E-DF11-ABEE-001A92971BD6.root',
                '/store/relval/CMSSW_3_5_2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V21-v1/0016/AE6887CC-2C1E-DF11-90DD-001A928116E0.root',
                '/store/relval/CMSSW_3_5_2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V21-v1/0016/9E058FDD-361E-DF11-A10F-0017313F02F2.root',
                '/store/relval/CMSSW_3_5_2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V21-v1/0016/9098C5D7-2B1E-DF11-8AD9-001A92971AD8.root'
                ]);

        elif (sampleFromRelease == 'CMSSW_3_3_6') and (useSample == 'RelValTTbar') :

            dataset = cms.untracked.vstring('/RelValTTbar/CMSSW_3_3_6-STARTUP3X_V8H-v1/GEN-SIM-DIGI-RAW-HLTDEBUG')
            print '   Running on ', useSample, ' sample produced with ', sampleFromRelease, '. Global tag used to run: ', useGlobalTag  
        
            readFiles.extend([
                '/store/relval/CMSSW_3_3_6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V8H-v1/0009/E44B9490-3BE4-DE11-962B-0026189437FD.root',
                '/store/relval/CMSSW_3_3_6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V8H-v1/0009/E0BA5492-3BE4-DE11-9417-002618943926.root',
                '/store/relval/CMSSW_3_3_6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V8H-v1/0009/6827DCDF-9EE4-DE11-8A58-002618943920.root'
               ]);

        elif (sampleFromRelease == 'CMSSW_2_2_12') and (useSample == 'RelValTTbar') :
            
            dataset = cms.untracked.vstring('/RelValTTbar/CMSSW_2_2_4_STARTUP_V8_v1/GEN-SIM-DIGI-RAW-HLTDEBUG')
            print '   Running on ', useSample, ' sample produced with ', sampleFromRelease, '. Global tag used to run: ', useGlobalTag  
        
            readFiles.extend([
                '/store/relval/CMSSW_2_2_4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V8_v1/0000/069AA022-5BF3-DD11-9A56-001617E30D12.root',
                '/store/relval/CMSSW_2_2_4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V8_v1/0000/08DA99A6-5AF3-DD11-AAC1-001D09F24493.root',
                '/store/relval/CMSSW_2_2_4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V8_v1/0000/0A725E15-5BF3-DD11-8B4B-000423D99CEE.root',
                '/store/relval/CMSSW_2_2_4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V8_v1/0000/0AF5B676-5AF3-DD11-A22F-001617DBCF1E.root'
                ]);

        else :
            print 'Error: no files for ', useSample, ' sample produced with ', sampleFromRelease, ' and global tag ', useGlobalTag, ' defined.'    
            sys.exit()

        secFiles.extend([
            ])
    else :
        print 'Error: Global Tag ', useGlobalTag, ' not defined.'    
        sys.exit()

elif (useRelValSample == False) and (useLocalFiles == False) :

    # data

    if runNumber == 123596 :
        dataset = '/Cosmics/BeamCommissioning09-v1/RAW'
        print '   Running on set: '+ dataset + ' with global tag ' +  useGlobalTag 
    
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
        
    elif runNumber == 121560 :
        dataset = '/Cosmics/Commissioning09-v3/RAW'
        print '   Running on set: '+ dataset + ' with global tag ' +  useGlobalTag 
    
        readFiles.extend( [                        
            '/store/data/BeamCommissioning09/Cosmics/RAW/v1/000/121/560/DC089E4B-5ED4-DE11-A179-000423D98FBC.root'
            ]);                                                                                               

        secFiles.extend([
            ])

    elif runNumber == 127715 :
        dataset = '/Cosmics/Commissioning10-v3/RAW'
        print '   Running on set: '+ dataset + ' with global tag ' +  useGlobalTag 
    
        readFiles.extend( [                        
            '/store/data/Commissioning10/Cosmics/RAW/v3/000/127/715/FCB12D5F-6C18-DF11-AB4B-000423D174FE.root'
            ]);                                                                                               

        secFiles.extend([
            ])

    else :
        print 'Error: run ', runNumber, ' not defined.'    
        sys.exit()
else :
    readFiles.extend( [                        
        'file:/afs/cern.ch/user/g/ghete/scratch0/CmsswTestFiles/testGt_L1GtEvmUnpacker_source.root'
        ]);                                                                                               

    secFiles.extend([
        ])

    print 'Local file(s) ', readFiles
        


# load and configure modules via Global Tag
# https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideFrontierConditions

process.load('Configuration.StandardSequences.Geometry_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.GlobalTag.globaltag = useGlobalTag+'::All'
# 22X
#process.load('L1Trigger.Configuration.L1Trigger_FakeConditions_cff')

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
