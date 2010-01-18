#
# cfg file to run the L1 GT test analyzer according to 
#   the options set in "user choices"
#
#

import FWCore.ParameterSet.Config as cms
import sys


###################### user choices ######################
# choose (pre)release
latestRelease = 'CMSSW_3_5_X'
useRelease = 'CMSSW_3_4_1'

# choose the type of sample used:
#   True for RelVal
#   False for data

# default value
useRelValSample = True
#
# comment/uncomment the next line to choose sample type 
# (un-commented selects data)
useRelValSample=False 

if useRelValSample == True :
    
    globalTag = 'MC'
    #globalTag = 'STARTUP'
    
    # RelVals 
    useSample = 'RelValQCD_Pt_80_120'
    #useSample = 'RelValTTbar'
    #useSample = 'RelValZTT'
 
    # data type: for RAW data, the unpacker sequence RawToDigi will be also run
    #dataType = 'RAW'
    dataType = 'RECO'
   
else :

    # data type
    dataType = 'RAW'
    #dataType = 'StreamFile'
    #dataType = 'RECO'

# change to True to use local files
#     the type of file should match the choice of useRelValSample and useGlobalTag
useLocalFiles = False 

# number of events to be run (-1 for all)
maxNumberEvents = 10
#maxNumberEvents = -1

###################### end user choices ###################

# global tags 
if useRelValSample == True :
    
    if useRelease == 'CMSSW_3_5_X' :
        if globalTag == 'MC' :
            useGlobalTag = 'MC_3XY_V15'
        else :
            useGlobalTag = 'STARTUP3X_V15'
    elif useRelease == 'CMSSW_3_4_1' :
        if globalTag == 'MC' :
            useGlobalTag = 'MC_3XY_V15'
        else :
            useGlobalTag = 'STARTUP3X_V15'
    elif useRelease == 'CMSSW_3_3_6' :
        if globalTag == 'MC' :
            useGlobalTag = 'MC_3XY_V9B'
        else :
            useGlobalTag = 'STARTUP3X_V8M'
    else :
        print 'Error: no global tag defined for release ', useRelease, ' used with RelVal sample'
        sys.exit()
   
else :
    # global tag
    
    if useRelease == 'CMSSW_3_5_X' :
        useGlobalTag = 'GR09_P_V8_34X'
    elif useRelease == 'CMSSW_3_4_1' :
        useGlobalTag = 'GR09_P_V8_34X'
    elif useRelease == 'CMSSW_3_3_6' :
        useGlobalTag = 'GR09_P_V8'
    else :
        print 'Error: no global tag defined for release ', useRelease, ' used with data sample'
        sys.exit()

#
#

process = cms.Process("TestGtAnalyzer")

#
# load and configure modules via Global Tag
# https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideFrontierConditions

process.load('Configuration.StandardSequences.Geometry_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.GlobalTag.globaltag = useGlobalTag+'::All'


# number of events to be processed and source file
process.maxEvents = cms.untracked.PSet(
    input=cms.untracked.int32(maxNumberEvents)
)

# for RAW data, run first the RAWTODIGI 
if dataType == 'StreamFile' :
    readFiles = cms.untracked.vstring()
    process.source = cms.Source("NewEventStreamFileReader", fileNames=readFiles)
else :        
    readFiles = cms.untracked.vstring()
    secFiles = cms.untracked.vstring() 
    process.source = cms.Source ('PoolSource', fileNames=readFiles, secondaryFileNames=secFiles)

# type of sample used (True for RelVal, False for data)

if useRelValSample == True :
    if useGlobalTag.count('IDEAL') or useGlobalTag.count('MC') :
        
        if (useRelease == 'CMSSW_3_5_X') and (useSample == 'RelValQCD_Pt_80_120') and (dataType == 'RECO') :

            print '   Running ', useRelease, ' and global tag ', useGlobalTag, ' on ', useSample, ' sample'  

            dataset = cms.untracked.vstring('/RelValQCD_Pt_80_120/CMSSW_3_2_2-MC_31X_V3-v1/GEN-SIM-RECO')
            readFiles.extend( [
                '/store/relval/CMSSW_3_2_2/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V3-v1/0001/C022431D-4178-DE11-8B2E-001731AF6A89.root',
                '/store/relval/CMSSW_3_2_2/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V3-v1/0001/68B8CFCD-7578-DE11-B953-001731AF698F.root',
                '/store/relval/CMSSW_3_2_2/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V3-v1/0001/10B7B30E-4178-DE11-BB59-00304867929E.root',
                '/store/relval/CMSSW_3_2_2/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V3-v1/0001/10B567B3-3F78-DE11-95DA-00304866C398.root' 
                ] );

        elif (useRelease == 'CMSSW_3_5_X') and (useSample == 'RelValTTbar') and (dataType == 'RECO') :
            
            print '   Running ', useRelease, ' and global tag ', useGlobalTag, ' on ', useSample, ' sample'  

            dataset = cms.untracked.vstring('/RelValTTbar/CMSSW_3_2_2-MC_31X_V3-v1/GEN-SIM-RECO')
            readFiles.extend( [
                '/store/relval/CMSSW_3_2_2/RelValTTbar/GEN-SIM-RECO/MC_31X_V3-v1/0001/DC42776C-4078-DE11-9C5C-0018F3D0961E.root',
                '/store/relval/CMSSW_3_2_2/RelValTTbar/GEN-SIM-RECO/MC_31X_V3-v1/0001/7C9AB0C6-4178-DE11-A33C-003048D3FC94.root',
                '/store/relval/CMSSW_3_2_2/RelValTTbar/GEN-SIM-RECO/MC_31X_V3-v1/0001/5AFDDA6B-4078-DE11-9E9C-0018F3D096A2.root',
                '/store/relval/CMSSW_3_2_2/RelValTTbar/GEN-SIM-RECO/MC_31X_V3-v1/0001/4AE1FF1C-4178-DE11-8D06-001A92971BA0.root',
                '/store/relval/CMSSW_3_2_2/RelValTTbar/GEN-SIM-RECO/MC_31X_V3-v1/0001/08950864-7578-DE11-8AB7-001731A28F19.root'
                ] );

        elif (useRelease == 'CMSSW_3_4_1') and (useSample == 'RelValQCD_Pt_80_120') and (dataType == 'RECO') :

            print '   Running ', useRelease, ' and global tag ', useGlobalTag, ' on ', useSample, ' sample'  

            dataset = cms.untracked.vstring('/RelValQCD_Pt_80_120/CMSSW_3_2_2-MC_31X_V3-v1/GEN-SIM-RECO')
            readFiles.extend( [
                '/store/relval/CMSSW_3_2_2/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V3-v1/0001/C022431D-4178-DE11-8B2E-001731AF6A89.root',
                '/store/relval/CMSSW_3_2_2/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V3-v1/0001/68B8CFCD-7578-DE11-B953-001731AF698F.root',
                '/store/relval/CMSSW_3_2_2/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V3-v1/0001/10B7B30E-4178-DE11-BB59-00304867929E.root',
                '/store/relval/CMSSW_3_2_2/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_31X_V3-v1/0001/10B567B3-3F78-DE11-95DA-00304866C398.root' 
                ] );

        elif (useRelease == 'CMSSW_3_4_1') and (useSample == 'RelValTTbar') and (dataType == 'RECO') :
            
            print '   Running ', useRelease, ' and global tag ', useGlobalTag, ' on ', useSample, ' sample'  

            dataset = cms.untracked.vstring('/RelValTTbar/CMSSW_3_2_2-MC_31X_V3-v1/GEN-SIM-RECO')
            readFiles.extend( [
                '/store/relval/CMSSW_3_2_2/RelValTTbar/GEN-SIM-RECO/MC_31X_V3-v1/0001/DC42776C-4078-DE11-9C5C-0018F3D0961E.root',
                '/store/relval/CMSSW_3_2_2/RelValTTbar/GEN-SIM-RECO/MC_31X_V3-v1/0001/7C9AB0C6-4178-DE11-A33C-003048D3FC94.root',
                '/store/relval/CMSSW_3_2_2/RelValTTbar/GEN-SIM-RECO/MC_31X_V3-v1/0001/5AFDDA6B-4078-DE11-9E9C-0018F3D096A2.root',
                '/store/relval/CMSSW_3_2_2/RelValTTbar/GEN-SIM-RECO/MC_31X_V3-v1/0001/4AE1FF1C-4178-DE11-8D06-001A92971BA0.root',
                '/store/relval/CMSSW_3_2_2/RelValTTbar/GEN-SIM-RECO/MC_31X_V3-v1/0001/08950864-7578-DE11-8AB7-001731A28F19.root'
                ] );


        else :
            print 'Error: no files for sample ', useSample, ', (pre)release ', useRelease, ' and global tag ', useGlobalTag, ' defined.'    
            sys.exit()

        secFiles.extend([
            ])

    elif useGlobalTag.count('STARTUP') :

        if (useRelease == 'CMSSW_3_5_X') and (useSample == 'RelValQCD_Pt_80_120') and (dataType == 'RECO') :
        
            print '   Running ', useRelease, ' and global tag ', useGlobalTag, ' on ', useSample, ' sample'  

            dataset = cms.untracked.vstring('/RelValQCD_Pt_80_120/CMSSW_3_2_2-MC_31X_V3-v1/GEN-SIM-RECO')
            readFiles.extend( [
                '/store/relval/CMSSW_3_2_2/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP31X_V2-v1/0001/A035735C-3E78-DE11-A21D-001731AF678D.root',
                '/store/relval/CMSSW_3_2_2/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP31X_V2-v1/0001/704BF4FE-3E78-DE11-A352-003048678B0A.root',
                '/store/relval/CMSSW_3_2_2/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP31X_V2-v1/0001/6C96FAE5-3C78-DE11-AACB-0018F3D096A2.root',
                '/store/relval/CMSSW_3_2_2/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP31X_V2-v1/0001/40911F0A-7578-DE11-9619-001A92971B88.root'
                ] );


        elif (useRelease == 'CMSSW_3_5_X') and (useSample == 'RelValTTbar') and (dataType == 'RECO') :
        
            print '   Running ', useRelease, ' and global tag ', useGlobalTag, ' on ', useSample, ' sample'  

            dataset = cms.untracked.vstring('/RelValTTbar/CMSSW_3_2_2-STARTUP31X_V2-v1/GEN-SIM-RECO')
            readFiles.extend( [
                '/store/relval/CMSSW_3_2_2/RelValTTbar/GEN-SIM-RECO/STARTUP31X_V2-v1/0001/D2ED5A8E-4478-DE11-BA8A-0018F3D096A2.root',
                '/store/relval/CMSSW_3_2_2/RelValTTbar/GEN-SIM-RECO/STARTUP31X_V2-v1/0001/9ABEFAC9-7578-DE11-AA25-0018F3D095EC.root',
                '/store/relval/CMSSW_3_2_2/RelValTTbar/GEN-SIM-RECO/STARTUP31X_V2-v1/0001/70B9689B-3D78-DE11-B90F-001731AF6A89.root',
                '/store/relval/CMSSW_3_2_2/RelValTTbar/GEN-SIM-RECO/STARTUP31X_V2-v1/0001/5C711444-4578-DE11-84B5-001731A28FC9.root',
                '/store/relval/CMSSW_3_2_2/RelValTTbar/GEN-SIM-RECO/STARTUP31X_V2-v1/0001/0A5F22EE-3C78-DE11-8EE1-0018F3D096FE.root'            
                ] );

        elif (useRelease == 'CMSSW_3_4_1') and (useSample == 'RelValQCD_Pt_80_120') and (dataType == 'RECO') :
        
            print '   Running ', useRelease, ' and global tag ', useGlobalTag, ' on ', useSample, ' sample'  

            dataset = cms.untracked.vstring('/RelValQCD_Pt_80_120/CMSSW_3_2_2-MC_31X_V3-v1/GEN-SIM-RECO')
            readFiles.extend( [
                '/store/relval/CMSSW_3_2_2/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP31X_V2-v1/0001/A035735C-3E78-DE11-A21D-001731AF678D.root',
                '/store/relval/CMSSW_3_2_2/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP31X_V2-v1/0001/704BF4FE-3E78-DE11-A352-003048678B0A.root',
                '/store/relval/CMSSW_3_2_2/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP31X_V2-v1/0001/6C96FAE5-3C78-DE11-AACB-0018F3D096A2.root',
                '/store/relval/CMSSW_3_2_2/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP31X_V2-v1/0001/40911F0A-7578-DE11-9619-001A92971B88.root'
                ] );


        elif (useRelease == 'CMSSW_3_4_1') and (useSample == 'RelValTTbar') and (dataType == 'RECO') :
        
            print '   Running ', useRelease, ' and global tag ', useGlobalTag, ' on ', useSample, ' sample'  

            dataset = cms.untracked.vstring('/RelValTTbar/CMSSW_3_2_2-STARTUP31X_V2-v1/GEN-SIM-RECO')
            readFiles.extend( [
                '/store/relval/CMSSW_3_2_2/RelValTTbar/GEN-SIM-RECO/STARTUP31X_V2-v1/0001/D2ED5A8E-4478-DE11-BA8A-0018F3D096A2.root',
                '/store/relval/CMSSW_3_2_2/RelValTTbar/GEN-SIM-RECO/STARTUP31X_V2-v1/0001/9ABEFAC9-7578-DE11-AA25-0018F3D095EC.root',
                '/store/relval/CMSSW_3_2_2/RelValTTbar/GEN-SIM-RECO/STARTUP31X_V2-v1/0001/70B9689B-3D78-DE11-B90F-001731AF6A89.root',
                '/store/relval/CMSSW_3_2_2/RelValTTbar/GEN-SIM-RECO/STARTUP31X_V2-v1/0001/5C711444-4578-DE11-84B5-001731A28FC9.root',
                '/store/relval/CMSSW_3_2_2/RelValTTbar/GEN-SIM-RECO/STARTUP31X_V2-v1/0001/0A5F22EE-3C78-DE11-8EE1-0018F3D096FE.root'            
                ] );


        else :
            print 'Error: no files for sample ', useSample, ', (pre)release ', useRelease, ' and global tag ', useGlobalTag, ' defined.'    
            sys.exit()

                
        secFiles.extend([
            ])
        
    else :
        print 'Error: Global Tag ', useGlobalTag, ' not defined.'    
        sys.exit()

else :
    if dataType == 'RAW' : 

        # data POOL
        dataset = cms.untracked.vstring('/Cosmics/Commissioning09-v3/RAW')
        readFiles.extend( [
                '/store/data/BeamCommissioning09/Cosmics/RAW/v1/000/121/560/DC089E4B-5ED4-DE11-A179-000423D98FBC.root'
        
            ] );

        secFiles.extend([
            ])
    
    elif dataType == 'FileStream' : 
        # data dat
        readFiles.extend( [
                'file:/lookarea_SM/MWGR_29.00105760.0001.A.storageManager.00.0000.dat'
        
            ] );


if useLocalFiles :
    readFiles = 'file:/afs/cern.ch/user/g/ghete/scratch0/CmsswTestFiles/l1GtAnalyzer_source.root'

# processes to be run


process.load("L1Trigger.GlobalTriggerAnalyzer.l1GtAnalyzer_cfi")
#
# input tag for GT readout collection: 
#process.l1GtAnalyzer.L1GtDaqInputTag = 'gtDigis' 
 
# input tags for GT lite record
#process.l1GtAnalyzer.L1GtRecordInputTag = 'l1GtRecord'

# input tag for GT object map collection
#process.l1GtAnalyzer.L1GtObjectMapTag = 'hltL1GtObjectMap'

# physics algorithm name or alias, technical trigger name 
process.l1GtAnalyzer.AlgorithmName = 'L1_SingleEG20'

# condition in the above algorithm to test the object maps
process.l1GtAnalyzer.ConditionName = 'SingleNoIsoEG_0x14'




process.load("L1Trigger.GlobalTriggerAnalyzer.l1GtTrigReport_cfi")

# boolean flag to select the input record
#process.l1GtTrigReport.UseL1GlobalTriggerRecord = True

# input tag for the GT record requested: 
#   GT emulator:    gtDigis (DAQ record)
#   GT unpacker:    gtDigis (DAQ record)
#   GT lite record: l1GtRecord 
#process.l1GtTrigReport.L1GtRecordInputTag = "gtDigis"

#process.l1GtTrigReport.PrintVerbosity = 2

# print output: 0 = std::cout; 1 = LogTrace; 2 = LogVerbatim; 3 = LogInfo
#process.l1GtTrigReport.PrintOutput = 1


# for RAW data, run first the RAWTODIGI 
if dataType == 'RAW' :
    process.load('Configuration/StandardSequences/RawToDigi_Data_cff')
    process.l1GtTrigReport.L1GtRecordInputTag = "gtDigis"
    process.p = cms.Path(process.RawToDigi+process.l1GtTrigReport+process.l1GtAnalyzer)
    
else :        
    # path to be run for RECO
    process.p = cms.Path(process.l1GtTrigReport+process.l1GtAnalyzer)


# Message Logger
process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.debugModules = ['l1GtAnalyzer']
process.MessageLogger.categories.append('L1GtAnalyzer')
process.MessageLogger.categories.append('L1GtUtils')

process.MessageLogger.cerr.default.limit = 0
process.MessageLogger.cerr.FwkJob.limit = 0
process.MessageLogger.cerr.FwkReport.limit = 0
process.MessageLogger.cerr.FwkSummary.limit = 0

process.MessageLogger.debugs = cms.untracked.PSet( 
        threshold = cms.untracked.string('DEBUG'),
        DEBUG = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        INFO = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        WARNING = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        ERROR = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        L1GtAnalyzer = cms.untracked.PSet( limit = cms.untracked.int32(-1) ), 
        L1GtUtils = cms.untracked.PSet( limit = cms.untracked.int32(0) ) 
        )

process.MessageLogger.warnings = cms.untracked.PSet( 
        threshold = cms.untracked.string('WARNING'),
        WARNING = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        ERROR = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        L1GtAnalyzer = cms.untracked.PSet( limit = cms.untracked.int32(-1) ), 
        L1GtUtils = cms.untracked.PSet( limit = cms.untracked.int32(0) ) 
        )

process.MessageLogger.errors = cms.untracked.PSet( 
        threshold = cms.untracked.string('ERROR'),
        ERROR = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
        L1GtAnalyzer = cms.untracked.PSet( limit = cms.untracked.int32(-1) ), 
        L1GtUtils = cms.untracked.PSet( limit = cms.untracked.int32(0) ) 
        )