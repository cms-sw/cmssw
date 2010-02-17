#
# cfg file to run the L1 GT test analyzer according to 
#   the options set in "user choices"
#
# 2010-02-17 Updated to 3_5_0 (RelVals and GTags)
#

import FWCore.ParameterSet.Config as cms
import sys


###################### user choices ######################
# choose (pre)release
latestRelease = 'CMSSW_3_5_X'
useRelease = 'CMSSW_3_5_X'

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
    #globalTag = 'START'
    
    # RelVals 
    useSample = 'RelValQCD_Pt_80_120'
    #useSample = 'RelValTTbar'
 
    # data type
    dataType = 'RECO'
   
else :

    # data type: StreamFile is not associated to these runs (no available files)
    #           for RAW data, the unpacker sequence RawToDigi will be also run
    #dataType = 'RAW'
    #dataType = 'StreamFile'
    dataType = 'RECO'
    
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

# global tags 
if (useRelValSample == True) and (useLocalFiles == False) :
    
    if useRelease == 'CMSSW_3_5_X' :
        if globalTag == 'MC' :
            useGlobalTag = 'MC_3XY_V21'
        else :
            useGlobalTag = 'START3X_V21'
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
   
elif (useRelValSample == False) and (useLocalFiles == False) :
    # global tag
    
    if useRelease == 'CMSSW_3_5_X' :
        useGlobalTag = 'GR10_P_V2'
    elif useRelease == 'CMSSW_3_4_1' :
        useGlobalTag = 'GR09_P_V8_34X'
    elif useRelease == 'CMSSW_3_3_6' :
        useGlobalTag = 'GR09_P_V8'
    else :
        print 'Error: no global tag defined for release ', useRelease, ' used with data sample'
        sys.exit()
else :
       print 'Using local file(s) with global tag ',  useGlobalTag, ' and release ', useRelease
     
#
#

process = cms.Process("L1GtTriggerMenuLite")

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

if (useRelValSample == True) and (useLocalFiles == False) :
    if useGlobalTag.count('IDEAL') or useGlobalTag.count('MC') :
        
        if (useRelease == 'CMSSW_3_5_X') and (useSample == 'RelValQCD_Pt_80_120') and (dataType == 'RECO') :

            print '   Running ', useRelease, ' and global tag ', useGlobalTag, ' on ', useSample, ' sample'  

            dataset = cms.untracked.vstring('/RelValQCD_Pt_80_120/CMSSW_3_5_0-MC_3XY_V21-v1/GEN-SIM-RECO')
            readFiles.extend( [
                '/store/relval/CMSSW_3_5_0/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_3XY_V21-v1/0013/A6A96D0E-6313-DF11-8B6F-001A928116EE.root',
                '/store/relval/CMSSW_3_5_0/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_3XY_V21-v1/0012/FEDBEF25-3513-DF11-BD91-003048678A6A.root',
                '/store/relval/CMSSW_3_5_0/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_3XY_V21-v1/0012/FA06ACF3-3313-DF11-9FBE-002618943947.root',
                '/store/relval/CMSSW_3_5_0/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_3XY_V21-v1/0012/A4BA5423-3513-DF11-95AA-003048D15E02.root',
                '/store/relval/CMSSW_3_5_0/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_3XY_V21-v1/0012/7870118C-3513-DF11-8B09-002354EF3BDE.root',
                '/store/relval/CMSSW_3_5_0/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_3XY_V21-v1/0012/7853EB0B-3513-DF11-9454-00304867C16A.root',
                '/store/relval/CMSSW_3_5_0/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_3XY_V21-v1/0012/289824F3-3313-DF11-B31A-0026189438B1.root',
                '/store/relval/CMSSW_3_5_0/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_3XY_V21-v1/0012/16F61E92-3513-DF11-97DB-00248C0BE013.root' 
                ] );


        elif (useRelease == 'CMSSW_3_5_X') and (useSample == 'RelValTTbar') and (dataType == 'RECO') :
            
            print '   Running ', useRelease, ' and global tag ', useGlobalTag, ' on ', useSample, ' sample'  

            dataset = cms.untracked.vstring('/RelValTTbar/CMSSW_3_5_0-MC_3XY_V21-v1/GEN-SIM-RECO')
            readFiles.extend( [
                '/store/relval/CMSSW_3_5_0/RelValTTbar/GEN-SIM-RECO/MC_3XY_V21-v1/0014/5837B301-6D13-DF11-970D-001A928116EE.root',
                '/store/relval/CMSSW_3_5_0/RelValTTbar/GEN-SIM-RECO/MC_3XY_V21-v1/0013/F203E839-6213-DF11-A013-0018F3D0960A.root',
                '/store/relval/CMSSW_3_5_0/RelValTTbar/GEN-SIM-RECO/MC_3XY_V21-v1/0013/C0FA4F0F-3D13-DF11-9C08-002618943914.root',
                '/store/relval/CMSSW_3_5_0/RelValTTbar/GEN-SIM-RECO/MC_3XY_V21-v1/0013/B2BC513E-4913-DF11-B3E8-001731AF68BF.root',
                '/store/relval/CMSSW_3_5_0/RelValTTbar/GEN-SIM-RECO/MC_3XY_V21-v1/0013/B01AE947-4713-DF11-861B-0018F3D095EA.root',
                '/store/relval/CMSSW_3_5_0/RelValTTbar/GEN-SIM-RECO/MC_3XY_V21-v1/0013/920E11A7-4413-DF11-B796-001731AF6A91.root',
                '/store/relval/CMSSW_3_5_0/RelValTTbar/GEN-SIM-RECO/MC_3XY_V21-v1/0013/7C15A7E1-6913-DF11-B0A0-00261894393E.root',
                '/store/relval/CMSSW_3_5_0/RelValTTbar/GEN-SIM-RECO/MC_3XY_V21-v1/0013/74EB4EA3-4613-DF11-87E0-002618943854.root',
                '/store/relval/CMSSW_3_5_0/RelValTTbar/GEN-SIM-RECO/MC_3XY_V21-v1/0013/36A56A12-4513-DF11-B04B-002618943944.root',
                '/store/relval/CMSSW_3_5_0/RelValTTbar/GEN-SIM-RECO/MC_3XY_V21-v1/0013/30D1DBC5-4A13-DF11-813E-003048679296.root',
                '/store/relval/CMSSW_3_5_0/RelValTTbar/GEN-SIM-RECO/MC_3XY_V21-v1/0013/304C7A3B-4813-DF11-9E3E-001A92971AEC.root' 
                ] );

        elif (useRelease == 'CMSSW_3_4_1') and (useSample == 'RelValQCD_Pt_80_120') and (dataType == 'RECO') :

            print '   Running ', useRelease, ' and global tag ', useGlobalTag, ' on ', useSample, ' sample'  

            dataset = cms.untracked.vstring('/RelValQCD_Pt_80_120/CMSSW_3_4_1-MC_3XY_V14-v1/GEN-SIM-RECO')

            readFiles.extend( [
                '/store/relval/CMSSW_3_4_1/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_3XY_V14-v1/0004/FE21D38C-7AED-DE11-8572-001D09F23F2A.root',
                '/store/relval/CMSSW_3_4_1/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_3XY_V14-v1/0004/EA8D76BF-B5ED-DE11-B84A-000423D9853C.root',
                '/store/relval/CMSSW_3_4_1/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_3XY_V14-v1/0004/D08D7EE2-7CED-DE11-B2CA-001D09F254CE.root',
                '/store/relval/CMSSW_3_4_1/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_3XY_V14-v1/0004/90EF4AA4-7CED-DE11-B81E-001D09F2AF1E.root',
                '/store/relval/CMSSW_3_4_1/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_3XY_V14-v1/0004/907B2B6B-7BED-DE11-83E5-0019B9F7312C.root',
                '/store/relval/CMSSW_3_4_1/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_3XY_V14-v1/0004/8C5DBBA2-7BED-DE11-AD49-001D09F290BF.root',
                '/store/relval/CMSSW_3_4_1/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_3XY_V14-v1/0004/6E9C78EE-79ED-DE11-BFD0-000423D98BC4.root',
                '/store/relval/CMSSW_3_4_1/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_3XY_V14-v1/0004/5E79FB96-7CED-DE11-9300-0019B9F7312C.root' 
                ] );

        elif (useRelease == 'CMSSW_3_4_1') and (useSample == 'RelValTTbar') and (dataType == 'RECO') :
            
            print '   Running ', useRelease, ' and global tag ', useGlobalTag, ' on ', useSample, ' sample'  

            dataset = cms.untracked.vstring('/RelValTTbar/CMSSW_3_4_1-MC_3XY_V14-v1/GEN-SIM-RECO')
            readFiles.extend( [
                '/store/relval/CMSSW_3_4_1/RelValTTbar/GEN-SIM-RECO/MC_3XY_V14-v1/0004/F4996794-A2ED-DE11-81E5-001D09F2423B.root',
                '/store/relval/CMSSW_3_4_1/RelValTTbar/GEN-SIM-RECO/MC_3XY_V14-v1/0004/BCF09642-A1ED-DE11-A424-001D09F231B0.root',
                '/store/relval/CMSSW_3_4_1/RelValTTbar/GEN-SIM-RECO/MC_3XY_V14-v1/0004/AA0AFA35-A2ED-DE11-A322-000423D98868.root',
                '/store/relval/CMSSW_3_4_1/RelValTTbar/GEN-SIM-RECO/MC_3XY_V14-v1/0004/9C5ACF97-A1ED-DE11-9AA9-0030487A1990.root',
                '/store/relval/CMSSW_3_4_1/RelValTTbar/GEN-SIM-RECO/MC_3XY_V14-v1/0004/88C36CAE-B6ED-DE11-A0C9-000423D985E4.root',
                '/store/relval/CMSSW_3_4_1/RelValTTbar/GEN-SIM-RECO/MC_3XY_V14-v1/0004/6E40CDC2-A2ED-DE11-855A-001D09F29146.root',
                '/store/relval/CMSSW_3_4_1/RelValTTbar/GEN-SIM-RECO/MC_3XY_V14-v1/0004/68F7F3F5-A1ED-DE11-A407-0030487A3C9A.root',
                '/store/relval/CMSSW_3_4_1/RelValTTbar/GEN-SIM-RECO/MC_3XY_V14-v1/0004/5AF3C074-9FED-DE11-8F1C-001D09F241F0.root',
                '/store/relval/CMSSW_3_4_1/RelValTTbar/GEN-SIM-RECO/MC_3XY_V14-v1/0004/40545697-A0ED-DE11-A389-001D09F24FEC.root' 
                ] );


        else :
            print 'Error: no files for sample ', useSample, ', (pre)release ', useRelease, ' and global tag ', useGlobalTag, ' defined.'    
            sys.exit()

        secFiles.extend([
            ])

    elif useGlobalTag.count('START') :

        if (useRelease == 'CMSSW_3_5_X') and (useSample == 'RelValQCD_Pt_80_120') and (dataType == 'RECO') :
        
            print '   Running ', useRelease, ' and global tag ', useGlobalTag, ' on ', useSample, ' sample'  

            dataset = cms.untracked.vstring('/RelValQCD_Pt_80_120/CMSSW_3_5_0-START3X_V21-v1/GEN-SIM-RECO')
            readFiles.extend( [
                '/store/relval/CMSSW_3_5_0/RelValQCD_Pt_80_120/GEN-SIM-RECO/START3X_V21-v1/0013/B27E46BF-3E13-DF11-A7EE-001A9281172C.root',
                '/store/relval/CMSSW_3_5_0/RelValQCD_Pt_80_120/GEN-SIM-RECO/START3X_V21-v1/0013/44E856CB-3F13-DF11-8B01-001A92971B7C.root',
                '/store/relval/CMSSW_3_5_0/RelValQCD_Pt_80_120/GEN-SIM-RECO/START3X_V21-v1/0013/3ACDFD75-4013-DF11-A204-001A92971BDC.root',
                '/store/relval/CMSSW_3_5_0/RelValQCD_Pt_80_120/GEN-SIM-RECO/START3X_V21-v1/0013/3A52592E-3E13-DF11-8EAE-001A92810AEA.root',
                '/store/relval/CMSSW_3_5_0/RelValQCD_Pt_80_120/GEN-SIM-RECO/START3X_V21-v1/0013/38DA44B8-3D13-DF11-B10D-0018F3D09642.root',
                '/store/relval/CMSSW_3_5_0/RelValQCD_Pt_80_120/GEN-SIM-RECO/START3X_V21-v1/0013/14004A7D-6213-DF11-B250-001A92810AEE.root',
                '/store/relval/CMSSW_3_5_0/RelValQCD_Pt_80_120/GEN-SIM-RECO/START3X_V21-v1/0012/EA2B34EC-3913-DF11-8E34-0026189438BC.root',
                '/store/relval/CMSSW_3_5_0/RelValQCD_Pt_80_120/GEN-SIM-RECO/START3X_V21-v1/0012/DA4420C6-3813-DF11-84CC-003048679000.root' 
                ] );


        elif (useRelease == 'CMSSW_3_5_X') and (useSample == 'RelValTTbar') and (dataType == 'RECO') :
        
            print '   Running ', useRelease, ' and global tag ', useGlobalTag, ' on ', useSample, ' sample'  

            dataset = cms.untracked.vstring('/RelValTTbar/CMSSW_3_5_0-START3X_V21-v1/GEN-SIM-RECO')
            readFiles.extend( [
                '/store/relval/CMSSW_3_5_0/RelValTTbar/GEN-SIM-RECO/START3X_V21-v1/0013/E27A62CF-4013-DF11-8D9C-001A92971B32.root',
                '/store/relval/CMSSW_3_5_0/RelValTTbar/GEN-SIM-RECO/START3X_V21-v1/0013/C6BED359-4013-DF11-B77E-002618943800.root',
                '/store/relval/CMSSW_3_5_0/RelValTTbar/GEN-SIM-RECO/START3X_V21-v1/0013/B250B859-4213-DF11-9CE7-001A92971B12.root',
                '/store/relval/CMSSW_3_5_0/RelValTTbar/GEN-SIM-RECO/START3X_V21-v1/0013/984BA759-4013-DF11-8E61-002618FDA207.root',
                '/store/relval/CMSSW_3_5_0/RelValTTbar/GEN-SIM-RECO/START3X_V21-v1/0013/6C7C2ADA-4113-DF11-AFB7-001A92971B64.root',
                '/store/relval/CMSSW_3_5_0/RelValTTbar/GEN-SIM-RECO/START3X_V21-v1/0013/606521E1-4313-DF11-845B-001731AF669D.root',
                '/store/relval/CMSSW_3_5_0/RelValTTbar/GEN-SIM-RECO/START3X_V21-v1/0013/6038B443-3F13-DF11-A075-001A92811700.root',
                '/store/relval/CMSSW_3_5_0/RelValTTbar/GEN-SIM-RECO/START3X_V21-v1/0013/4E6B8895-3E13-DF11-B951-003048678BB8.root',
                '/store/relval/CMSSW_3_5_0/RelValTTbar/GEN-SIM-RECO/START3X_V21-v1/0012/AE86CDB7-3813-DF11-AB79-0030486790BA.root' 
                ] );

        elif (useRelease == 'CMSSW_3_4_1') and (useSample == 'RelValQCD_Pt_80_120') and (dataType == 'RECO') :
        
            print '   Running ', useRelease, ' and global tag ', useGlobalTag, ' on ', useSample, ' sample'  

            dataset = cms.untracked.vstring('/RelValQCD_Pt_80_120/CMSSW_3_4_1-STARTUP3X_V14-v1/GEN-SIM-RECO')
            readFiles.extend( [
                '/store/relval/CMSSW_3_4_1/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP3X_V14-v1/0004/D8AE64A2-8FED-DE11-B37B-000423D99658.root',
                '/store/relval/CMSSW_3_4_1/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP3X_V14-v1/0004/D814474A-90ED-DE11-A54F-001D09F2A49C.root',
                '/store/relval/CMSSW_3_4_1/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP3X_V14-v1/0004/D2235DA4-8FED-DE11-A5FE-001D09F241F0.root',
                '/store/relval/CMSSW_3_4_1/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP3X_V14-v1/0004/C2371774-B6ED-DE11-9292-0030487C5CFA.root',
                '/store/relval/CMSSW_3_4_1/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP3X_V14-v1/0004/B84A6EE6-8EED-DE11-9733-001D09F29321.root',
                '/store/relval/CMSSW_3_4_1/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP3X_V14-v1/0004/58E9DA6E-8DED-DE11-91C8-003048D3756A.root',
                '/store/relval/CMSSW_3_4_1/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP3X_V14-v1/0004/48D81FB5-91ED-DE11-9BFA-001617C3B5E4.root',
                '/store/relval/CMSSW_3_4_1/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP3X_V14-v1/0004/0AB1FC24-91ED-DE11-8A56-001D09F231B0.root' 
                ] );


        elif (useRelease == 'CMSSW_3_4_1') and (useSample == 'RelValTTbar') and (dataType == 'RECO') :
        
            print '   Running ', useRelease, ' and global tag ', useGlobalTag, ' on ', useSample, ' sample'  

            dataset = cms.untracked.vstring('/RelValTTbar/CMSSW_3_4_1-STARTUP3X_V14-v1/GEN-SIM-RECO')
            readFiles.extend( [
                '/store/relval/CMSSW_3_4_1/RelValTTbar/GEN-SIM-RECO/STARTUP3X_V14-v1/0004/CE62D4D8-85ED-DE11-8BD2-000423D9853C.root',
                '/store/relval/CMSSW_3_4_1/RelValTTbar/GEN-SIM-RECO/STARTUP3X_V14-v1/0004/A0FB9B2E-85ED-DE11-8A8D-001D09F290CE.root',
                '/store/relval/CMSSW_3_4_1/RelValTTbar/GEN-SIM-RECO/STARTUP3X_V14-v1/0004/9A2F0DDF-85ED-DE11-B5D1-001D09F290CE.root',
                '/store/relval/CMSSW_3_4_1/RelValTTbar/GEN-SIM-RECO/STARTUP3X_V14-v1/0004/820C7C8C-86ED-DE11-83D4-001D09F295FB.root',
                '/store/relval/CMSSW_3_4_1/RelValTTbar/GEN-SIM-RECO/STARTUP3X_V14-v1/0004/685C77F0-87ED-DE11-A4A5-000423D60FF6.root',
                '/store/relval/CMSSW_3_4_1/RelValTTbar/GEN-SIM-RECO/STARTUP3X_V14-v1/0004/4CFCC894-86ED-DE11-B3F4-001D09F2447F.root',
                '/store/relval/CMSSW_3_4_1/RelValTTbar/GEN-SIM-RECO/STARTUP3X_V14-v1/0004/3EA206BD-B5ED-DE11-B481-000423D6C8E6.root',
                '/store/relval/CMSSW_3_4_1/RelValTTbar/GEN-SIM-RECO/STARTUP3X_V14-v1/0004/3CCCE28D-86ED-DE11-A583-000423D986C4.root',
                '/store/relval/CMSSW_3_4_1/RelValTTbar/GEN-SIM-RECO/STARTUP3X_V14-v1/0004/2CF90F4D-87ED-DE11-A3AF-003048D375AA.root' 
                ] );


        else :
            print 'Error: no files for sample ', useSample, ', (pre)release ', useRelease, ' and global tag ', useGlobalTag, ' defined.'    
            sys.exit()

                
        secFiles.extend([
            ])
        
    else :
        print 'Error: Global Tag ', useGlobalTag, ' not defined.'    
        sys.exit()

elif (useRelValSample == False) and (useLocalFiles == False) :
    if dataType == 'RAW' : 

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

    
    elif dataType == 'RECO' : 
        # data dat
        if runNumber == 123596 :
            dataset = '/Cosmics/BeamCommissioning09-v2/RECO'
            readFiles.extend( [
                '/store/data/BeamCommissioning09/Cosmics/RECO/v2/000/123/596/FC5C3B0F-8AE2-DE11-A905-003048D37456.root'
        
                ] );
        elif (runNumber == 127715) and (useRelease == 'CMSSW_3_5_X') :
            dataset = '/Cosmics/Commissioning10-v3/RECO'
            print '   Running on set: '+ dataset + ' with global tag ' +  useGlobalTag 
    
            readFiles.extend( [                        
                '/store/data/Commissioning10/Cosmics/RECO/v3/000/127/715/261A3141-9F18-DF11-883E-001D09F24493.root'
                ]);                                                                                               

            secFiles.extend([
                ])

        else :
            print 'Error: run ', runNumber, ' has no RECO file available.'    
            sys.exit()

    elif dataType == 'FileStream' : 
        # data dat
        readFiles.extend( [
                'file:/lookarea_SM/MWGR_29.00105760.0001.A.storageManager.00.0000.dat'
        
            ] );

    else :
        print 'Error: dataType ',dataType, ' not defined.'    
        sys.exit()

else :
    readFiles.extend( [                        
        'file:/afs/cern.ch/user/g/ghete/scratch0/CmsswTestFiles/L1GtAnalyzer_source.root'
        ]);                                                                                               

    secFiles.extend([
        ])

    print 'Local file(s) ', readFiles


###################



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
#process.l1GtAnalyzer.AlgorithmName = 'L1_SingleEG5'
process.l1GtAnalyzer.AlgorithmName = 'L1_BscMinBiasOR_BptxPlusORMinus'
#process.l1GtAnalyzer.AlgorithmName = 'L1Tech_BPTX_plus_AND_minus_instance1.v0'
#process.l1GtAnalyzer.AlgorithmName = 'L1Tech_BPTX_quiet.v0'
#process.l1GtAnalyzer.AlgorithmName = 'L1Tech_BPTX_plus_AND_minus.v0'

# condition in the above algorithm to test the object maps
process.l1GtAnalyzer.ConditionName = 'SingleNoIsoEG_0x0A'

# a bit number
process.l1GtAnalyzer.BitNumber = 10




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
        L1GtUtils = cms.untracked.PSet( limit = cms.untracked.int32(-1) ) 
        )

process.MessageLogger.warnings = cms.untracked.PSet( 
        threshold = cms.untracked.string('WARNING'),
        WARNING = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        ERROR = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        L1GtAnalyzer = cms.untracked.PSet( limit = cms.untracked.int32(-1) ), 
        L1GtUtils = cms.untracked.PSet( limit = cms.untracked.int32(-1) ) 
        )

process.MessageLogger.errors = cms.untracked.PSet( 
        threshold = cms.untracked.string('ERROR'),
        ERROR = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
        L1GtAnalyzer = cms.untracked.PSet( limit = cms.untracked.int32(-1) ), 
        L1GtUtils = cms.untracked.PSet( limit = cms.untracked.int32(-1) ) 
        )