from __future__ import print_function
#
# cfg file to produce L1GlobalTriggerRecord from a L1GlobalTriggerReadoutRecord
# when running on RAW data, one also unpack the data
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
    
    runNumber = 123596
    #runNumber = 116035
    #runNumber = 121560


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
        print('Error: no global tag defined for release ', useRelease, ' used with RelVal sample')
        sys.exit()
   
elif (useRelValSample == False) and (useLocalFiles == False) :
    # global tag
    
    if useRelease == 'CMSSW_3_5_X' :
        useGlobalTag = 'GR09_P_V8_34X'
    elif useRelease == 'CMSSW_3_4_1' :
        useGlobalTag = 'GR09_P_V8_34X'
    elif useRelease == 'CMSSW_3_3_6' :
        useGlobalTag = 'GR09_P_V8'
    else :
        print('Error: no global tag defined for release ', useRelease, ' used with data sample')
        sys.exit()
else :
       print('Using local file(s) with global tag ',  useGlobalTag, ' and release ', useRelease)
     
#
#

process = cms.Process("TestL1GtRecord")

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

            print('   Running ', useRelease, ' and global tag ', useGlobalTag, ' on ', useSample, ' sample')  

            dataset = cms.untracked.vstring('/RelValQCD_Pt_80_120/CMSSW_3_5_0_pre3-MC_3XY_V15-v1/GEN-SIM-RECO')
            readFiles.extend( [
                '/store/relval/CMSSW_3_5_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_3XY_V15-v1/0006/1469E02C-D203-DF11-9C7F-00304879BAB2.root',
                '/store/relval/CMSSW_3_5_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_3XY_V15-v1/0005/E87FE5E9-4303-DF11-817C-0030487A3DE0.root',
                '/store/relval/CMSSW_3_5_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_3XY_V15-v1/0005/A063F4AC-4403-DF11-AB21-003048D2C092.root',
                '/store/relval/CMSSW_3_5_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_3XY_V15-v1/0005/9CFD43E8-4103-DF11-910C-0030487CD180.root',
                '/store/relval/CMSSW_3_5_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_3XY_V15-v1/0005/585E7702-4103-DF11-939A-0030487A18F2.root',
                '/store/relval/CMSSW_3_5_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_3XY_V15-v1/0005/3E675407-4303-DF11-9448-00304879BAB2.root',
                '/store/relval/CMSSW_3_5_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_3XY_V15-v1/0005/387D83E9-4403-DF11-8C17-00304879EDEA.root',
                '/store/relval/CMSSW_3_5_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-RECO/MC_3XY_V15-v1/0005/1E7F273C-4303-DF11-B884-0030487A3C92.root' 
                ] );


        elif (useRelease == 'CMSSW_3_5_X') and (useSample == 'RelValTTbar') and (dataType == 'RECO') :
            
            print('   Running ', useRelease, ' and global tag ', useGlobalTag, ' on ', useSample, ' sample')  

            dataset = cms.untracked.vstring('/RelValTTbar/CMSSW_3_5_0_pre3-MC_3XY_V15-v1/GEN-SIM-RECO')
            readFiles.extend( [
                '/store/relval/CMSSW_3_5_0_pre3/RelValTTbar/GEN-SIM-RECO/MC_3XY_V15-v1/0006/E80BF730-D203-DF11-8667-0030487CD6DA.root',
                '/store/relval/CMSSW_3_5_0_pre3/RelValTTbar/GEN-SIM-RECO/MC_3XY_V15-v1/0005/F8071FA0-0203-DF11-947E-0030487C90D4.root',
                '/store/relval/CMSSW_3_5_0_pre3/RelValTTbar/GEN-SIM-RECO/MC_3XY_V15-v1/0005/B4B3D656-0403-DF11-890D-0030487C60AE.root',
                '/store/relval/CMSSW_3_5_0_pre3/RelValTTbar/GEN-SIM-RECO/MC_3XY_V15-v1/0005/AEB20358-0503-DF11-85E1-0030487A18A4.root',
                '/store/relval/CMSSW_3_5_0_pre3/RelValTTbar/GEN-SIM-RECO/MC_3XY_V15-v1/0005/AE60F36C-0203-DF11-863B-0030487C608C.root',
                '/store/relval/CMSSW_3_5_0_pre3/RelValTTbar/GEN-SIM-RECO/MC_3XY_V15-v1/0005/82371379-0303-DF11-923E-0030487CD7C0.root',
                '/store/relval/CMSSW_3_5_0_pre3/RelValTTbar/GEN-SIM-RECO/MC_3XY_V15-v1/0005/78D9A28B-0F03-DF11-8D39-00304879FBB2.root',
                '/store/relval/CMSSW_3_5_0_pre3/RelValTTbar/GEN-SIM-RECO/MC_3XY_V15-v1/0005/787178D3-0903-DF11-8A43-0030487A1990.root',
                '/store/relval/CMSSW_3_5_0_pre3/RelValTTbar/GEN-SIM-RECO/MC_3XY_V15-v1/0005/0802E3D1-0703-DF11-88B5-0030487A3C9A.root' 
                ] );

        elif (useRelease == 'CMSSW_3_4_1') and (useSample == 'RelValQCD_Pt_80_120') and (dataType == 'RECO') :

            print('   Running ', useRelease, ' and global tag ', useGlobalTag, ' on ', useSample, ' sample')  

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
            
            print('   Running ', useRelease, ' and global tag ', useGlobalTag, ' on ', useSample, ' sample')  

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
            print('Error: no files for sample ', useSample, ', (pre)release ', useRelease, ' and global tag ', useGlobalTag, ' defined.')    
            sys.exit()

        secFiles.extend([
            ])

    elif useGlobalTag.count('STARTUP') :

        if (useRelease == 'CMSSW_3_5_X') and (useSample == 'RelValQCD_Pt_80_120') and (dataType == 'RECO') :
        
            print('   Running ', useRelease, ' and global tag ', useGlobalTag, ' on ', useSample, ' sample')  

            dataset = cms.untracked.vstring('/RelValQCD_Pt_80_120/CMSSW_3_2_2-MC_31X_V3-v1/GEN-SIM-RECO')
            readFiles.extend( [
                '/store/relval/CMSSW_3_5_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP3X_V15-v1/0005/B0026249-D103-DF11-A139-0030487A195C.root',
                '/store/relval/CMSSW_3_5_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP3X_V15-v1/0005/9CDC89D9-0803-DF11-9D10-0030487CD13A.root',
                '/store/relval/CMSSW_3_5_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP3X_V15-v1/0005/9C979251-0403-DF11-9C5F-0030487A17B8.root',
                '/store/relval/CMSSW_3_5_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP3X_V15-v1/0005/9022DC83-0403-DF11-B5BA-001D09F23A20.root',
                '/store/relval/CMSSW_3_5_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP3X_V15-v1/0005/8ACA874E-0503-DF11-B21A-0030487A1FEC.root',
                '/store/relval/CMSSW_3_5_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP3X_V15-v1/0005/3CB6012F-0303-DF11-8183-0030487C60AE.root',
                '/store/relval/CMSSW_3_5_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP3X_V15-v1/0005/1E6C658F-0B03-DF11-8913-0030487D1BCC.root',
                '/store/relval/CMSSW_3_5_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP3X_V15-v1/0005/1A01CF8D-0303-DF11-9F31-0030487CD77E.root' 
                ] );


        elif (useRelease == 'CMSSW_3_5_X') and (useSample == 'RelValTTbar') and (dataType == 'RECO') :
        
            print('   Running ', useRelease, ' and global tag ', useGlobalTag, ' on ', useSample, ' sample')  

            dataset = cms.untracked.vstring('/RelValTTbar/CMSSW_3_5_0_pre3-STARTUP3X_V15-v1/GEN-SIM-RECO')
            readFiles.extend( [
                '/store/relval/CMSSW_3_5_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP3X_V15-v1/0005/B0026249-D103-DF11-A139-0030487A195C.root',
                '/store/relval/CMSSW_3_5_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP3X_V15-v1/0005/9CDC89D9-0803-DF11-9D10-0030487CD13A.root',
                '/store/relval/CMSSW_3_5_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP3X_V15-v1/0005/9C979251-0403-DF11-9C5F-0030487A17B8.root',
                '/store/relval/CMSSW_3_5_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP3X_V15-v1/0005/9022DC83-0403-DF11-B5BA-001D09F23A20.root',
                '/store/relval/CMSSW_3_5_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP3X_V15-v1/0005/8ACA874E-0503-DF11-B21A-0030487A1FEC.root',
                '/store/relval/CMSSW_3_5_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP3X_V15-v1/0005/3CB6012F-0303-DF11-8183-0030487C60AE.root',
                '/store/relval/CMSSW_3_5_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP3X_V15-v1/0005/1E6C658F-0B03-DF11-8913-0030487D1BCC.root',
                '/store/relval/CMSSW_3_5_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP3X_V15-v1/0005/1A01CF8D-0303-DF11-9F31-0030487CD77E.root' 
                ] );


        elif (useRelease == 'CMSSW_3_4_1') and (useSample == 'RelValQCD_Pt_80_120') and (dataType == 'RECO') :
        
            print('   Running ', useRelease, ' and global tag ', useGlobalTag, ' on ', useSample, ' sample')  

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
        
            print('   Running ', useRelease, ' and global tag ', useGlobalTag, ' on ', useSample, ' sample')  

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
            print('Error: no files for sample ', useSample, ', (pre)release ', useRelease, ' and global tag ', useGlobalTag, ' defined.')    
            sys.exit()

                
        secFiles.extend([
            ])
        
    else :
        print('Error: Global Tag ', useGlobalTag, ' not defined.')    
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
            print('   Running on set: '+ dataset + ' with global tag ' +  useGlobalTag) 
    
            readFiles.extend( [                        
                '/store/data/Commissioning09/Cosmics/RAW/v3/000/116/035/34A8317D-76AF-DE11-91DB-000423D98DC4.root'
                ]);                                                                                               

            secFiles.extend([
                ])
        
        elif runNumber == 121560 :
            dataset = '/Cosmics/Commissioning09-v3/RAW'
            print('   Running on set: '+ dataset + ' with global tag ' +  useGlobalTag) 
    
            readFiles.extend( [                        
                '/store/data/BeamCommissioning09/Cosmics/RAW/v1/000/121/560/DC089E4B-5ED4-DE11-A179-000423D98FBC.root'
                ]);                                                                                               

            secFiles.extend([
                ])

        else :
            print('Error: run ', runNumber, ' not defined.')    
            sys.exit()

    
    elif dataType == 'FileStream' : 
        # data dat
        readFiles.extend( [
                'file:/lookarea_SM/MWGR_29.00105760.0001.A.storageManager.00.0000.dat'
        
            ] );

    else :
        print('Error: dataType ',dataType, ' not defined.')    
        sys.exit()

else :
    readFiles.extend( [                        
        'file:/afs/cern.ch/user/g/ghete/scratch0/CmsswTestFiles/L1GtRecord_source.root'
        ]);                                                                                               

    secFiles.extend([
        ])

    print('Local file(s) ', readFiles)



###################



# processes to be run

# L1 GT record producer
process.load("EventFilter.L1GlobalTriggerRawToDigi.l1GtRecord_cfi")

# input tag for GT readout collection: 
#     gtDigis       = GT emulator or GT unpacker (default)
#replace l1GtRecord.L1GtReadoutRecordTag = l1GtEmulDigis


# L1 GT trigger report for readout record
import L1Trigger.GlobalTriggerAnalyzer.l1GtTrigReport_cfi
process.l1GtTrigReadoutReport = L1Trigger.GlobalTriggerAnalyzer.l1GtTrigReport_cfi.l1GtTrigReport.clone()

# boolean flag to select the input record
#process.l1GtTrigReadoutReport.UseL1GlobalTriggerRecord = True

# input tag for the GT record requested: 
#   GT emulator:    gtDigis (DAQ record)
#   GT unpacker:    gtDigis (DAQ record)
#   GT lite record: l1GtRecord 
process.l1GtTrigReadoutReport.L1GtRecordInputTag = "gtDigis"

#process.l1GtTrigReadoutReport.PrintVerbosity = 2

# print output: 0 = std::cout; 1 = LogTrace; 2 = LogVerbatim; 3 = LogInfo
#process.l1GtTrigReadoutReport.PrintOutput = 1


# L1 GT trigger report for lite record
process.load("L1Trigger.GlobalTriggerAnalyzer.l1GtTrigReport_cfi")

# boolean flag to select the input record
process.l1GtTrigReport.UseL1GlobalTriggerRecord = True

# input tag for the GT record requested: 
#   GT emulator:    gtDigis (DAQ record)
#   GT unpacker:    gtDigis (DAQ record)
#   GT lite record: l1GtRecord 
process.l1GtTrigReport.L1GtRecordInputTag = "l1GtRecord"

#process.l1GtTrigReport.PrintVerbosity = 2

# print output: 0 = std::cout; 1 = LogTrace; 2 = LogVerbatim; 3 = LogInfo
#process.l1GtTrigReport.PrintOutput = 1

# for RAW data, run first the RAWTODIGI 
if dataType == 'RAW' :
    process.load('Configuration/StandardSequences/RawToDigi_Data_cff')
    process.p = cms.Path(process.RawToDigi+process.l1GtTrigReadoutReport+process.l1GtRecord+
                         process.l1GtTrigReport)
    
else :        
    # path to be run for RECO
    process.p = cms.Path(process.l1GtTrigReadoutReport+process.l1GtRecord+
                         process.l1GtTrigReport)


# Message Logger
process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.debugModules = ['l1GtRecord']
process.MessageLogger.categories.append('L1GlobalTriggerRecord')
process.MessageLogger.categories.append('L1GtTrigReport')

process.MessageLogger.cerr.default.limit = 0
process.MessageLogger.cerr.FwkReport.limit = 0
process.MessageLogger.cerr.FwkSummary.limit = 0

process.MessageLogger.debugs = cms.untracked.PSet( 
        threshold = cms.untracked.string('DEBUG'),
        DEBUG = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        INFO = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        WARNING = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        ERROR = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        L1GlobalTriggerRecord = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
        L1GtTrigReport = cms.untracked.PSet( limit = cms.untracked.int32(-1) )
        )

process.MessageLogger.warnings = cms.untracked.PSet( 
        threshold = cms.untracked.string('WARNING'),
        WARNING = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        ERROR = cms.untracked.PSet( limit = cms.untracked.int32(0) ),
        L1GlobalTriggerRecord = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
        L1GtTrigReport = cms.untracked.PSet( limit = cms.untracked.int32(-1) )
        )

process.MessageLogger.errors = cms.untracked.PSet( 
        threshold = cms.untracked.string('ERROR'),
        ERROR = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
        L1GlobalTriggerRecord = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
        L1GtTrigReport = cms.untracked.PSet( limit = cms.untracked.int32(-1) )
        )


# output 
process.outputL1GtRecord = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testGt_GtRecord_output.root'),
    # keep only unpacked data in the ROOT file
    outputCommands = cms.untracked.vstring('drop *', 
        'keep *_l1GtRecord_*_*')
)

process.outpath = cms.EndPath(process.outputL1GtRecord)

