#
# configuration file fragment containing user choices
# for release to run, sample and data type to use in
# a cfg file
#

#

import FWCore.ParameterSet.Config as cms
import sys

###################### user choices ######################

# (pre)release (cycle) to be run - it is used to choose a correct global tag
cmsRunRelease = 'CMSSW_3_11_X'
#cmsRunRelease = 'CMSSW_3_6_X'
#cmsRunRelease = 'CMSSW_3_5_X'

# choose (pre)release used to produce the RelVal samples (data are independent)
sampleFromRelease = 'CMSSW_3_11_0'
#sampleFromRelease = 'CMSSW_3_5_6'
#sampleFromRelease = 'CMSSW_3_5_2'
#sampleFromRelease = 'CMSSW_3_5_0'
#sampleFromRelease = 'CMSSW_3_4_1'
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

if useRelValSample == True :
    
    #globalTag = 'MC'
    globalTag = 'START'
    #globalTag = 'CRAFT'
    
    # RelVals samples 
    # not all combinations (sampleFromRelease, useSample, dataType) are included
    #
    #useSample = 'MinBias'
    useSample = 'RelValTTbar'
    #useSample = 'RelValQCD_Pt_80_120'
    #useSample = 'Cosmics_CRAFT09_R_V10'
     
    # data type
    #dataType = 'RECO'
    dataType = 'RAW'
   
else :

    # data type: StreamFile is not associated to these runs (no available files)
    #           for RAW data, the unpacker sequence RawToDigi will be also run
    dataType = 'RAW'
    #dataType = 'StreamFile'
    #dataType = 'RECO'
    
    #runNumber = '123596'
    #runNumber = '116035'
    #runNumber = '121560'
    #runNumber = '127715'
    #runNumber = '132440_132439_Cosmics'
    #runNumber = '132442_132440_MinimumBias_small'
    #runNumber = 'Commissioning10-Apr1Skim_Muon_skim-v1' 
    #runNumber = 'MinimumBias_Commissioning10-May13thReReco_preproduction-v1_RECO'
    #runNumber = '137028'
    runNumber = '156508'
       
# change to True to use local files
#     the type of file must be matched by hand
#     useGlobalTag must be also defined here

useLocalFiles = False 
#useLocalFiles = True 

if (useLocalFiles == True) :
    useGlobalTag = 'GR_P_V13'
    dataType = 'RECO'
    
# number of events to be run (-1 for all)
maxNumberEvents = 100
#maxNumberEvents = -1

###################### end user choices ###################


errorUserOptions = False

# global tags for the release used to run
if (useRelValSample == True) and (useLocalFiles == False) :
    
    if cmsRunRelease == 'CMSSW_3_11_X' :
        if globalTag == 'MC' :
            useGlobalTag = 'MC_311_V1'
        elif globalTag == 'START' :
            useGlobalTag = 'START_311_V1'
        else :
            print '\nError: no global tag defined for release', cmsRunRelease, 'of type', globalTag, 'used with RelVal sample'
            errorUserOptions = True
            
    elif cmsRunRelease == 'CMSSW_3_7_X' :
        if globalTag == 'MC' :
            useGlobalTag = 'MC_37Y_V4'
        elif globalTag == 'START' :
            useGlobalTag = 'START37_V4'
        elif globalTag == 'CRAFT' :
            useGlobalTag = 'CRFT9_36R_V03'
        else :
            print '\nError: no global tag defined for release', cmsRunRelease, 'of type', globalTag, 'used with RelVal sample'
            errorUserOptions = True
            
    elif cmsRunRelease == 'CMSSW_3_6_X' :
        if globalTag == 'MC' :
            useGlobalTag = 'MC_36Y_V7A'
        elif globalTag == 'START' :
            useGlobalTag = 'START36_V7'
        elif globalTag == 'CRAFT' :
            useGlobalTag = 'CRAFT09_R_V10'
        else :
            print '\nError: no global tag defined for release', cmsRunRelease, 'of type', globalTag, 'used with RelVal sample'
            errorUserOptions = True
            
    elif cmsRunRelease == 'CMSSW_3_5_X' :
        if globalTag == 'MC' :
            useGlobalTag = 'MC_3XY_V25'
        elif globalTag == 'START' :
            useGlobalTag = 'START3X_V25'
        elif globalTag == 'CRAFT09' :
            useGlobalTag = 'CRAFT09_R_V10'
        else :
            print '\nError: no global tag defined for release', cmsRunRelease, 'of type', globalTag, 'used with RelVal sample'
            errorUserOptions = True
    elif cmsRunRelease == 'CMSSW_3_4_X' :
        if globalTag == 'MC' :
            useGlobalTag = 'MC_3XY_V18'
        elif globalTag == 'START' :
            useGlobalTag = 'START3X_V18'
        else :
            print '\nError: no global tag defined for release', cmsRunRelease, 'of type', globalTag, 'used with RelVal sample'
            errorUserOptions = True
    elif cmsRunRelease == 'CMSSW_3_3_6' :
        if globalTag == 'MC' :
            useGlobalTag = 'MC_3XY_V9B'
        elif globalTag == 'START' :
            useGlobalTag = 'STARTUP3X_V8M'
        else :
            print '\nError: no global tag defined for release', cmsRunRelease, 'of type', globalTag, 'used with RelVal sample'
            errorUserOptions = True
    else :
        print 'Error: no global tag defined for release', cmsRunRelease, 'used with RelVal sample'
        errorUserOptions = True
        
   
elif (useRelValSample == False) and (useLocalFiles == False) :
    # global tag for data taking
    
    if cmsRunRelease == 'CMSSW_3_11_X' :
        useGlobalTag = 'GR_P_V13'
    elif cmsRunRelease == 'CMSSW_3_7_X' :
        useGlobalTag = 'GR_R_37X_V4'
    elif cmsRunRelease == 'CMSSW_3_6_X' :
        useGlobalTag = 'GR10_P_V4'
    elif cmsRunRelease == 'CMSSW_3_5_X' :
        useGlobalTag = 'GR10_P_V4'
    elif cmsRunRelease == 'CMSSW_3_4_1' :
        useGlobalTag = 'GR10_P_V3'
    elif cmsRunRelease == 'CMSSW_3_3_6' :
        useGlobalTag = 'GR10_P_V3'
    else :
        print 'Error: no global tag defined for release', cmsRunRelease, 'used with data sample'
        errorUserOptions = True
        
else :
       print 'Using local file(s) with global tag',  useGlobalTag, 'and release', cmsRunRelease
     

# source according to data type
if dataType == 'StreamFile' :
    readFiles = cms.untracked.vstring()
else :        
    readFiles = cms.untracked.vstring()
    secFiles = cms.untracked.vstring() 


# type of sample used (True for RelVal, False for data)

if (useRelValSample == True) and (useLocalFiles == False) :
    if useGlobalTag.count('IDEAL') or useGlobalTag.count('MC') :

        if (sampleFromRelease == 'CMSSW_3_7_0') and (useSample == 'RelValTTbar') and (dataType == 'RAW') :

            dataset = '/RelValTTbar/CMSSW_3_7_0-MC_37Y_V4-v1/GEN-SIM-DIGI-RAW-HLTDEBUG'
            print '   Running on dataset', dataset, '\n   produced with', sampleFromRelease, '\n   Global tag used to run:', useGlobalTag  
 
            readFiles.extend( [
                '/store/relval/CMSSW_3_7_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_37Y_V4-v1/0024/F06AD39F-3C69-DF11-BDDF-0026189438C2.root',
                '/store/relval/CMSSW_3_7_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_37Y_V4-v1/0024/F0067329-3B69-DF11-A543-0018F3D096BC.root',
                '/store/relval/CMSSW_3_7_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_37Y_V4-v1/0024/F00014C0-4069-DF11-8045-003048678E92.root',
                '/store/relval/CMSSW_3_7_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_37Y_V4-v1/0024/E61DF7FE-3869-DF11-AC16-00304867D446.root',
                '/store/relval/CMSSW_3_7_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_37Y_V4-v1/0024/D4EBA424-3B69-DF11-A8FE-003048B95B30.root',
                '/store/relval/CMSSW_3_7_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_37Y_V4-v1/0024/D0424F9F-3B69-DF11-A3CA-002618FDA277.root',
                '/store/relval/CMSSW_3_7_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_37Y_V4-v1/0024/C800F5A9-4369-DF11-839E-003048678F1C.root',
                '/store/relval/CMSSW_3_7_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_37Y_V4-v1/0024/BEC39F26-3B69-DF11-8D6E-002618943924.root',
                '/store/relval/CMSSW_3_7_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_37Y_V4-v1/0024/B6FDFF16-3B69-DF11-B4F6-00304867BEDE.root',
                '/store/relval/CMSSW_3_7_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_37Y_V4-v1/0024/B29ADB07-3A69-DF11-A379-0018F3D09696.root',
                '/store/relval/CMSSW_3_7_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_37Y_V4-v1/0024/98656090-3A69-DF11-B17D-002618943963.root',
                '/store/relval/CMSSW_3_7_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_37Y_V4-v1/0024/80412107-3A69-DF11-907D-0018F3D09696.root',
                '/store/relval/CMSSW_3_7_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_37Y_V4-v1/0024/7EF1D293-3969-DF11-9121-003048D15DDA.root',
                '/store/relval/CMSSW_3_7_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_37Y_V4-v1/0024/72F4FC4C-4169-DF11-95F1-0026189438F9.root',
                '/store/relval/CMSSW_3_7_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_37Y_V4-v1/0024/1EB7CE2A-3C69-DF11-9B67-002618FDA248.root',
                '/store/relval/CMSSW_3_7_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_37Y_V4-v1/0024/16173D1E-3C69-DF11-80C4-00304867D446.root',
                '/store/relval/CMSSW_3_7_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_37Y_V4-v1/0024/14FDF29D-3B69-DF11-A3F6-0026189438CC.root',
                '/store/relval/CMSSW_3_7_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_37Y_V4-v1/0024/0C27D0A6-3D69-DF11-99F5-003048678B88.root',
                '/store/relval/CMSSW_3_7_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_37Y_V4-v1/0024/0692BD93-3A69-DF11-8400-0026189438D7.root',
                '/store/relval/CMSSW_3_7_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_37Y_V4-v1/0024/04ECC126-3C69-DF11-AF07-002618943896.root' 
                ] );


        elif (sampleFromRelease == 'CMSSW_3_6_0_pre2') and (useSample == 'RelValTTbar') and (dataType == 'RAW') :

            dataset = '/RelValTTbar/CMSSW_3_6_0_pre2-MC_3XY_V24-v1/GEN-SIM-DIGI-RAW-HLTDEBUG'
            print '   Running on dataset', dataset, '\n   produced with', sampleFromRelease, '\n   Global tag used to run:', useGlobalTag  
        
            readFiles.extend( [
                '/store/relval/CMSSW_3_6_0_pre2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V24-v1/0001/9A4F4BE5-6E27-DF11-B3B6-002618FDA250.root',
                '/store/relval/CMSSW_3_6_0_pre2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V24-v1/0000/F656B24E-FA26-DF11-8E18-002618943977.root',
                '/store/relval/CMSSW_3_6_0_pre2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V24-v1/0000/D6B4B4E5-FA26-DF11-9EDB-001731A28461.root',
                '/store/relval/CMSSW_3_6_0_pre2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V24-v1/0000/C8AB5953-FA26-DF11-BDA7-002354EF3BE0.root',
                '/store/relval/CMSSW_3_6_0_pre2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V24-v1/0000/C8090554-FA26-DF11-93E7-0026189438DE.root',
                '/store/relval/CMSSW_3_6_0_pre2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V24-v1/0000/BCFB047B-FC26-DF11-A46D-002618943923.root',
                '/store/relval/CMSSW_3_6_0_pre2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V24-v1/0000/B6B2BBE6-FB26-DF11-949C-003048678B76.root',
                '/store/relval/CMSSW_3_6_0_pre2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V24-v1/0000/AC28B6F1-0227-DF11-B70C-002618943985.root',
                '/store/relval/CMSSW_3_6_0_pre2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V24-v1/0000/AA4D52E5-FA26-DF11-9A90-003048679182.root',
                '/store/relval/CMSSW_3_6_0_pre2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V24-v1/0000/A47604E7-FB26-DF11-9283-003048678B76.root',
                '/store/relval/CMSSW_3_6_0_pre2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V24-v1/0000/A01DBD80-FC26-DF11-B327-001731AF669D.root',
                '/store/relval/CMSSW_3_6_0_pre2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V24-v1/0000/90CEA8EA-FB26-DF11-BB4A-001731AF6861.root',
                '/store/relval/CMSSW_3_6_0_pre2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V24-v1/0000/7E5C77EF-0127-DF11-A946-002618943919.root',
                '/store/relval/CMSSW_3_6_0_pre2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V24-v1/0000/7CE1A5E8-FC26-DF11-A885-0026189438DD.root',
                '/store/relval/CMSSW_3_6_0_pre2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V24-v1/0000/68ADD4EF-FA26-DF11-B5A5-001731AF6943.root',
                '/store/relval/CMSSW_3_6_0_pre2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V24-v1/0000/60A3B079-FC26-DF11-BEC7-0026189438AD.root',
                '/store/relval/CMSSW_3_6_0_pre2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V24-v1/0000/542C6B69-0227-DF11-9E3A-00261894391C.root',
                '/store/relval/CMSSW_3_6_0_pre2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V24-v1/0000/52935EF5-FA26-DF11-9906-001731A284FD.root',
                '/store/relval/CMSSW_3_6_0_pre2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V24-v1/0000/4EFBD5E2-FA26-DF11-A5B8-001A92971B92.root',
                '/store/relval/CMSSW_3_6_0_pre2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V24-v1/0000/3E263E75-0327-DF11-AC09-001731AF68B7.root',
                '/store/relval/CMSSW_3_6_0_pre2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V24-v1/0000/32BD7D61-0127-DF11-8798-0026189438CC.root',
                '/store/relval/CMSSW_3_6_0_pre2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V24-v1/0000/0265CC6B-0327-DF11-BB53-003048678EE2.root'
                ]);

        elif (sampleFromRelease == 'CMSSW_3_5_2') and (useSample == 'RelValTTbar') and (dataType == 'RAW') :

            dataset = '/RelValTTbar/CMSSW_3_5_2-MC_3XY_V21-v1/GEN-SIM-DIGI-RAW-HLTDEBUG'
            print '   Running on dataset', dataset, '\n   produced with', sampleFromRelease, '\n   Global tag used to run:', useGlobalTag  
        
            readFiles.extend( [
                '/store/relval/CMSSW_3_5_2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V21-v1/0016/FAA58A57-3D1E-DF11-87A5-001731A283DF.root',
                '/store/relval/CMSSW_3_5_2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V21-v1/0016/E64BA05D-3A1E-DF11-8861-00261894380D.root',
                '/store/relval/CMSSW_3_5_2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V21-v1/0016/D80D18C7-311E-DF11-93E9-0018F3D09676.root',
                '/store/relval/CMSSW_3_5_2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V21-v1/0016/D27E27EA-391E-DF11-852C-0017319EB92B.root',
                '/store/relval/CMSSW_3_5_2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V21-v1/0016/CC8E2952-391E-DF11-8EE5-0018F3D096D8.root',
                '/store/relval/CMSSW_3_5_2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V21-v1/0016/CACE2BE1-371E-DF11-906C-001731AF685D.root',
                '/store/relval/CMSSW_3_5_2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V21-v1/0016/B6A6D3D4-3C1E-DF11-BCBC-001731AF68B9.root'
                ]);

        elif (sampleFromRelease == 'CMSSW_3_5_0') and (useSample == 'RelValQCD_Pt_80_120') and (dataType == 'RAW') :
            

            dataset = '/RelValQCD_Pt_80_120/CMSSW_3_5_0-MC_3XY_V21-v1/GEN-SIM-DIGI-RAW-HLTDEBUG'
            print '   Running on dataset', dataset, '\n   produced with', sampleFromRelease, '\n   Global tag used to run:', useGlobalTag  

            readFiles.extend( [
                '/store/relval/CMSSW_3_5_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V21-v1/0013/88F831A5-6313-DF11-991F-001731AF67B7.root',
                '/store/relval/CMSSW_3_5_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V21-v1/0012/F42A8127-3513-DF11-9DD1-003048678CA2.root',
                '/store/relval/CMSSW_3_5_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V21-v1/0012/F40CAC1A-3513-DF11-A0C4-00304867C0EA.root',
                '/store/relval/CMSSW_3_5_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V21-v1/0012/E69ECAF1-3313-DF11-81CD-002618943877.root',
                '/store/relval/CMSSW_3_5_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V21-v1/0012/D0549F7C-3413-DF11-87E2-00304867C0EA.root',
                '/store/relval/CMSSW_3_5_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V21-v1/0012/CA63A125-3513-DF11-9375-003048678E6E.root',
                '/store/relval/CMSSW_3_5_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V21-v1/0012/B4E128F1-3313-DF11-A1CE-00261894396D.root',
                '/store/relval/CMSSW_3_5_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V21-v1/0012/B218BB14-3513-DF11-94A6-003048678F6C.root',
                '/store/relval/CMSSW_3_5_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V21-v1/0012/B0090A13-3513-DF11-947C-00304867C0FC.root',
                '/store/relval/CMSSW_3_5_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V21-v1/0012/AC9D4128-3513-DF11-BAC1-003048678DD6.root',
                '/store/relval/CMSSW_3_5_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V21-v1/0012/A6B40B1B-3513-DF11-B28F-003048678F6C.root',
                '/store/relval/CMSSW_3_5_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V21-v1/0012/4EBE4890-3513-DF11-9880-002618943975.root',
                '/store/relval/CMSSW_3_5_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V21-v1/0012/480C798B-3513-DF11-A638-001731A28799.root',
                '/store/relval/CMSSW_3_5_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V21-v1/0012/34ABA2F1-3313-DF11-A46F-0026189438B1.root',
                '/store/relval/CMSSW_3_5_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V21-v1/0012/2E377CF1-3313-DF11-84C1-00248C55CC40.root',
                '/store/relval/CMSSW_3_5_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V21-v1/0012/28A84FEF-3313-DF11-A8FF-0026189438E2.root',
                '/store/relval/CMSSW_3_5_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V21-v1/0012/1EAF6F8F-3513-DF11-8921-00261894391B.root',
                '/store/relval/CMSSW_3_5_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V21-v1/0012/0EA81C90-3513-DF11-BFD1-0026189438C4.root' 
                ] );

        elif (sampleFromRelease == 'CMSSW_3_3_6') and (useSample == 'RelValTTbar') and (dataType == 'RAW') :

            dataset = '/RelValTTbar/CMSSW_3_3_6-MC_3XY_V9A-v1/GEN-SIM-DIGI-RAW-HLTDEBUG'
            print '   Running on dataset', dataset, '\n   produced with', sampleFromRelease, '\n   Global tag used to run:', useGlobalTag  
        
            readFiles.extend([
                '/store/relval/CMSSW_3_3_6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V9A-v1/0009/F6C6F406-3CE4-DE11-8F12-00304867BEE4.root',
                '/store/relval/CMSSW_3_3_6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V9A-v1/0009/B2898985-3BE4-DE11-98B2-00261894396A.root',
                '/store/relval/CMSSW_3_3_6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V9A-v1/0009/ACB9360D-3CE4-DE11-904D-00261894391D.root',
                '/store/relval/CMSSW_3_3_6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V9A-v1/0009/A6ACEC91-3CE4-DE11-A6FB-00261894390E.root'
               ]);

        elif (sampleFromRelease == 'CMSSW_2_2_12') and (useSample == 'RelValTTbar') and (dataType == 'RAW') :

            dataset = '/RelValTTbar/CMSSW_2_2_4_IDEAL_V11_v1/GEN-SIM-DIGI-RAW-HLTDEBUG'
            print '   Running on dataset', dataset, '\n   produced with', sampleFromRelease, '\n   Global tag used to run:', useGlobalTag  
        
            readFiles.extend([
                '/store/relval/CMSSW_2_2_4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V11_v1/0000/02697009-5CF3-DD11-A862-001D09F2423B.root',
                '/store/relval/CMSSW_2_2_4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V11_v1/0000/064657A8-59F3-DD11-ACA5-000423D991F0.root',
                '/store/relval/CMSSW_2_2_4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V11_v1/0000/0817F6DE-5BF3-DD11-880D-0019DB29C5FC.root',
                '/store/relval/CMSSW_2_2_4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V11_v1/0000/0899697C-5AF3-DD11-9D21-001617DBD472.root'
                ]);

        elif (sampleFromRelease == 'CMSSW_3_5_0') and (useSample == 'RelValQCD_Pt_80_120') and (dataType == 'RECO') :

            dataset = '/RelValQCD_Pt_80_120/CMSSW_3_5_0-MC_3XY_V21-v1/GEN-SIM-RECO'
            print '   Running on dataset', dataset, '\n   produced with', sampleFromRelease, '\n   Global tag used to run:', useGlobalTag  

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


        elif (sampleFromRelease == 'CMSSW_3_6_0_pre2') and (useSample == 'RelValTTbar') and (dataType == 'RECO') :
            
            dataset = '/RelValTTbar/CMSSW_3_6_0_pre2-MC_3XY_V24-v1/GEN-SIM-RECO'
            print '   Running on dataset', dataset, '\n   produced with', sampleFromRelease, '\n   Global tag used to run:', useGlobalTag  

            readFiles.extend( [
                '/store/relval/CMSSW_3_6_0_pre2/RelValTTbar/GEN-SIM-RECO/MC_3XY_V24-v1/0001/6E01AFE7-6E27-DF11-986F-002618FDA207.root',
                '/store/relval/CMSSW_3_6_0_pre2/RelValTTbar/GEN-SIM-RECO/MC_3XY_V24-v1/0000/FE3124ED-FC26-DF11-A062-0026189438D7.root',
                '/store/relval/CMSSW_3_6_0_pre2/RelValTTbar/GEN-SIM-RECO/MC_3XY_V24-v1/0000/F8B934F6-0127-DF11-A603-003048678F74.root',
                '/store/relval/CMSSW_3_6_0_pre2/RelValTTbar/GEN-SIM-RECO/MC_3XY_V24-v1/0000/7C3F3EE5-FA26-DF11-80F9-0030486792B8.root',
                '/store/relval/CMSSW_3_6_0_pre2/RelValTTbar/GEN-SIM-RECO/MC_3XY_V24-v1/0000/50EBB34F-FA26-DF11-8A49-002618943845.root',
                '/store/relval/CMSSW_3_6_0_pre2/RelValTTbar/GEN-SIM-RECO/MC_3XY_V24-v1/0000/36CA96F4-0227-DF11-A7A8-002618943979.root',
                '/store/relval/CMSSW_3_6_0_pre2/RelValTTbar/GEN-SIM-RECO/MC_3XY_V24-v1/0000/2A4745DC-FB26-DF11-B535-0026189438AD.root',
                '/store/relval/CMSSW_3_6_0_pre2/RelValTTbar/GEN-SIM-RECO/MC_3XY_V24-v1/0000/0C9C6D7C-FC26-DF11-B5F6-001731AF68C1.root',
                '/store/relval/CMSSW_3_6_0_pre2/RelValTTbar/GEN-SIM-RECO/MC_3XY_V24-v1/0000/0256FAE6-FA26-DF11-8B72-002618FDA277.root'                
                ] );

        elif (sampleFromRelease == 'CMSSW_3_5_0') and (useSample == 'RelValTTbar') and (dataType == 'RECO') :
            
            dataset = '/RelValTTbar/CMSSW_3_5_0-MC_3XY_V21-v1/GEN-SIM-RECO'
            print '   Running on dataset', dataset, '\n   produced with', sampleFromRelease, '\n   Global tag used to run:', useGlobalTag  

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

        elif (sampleFromRelease == 'CMSSW_3_4_1') and (useSample == 'RelValQCD_Pt_80_120') and (dataType == 'RECO') :

            dataset = '/RelValQCD_Pt_80_120/CMSSW_3_4_1-MC_3XY_V14-v1/GEN-SIM-RECO'
            print '   Running on dataset', dataset, '\n   produced with', sampleFromRelease, '\n   Global tag used to run:', useGlobalTag  

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

        elif (sampleFromRelease == 'CMSSW_3_4_1') and (useSample == 'RelValTTbar') and (dataType == 'RECO') :
            
            dataset = '/RelValTTbar/CMSSW_3_4_1-MC_3XY_V14-v1/GEN-SIM-RECO'
            print '   Running on dataset', dataset, '\n   produced with', sampleFromRelease, '\n   Global tag used to run:', useGlobalTag  

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
            print 'Error: no files for:\n  sample', useSample, '\n  type', dataType, '\n  produced with', sampleFromRelease, '\n  global tag of type ', useGlobalTag    
            errorUserOptions = True
            

        secFiles.extend([
            ])



    elif useGlobalTag.count('START') :

        if (sampleFromRelease == 'CMSSW_3_7_0') and (useSample == 'RelValTTbar') and (dataType == 'RAW') :

            dataset = '/RelValTTbar/CMSSW_3_7_0-START37_V4-v1/GEN-SIM-DIGI-RAW-HLTDEBUG'
            print '   Running on dataset', dataset, '\n   produced with', sampleFromRelease, '\n   Global tag used to run:', useGlobalTag  
 
            readFiles.extend( [
                '/store/relval/CMSSW_3_7_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0025/DAA27EF5-5069-DF11-9B53-002618943982.root',
                '/store/relval/CMSSW_3_7_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0025/D23295F2-5069-DF11-8EAD-002354EF3BDE.root',
                '/store/relval/CMSSW_3_7_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0025/BEEC3A7D-5269-DF11-B4BD-003048678A88.root',
                '/store/relval/CMSSW_3_7_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0025/BCBC54C8-6069-DF11-A680-003048678ADA.root',
                '/store/relval/CMSSW_3_7_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0025/B28BFFF5-4F69-DF11-8BA5-002618943969.root',
                '/store/relval/CMSSW_3_7_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0025/A2CF55FD-4F69-DF11-AD67-00261894395F.root',
                '/store/relval/CMSSW_3_7_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0025/8AB3AA68-5069-DF11-9629-001A928116B8.root',
                '/store/relval/CMSSW_3_7_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0025/8813F4FB-5169-DF11-A602-0026189438DB.root',
                '/store/relval/CMSSW_3_7_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0025/72382A69-4F69-DF11-86D5-0018F3D096D2.root',
                '/store/relval/CMSSW_3_7_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0025/3C83A8F2-5069-DF11-8691-002618943856.root',
                '/store/relval/CMSSW_3_7_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0025/36D79AE9-5069-DF11-BEC3-00261894397D.root',
                '/store/relval/CMSSW_3_7_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0025/101CA9FE-4F69-DF11-BA81-002618943979.root',
                '/store/relval/CMSSW_3_7_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0025/0CB86B26-5A69-DF11-8D46-00261894380A.root',
                '/store/relval/CMSSW_3_7_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0025/064A62C7-5E69-DF11-9C52-003048678F84.root',
                '/store/relval/CMSSW_3_7_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0024/F6E38566-4E69-DF11-8E5E-0030486790BE.root',
                '/store/relval/CMSSW_3_7_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0024/F4CBE7D7-4E69-DF11-92FB-0026189438E3.root',
                '/store/relval/CMSSW_3_7_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0024/C2D99BDA-4D69-DF11-A886-002618943934.root',
                '/store/relval/CMSSW_3_7_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0024/BCD64BD9-4D69-DF11-B135-00261894396B.root',
                '/store/relval/CMSSW_3_7_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0024/5C5DF74D-4D69-DF11-B0BA-00261894387E.root',
                '/store/relval/CMSSW_3_7_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V4-v1/0024/200B80D9-4E69-DF11-BBE9-00261894396B.root' 
                ] );
       
        elif (sampleFromRelease == 'CMSSW_3_6_0_pre2') and (useSample == 'RelValTTbar') and (dataType == 'RAW') :

            dataset = '/RelValTTbar/CMSSW_3_6_0_pre2-START3X_V24-v1/GEN-SIM-DIGI-RAW-HLTDEBUG'
            print '   Running on dataset', dataset, '\n   produced with', sampleFromRelease, '\n   Global tag used to run:', useGlobalTag  
        
            readFiles.extend( [
                '/store/relval/CMSSW_3_6_0_pre2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0002/A047940F-A827-DF11-9015-0026189438E6.root',
                '/store/relval/CMSSW_3_6_0_pre2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0001/3AFEFC61-6F27-DF11-B472-002618FDA262.root',
                '/store/relval/CMSSW_3_6_0_pre2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0000/EE17E540-2627-DF11-B315-001731AF6A4F.root',
                '/store/relval/CMSSW_3_6_0_pre2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0000/E6EECCCA-2627-DF11-B6CD-001731AF684D.root',
                '/store/relval/CMSSW_3_6_0_pre2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0000/E0FA4D9A-2027-DF11-BFEE-002618943916.root',
                '/store/relval/CMSSW_3_6_0_pre2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0000/C68312A3-2027-DF11-96D3-0017312B5BAD.root',
                '/store/relval/CMSSW_3_6_0_pre2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0000/C0739436-2127-DF11-82A3-003048678D9A.root',
                '/store/relval/CMSSW_3_6_0_pre2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0000/AE526E30-2127-DF11-8813-003048678D86.root',
                '/store/relval/CMSSW_3_6_0_pre2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0000/90A69925-2027-DF11-8C36-002618943916.root',
                '/store/relval/CMSSW_3_6_0_pre2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0000/8AF69B24-2027-DF11-B10A-002618943944.root',
                '/store/relval/CMSSW_3_6_0_pre2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0000/82837EC9-2727-DF11-A754-001731AF684D.root',
                '/store/relval/CMSSW_3_6_0_pre2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0000/7022376B-2727-DF11-883F-001731AF684D.root',
                '/store/relval/CMSSW_3_6_0_pre2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0000/6C0A9A2F-2127-DF11-B055-003048679000.root',
                '/store/relval/CMSSW_3_6_0_pre2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0000/529F2059-2827-DF11-8BAC-001731A28F19.root',
                '/store/relval/CMSSW_3_6_0_pre2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0000/4270DF57-2227-DF11-ABFE-003048678F84.root',
                '/store/relval/CMSSW_3_6_0_pre2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0000/40425AA5-2027-DF11-86CE-001731AF65E7.root',
                '/store/relval/CMSSW_3_6_0_pre2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0000/3664214F-2927-DF11-AEEF-00261894397F.root',
                '/store/relval/CMSSW_3_6_0_pre2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0000/22AF8C2D-2127-DF11-BB88-00304867BFB0.root',
                '/store/relval/CMSSW_3_6_0_pre2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0000/1ADD64A8-2027-DF11-8CD6-0018F3D09698.root',
                '/store/relval/CMSSW_3_6_0_pre2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0000/18AAFE69-2727-DF11-9DEF-001731AF687F.root',
                '/store/relval/CMSSW_3_6_0_pre2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0000/1482955D-2827-DF11-8EF6-001731A28A31.root',
                '/store/relval/CMSSW_3_6_0_pre2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0000/0E866F4F-2227-DF11-9C0C-003048678D9A.root',
                '/store/relval/CMSSW_3_6_0_pre2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V24-v1/0000/04A74FCB-2727-DF11-BFE5-001731AF68B9.root'
                ]);

        elif (sampleFromRelease == 'CMSSW_3_5_6') and (useSample == 'MinBias') and (dataType == 'RAW') :

            dataset = '/MinBias/Spring10-START3X_V25B-v1/GEN-SIM-RAW'
            print '   Running on dataset', dataset, '\n   produced with', sampleFromRelease, '\n   Global tag used to run:', useGlobalTag  
        
            readFiles.extend( [
                '/store/mc/Spring10/MinBias/GEN-SIM-RAW/START3X_V25B-v1/0104/FECFDECD-9739-DF11-A00E-001A92971AAA.root',
                '/store/mc/Spring10/MinBias/GEN-SIM-RAW/START3X_V25B-v1/0104/FE747EEC-9D39-DF11-BD36-0018F3D09636.root',
                '/store/mc/Spring10/MinBias/GEN-SIM-RAW/START3X_V25B-v1/0104/FE706011-9139-DF11-8F2A-0018F3D096B6.root',
                '/store/mc/Spring10/MinBias/GEN-SIM-RAW/START3X_V25B-v1/0104/FE476F57-9239-DF11-BA70-002618943950.root',
                '/store/mc/Spring10/MinBias/GEN-SIM-RAW/START3X_V25B-v1/0104/FE12184C-8E39-DF11-89F8-00304867905A.root'
               ]);

        elif (sampleFromRelease == 'CMSSW_3_5_2') and (useSample == 'RelValTTbar') and (dataType == 'RAW') :

            dataset = '/RelValTTbar/CMSSW_3_5_2-START3X_V21-v1/GEN-SIM-DIGI-RAW-HLTDEBUG'
            print '   Running on dataset', dataset, '\n   produced with', sampleFromRelease, '\n   Global tag used to run:', useGlobalTag  
        
            readFiles.extend( [
                '/store/relval/CMSSW_3_5_2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V21-v1/0016/FE2573D6-381E-DF11-9B55-001731AF678D.root',
                '/store/relval/CMSSW_3_5_2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V21-v1/0016/F687B04E-331E-DF11-B1C3-0018F3D0960E.root',
                '/store/relval/CMSSW_3_5_2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V21-v1/0016/D2E9E5C7-2D1E-DF11-AA1D-003048679296.root',
                '/store/relval/CMSSW_3_5_2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V21-v1/0016/D0B6CCE8-321E-DF11-ABEE-001A92971BD6.root',
                '/store/relval/CMSSW_3_5_2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V21-v1/0016/AE6887CC-2C1E-DF11-90DD-001A928116E0.root',
                '/store/relval/CMSSW_3_5_2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V21-v1/0016/9E058FDD-361E-DF11-A10F-0017313F02F2.root',
                '/store/relval/CMSSW_3_5_2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V21-v1/0016/9098C5D7-2B1E-DF11-8AD9-001A92971AD8.root'
                ]);

        elif (sampleFromRelease == 'CMSSW_3_5_0') and (useSample == 'RelValQCD_Pt_80_120') and (dataType == 'RAW') :
        
            dataset = '/RelValQCD_Pt_80_120/CMSSW_3_5_0-START3X_V21-v1/GEN-SIM-DIGI-RAW-HLTDEBUG'
            print '   Running on dataset', dataset, '\n   produced with', sampleFromRelease, '\n   Global tag used to run:', useGlobalTag  

            readFiles.extend( [
                '/store/relval/CMSSW_3_5_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V21-v1/0013/A2650844-3F13-DF11-9F8B-0018F3D096C6.root',
                '/store/relval/CMSSW_3_5_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V21-v1/0013/7C862077-6213-DF11-A55E-0018F3D096DA.root',
                '/store/relval/CMSSW_3_5_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V21-v1/0013/64717429-3E13-DF11-A79C-0018F3D096DC.root',
                '/store/relval/CMSSW_3_5_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V21-v1/0013/5AB9E5C4-3E13-DF11-A5C6-0018F3D096AE.root',
                '/store/relval/CMSSW_3_5_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V21-v1/0013/3A229928-3E13-DF11-A0F6-0018F3D09702.root',
                '/store/relval/CMSSW_3_5_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V21-v1/0013/320738C4-3F13-DF11-B7EF-0018F3D09710.root',
                '/store/relval/CMSSW_3_5_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V21-v1/0013/2882F7B5-4013-DF11-A8AB-001A92971B64.root',
                '/store/relval/CMSSW_3_5_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V21-v1/0013/2437DA46-3F13-DF11-8BD2-001A92810ACA.root',
                '/store/relval/CMSSW_3_5_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V21-v1/0013/20B66439-3D13-DF11-A811-001A92971B0C.root',
                '/store/relval/CMSSW_3_5_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V21-v1/0013/16000C72-4013-DF11-B408-001A9281172E.root',
                '/store/relval/CMSSW_3_5_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V21-v1/0013/0604FA28-3E13-DF11-81A6-001A92971B06.root',
                '/store/relval/CMSSW_3_5_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V21-v1/0013/0453E0BD-3E13-DF11-AC08-001A92971B94.root',
                '/store/relval/CMSSW_3_5_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V21-v1/0012/E4A7683D-3913-DF11-A8DA-002618943852.root',
                '/store/relval/CMSSW_3_5_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V21-v1/0012/D6AF091A-3A13-DF11-AEFE-00304867901A.root',
                '/store/relval/CMSSW_3_5_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V21-v1/0012/D0EDD332-3813-DF11-81D8-003048678F8E.root',
                '/store/relval/CMSSW_3_5_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V21-v1/0012/BC3943E5-3813-DF11-AF6D-00304867901A.root',
                '/store/relval/CMSSW_3_5_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V21-v1/0012/B8631430-3813-DF11-894F-00261894393C.root',
                '/store/relval/CMSSW_3_5_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V21-v1/0012/20C261E3-3913-DF11-AC49-002618FDA287.root' 
                ] );


        elif (sampleFromRelease == 'CMSSW_3_3_6') and (useSample == 'RelValTTbar') and (dataType == 'RAW') :

            dataset = '/RelValTTbar/CMSSW_3_3_6-STARTUP3X_V8H-v1/GEN-SIM-DIGI-RAW-HLTDEBUG'
            print '   Running on dataset', dataset, '\n   produced with', sampleFromRelease, '\n   Global tag used to run:', useGlobalTag  
        
            readFiles.extend([
                '/store/relval/CMSSW_3_3_6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V8H-v1/0009/E44B9490-3BE4-DE11-962B-0026189437FD.root',
                '/store/relval/CMSSW_3_3_6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V8H-v1/0009/E0BA5492-3BE4-DE11-9417-002618943926.root',
                '/store/relval/CMSSW_3_3_6/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP3X_V8H-v1/0009/6827DCDF-9EE4-DE11-8A58-002618943920.root'
               ]);

        elif (sampleFromRelease == 'CMSSW_2_2_12') and (useSample == 'RelValTTbar') and (dataType == 'RAW') :
            
            dataset = '/RelValTTbar/CMSSW_2_2_4_STARTUP_V8_v1/GEN-SIM-DIGI-RAW-HLTDEBUG'
            print '   Running on', useSample, 'sample produced with', sampleFromRelease, '. Global tag used to run:', useGlobalTag  
        
            readFiles.extend([
                '/store/relval/CMSSW_2_2_4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V8_v1/0000/069AA022-5BF3-DD11-9A56-001617E30D12.root',
                '/store/relval/CMSSW_2_2_4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V8_v1/0000/08DA99A6-5AF3-DD11-AAC1-001D09F24493.root',
                '/store/relval/CMSSW_2_2_4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V8_v1/0000/0A725E15-5BF3-DD11-8B4B-000423D99CEE.root',
                '/store/relval/CMSSW_2_2_4/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V8_v1/0000/0AF5B676-5AF3-DD11-A22F-001617DBCF1E.root'
                ]);

        elif (sampleFromRelease == 'CMSSW_3_6_0_pre2') and (useSample == 'RelValTTbar') and (dataType == 'RECO') :
        
            dataset = '/RelValTTbar/CMSSW_3_6_0_pre2-START3X_V24-v1/GEN-SIM-RECO'
            print '   Running on dataset', dataset, '\n   produced with', sampleFromRelease, '\n   Global tag used to run:', useGlobalTag  

            readFiles.extend( [
                '/store/relval/CMSSW_3_6_0_pre2/RelValTTbar/GEN-SIM-RECO/START3X_V24-v1/0002/4610810E-A827-DF11-906C-002618943807.root',
                '/store/relval/CMSSW_3_6_0_pre2/RelValTTbar/GEN-SIM-RECO/START3X_V24-v1/0001/EC4E796F-6F27-DF11-BE30-002618FDA207.root',
                '/store/relval/CMSSW_3_6_0_pre2/RelValTTbar/GEN-SIM-RECO/START3X_V24-v1/0000/F2DE9C36-2127-DF11-B6CD-003048678D9A.root',
                '/store/relval/CMSSW_3_6_0_pre2/RelValTTbar/GEN-SIM-RECO/START3X_V24-v1/0000/986832A9-2027-DF11-BA3C-001731A28319.root',
                '/store/relval/CMSSW_3_6_0_pre2/RelValTTbar/GEN-SIM-RECO/START3X_V24-v1/0000/72E3CCBE-2027-DF11-A796-001731AF6A4F.root',
                '/store/relval/CMSSW_3_6_0_pre2/RelValTTbar/GEN-SIM-RECO/START3X_V24-v1/0000/64700BBC-2427-DF11-B061-00261894393B.root',
                '/store/relval/CMSSW_3_6_0_pre2/RelValTTbar/GEN-SIM-RECO/START3X_V24-v1/0000/38E12CCB-2727-DF11-B4DA-001731A281B1.root',
                '/store/relval/CMSSW_3_6_0_pre2/RelValTTbar/GEN-SIM-RECO/START3X_V24-v1/0000/385D7755-2727-DF11-9F17-001731A28F19.root',
                '/store/relval/CMSSW_3_6_0_pre2/RelValTTbar/GEN-SIM-RECO/START3X_V24-v1/0000/306EBF32-2127-DF11-8865-003048678D86.root',
                '/store/relval/CMSSW_3_6_0_pre2/RelValTTbar/GEN-SIM-RECO/START3X_V24-v1/0000/0069AC5F-2827-DF11-8986-001731AF698F.root'
                ] );

        elif (sampleFromRelease == 'CMSSW_3_5_6') and (useSample == 'MinBias') and (dataType == 'RECO') :
            
            dataset = '/MinBias/Spring10-START3X_V26A_356ReReco-v1/GEN-SIM-RECO'
            print '   Running on dataset', dataset, '\n   produced with', sampleFromRelease, '\n   Global tag used to run:', useGlobalTag  

            readFiles.extend( [
                '/store/mc/Spring10/MinBias/GEN-SIM-RECO/START3X_V26A_356ReReco-v1/0009/FEFC70B6-F53D-DF11-B57E-003048679150.root',
                '/store/mc/Spring10/MinBias/GEN-SIM-RECO/START3X_V26A_356ReReco-v1/0009/FED8673E-F53D-DF11-9E58-0026189437EB.root',
                '/store/mc/Spring10/MinBias/GEN-SIM-RECO/START3X_V26A_356ReReco-v1/0009/FEBF7874-EF3D-DF11-910D-002354EF3BDF.root',
                '/store/mc/Spring10/MinBias/GEN-SIM-RECO/START3X_V26A_356ReReco-v1/0009/FEA8ECD8-F13D-DF11-8EBD-00304867BFAE.root',
                '/store/mc/Spring10/MinBias/GEN-SIM-RECO/START3X_V26A_356ReReco-v1/0009/FE838E9F-F43D-DF11-BEBA-00261894393B.root'
                ] );

        elif (sampleFromRelease == 'CMSSW_3_5_0') and (useSample == 'RelValQCD_Pt_80_120') and (dataType == 'RECO') :
        
            dataset = '/RelValQCD_Pt_80_120/CMSSW_3_5_0-START3X_V21-v1/GEN-SIM-RECO'
            print '   Running on dataset', dataset, '\n   produced with', sampleFromRelease, '\n   Global tag used to run:', useGlobalTag  

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


        elif (sampleFromRelease == 'CMSSW_3_5_0') and (useSample == 'RelValTTbar') and (dataType == 'RECO') :
        
            dataset = '/RelValTTbar/CMSSW_3_5_0-START3X_V21-v1/GEN-SIM-RECO'
            print '   Running on dataset', dataset, '\n   produced with', sampleFromRelease, '\n   Global tag used to run:', useGlobalTag  

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

        elif (sampleFromRelease == 'CMSSW_3_4_1') and (useSample == 'RelValQCD_Pt_80_120') and (dataType == 'RECO') :
        
            dataset = '/RelValQCD_Pt_80_120/CMSSW_3_4_1-STARTUP3X_V14-v1/GEN-SIM-RECO'
            print '   Running on dataset', dataset, '\n   produced with', sampleFromRelease, '\n   Global tag used to run:', useGlobalTag  

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


        elif (sampleFromRelease == 'CMSSW_3_4_1') and (useSample == 'RelValTTbar') and (dataType == 'RECO') :
        
            dataset = '/RelValTTbar/CMSSW_3_4_1-STARTUP3X_V14-v1/GEN-SIM-RECO'
            print '   Running on dataset', dataset, '\n   produced with', sampleFromRelease, '\n   Global tag used to run:', useGlobalTag  

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
            print 'Error: no files for:\n  sample', useSample, '\n  type', dataType, '\n  produced with', sampleFromRelease, '\n  global tag of type ', useGlobalTag    
            errorUserOptions = True
            

        secFiles.extend([
            ])
    elif useGlobalTag.count('CRAFT') :

        if (sampleFromRelease == 'CMSSW_3_6_0_pre2') and (useSample == 'Cosmics_CRAFT09_R_V10') and (dataType == 'RECO') :
            
            dataset = '/Cosmics/CMSSW_3_6_0_pre2-CRAFT09_R_V10_RelVal_col_09-v1/RECO'
            print '   Running on dataset', dataset, '\n   produced with', sampleFromRelease, '\n   Global tag used to run:', useGlobalTag  

            readFiles.extend( [
                '/store/relval/CMSSW_3_6_0_pre2/Cosmics/RECO/CRAFT09_R_V10_RelVal_col_09-v1/0000/FE7AD6DA-E626-DF11-A138-001A928116EE.root',
                '/store/relval/CMSSW_3_6_0_pre2/Cosmics/RECO/CRAFT09_R_V10_RelVal_col_09-v1/0000/FCAF7AF4-E626-DF11-99A6-001A92971B30.root',
                '/store/relval/CMSSW_3_6_0_pre2/Cosmics/RECO/CRAFT09_R_V10_RelVal_col_09-v1/0000/FA1172F7-E626-DF11-B815-001A92971B9C.root',
                '/store/relval/CMSSW_3_6_0_pre2/Cosmics/RECO/CRAFT09_R_V10_RelVal_col_09-v1/0000/F8D4FCF1-E626-DF11-9E8C-001A92971BBA.root',
                '/store/relval/CMSSW_3_6_0_pre2/Cosmics/RECO/CRAFT09_R_V10_RelVal_col_09-v1/0000/F8B1552A-E726-DF11-87EE-001A92810A9E.root',
                '/store/relval/CMSSW_3_6_0_pre2/Cosmics/RECO/CRAFT09_R_V10_RelVal_col_09-v1/0000/F2448527-E726-DF11-A88C-001A92810A92.root',
                '/store/relval/CMSSW_3_6_0_pre2/Cosmics/RECO/CRAFT09_R_V10_RelVal_col_09-v1/0000/F22AE4F7-3527-DF11-996D-001A92971BBE.root',
                '/store/relval/CMSSW_3_6_0_pre2/Cosmics/RECO/CRAFT09_R_V10_RelVal_col_09-v1/0000/EC73370B-E726-DF11-B9A1-001BFCDBD130.root',
                '/store/relval/CMSSW_3_6_0_pre2/Cosmics/RECO/CRAFT09_R_V10_RelVal_col_09-v1/0000/E8C39227-E726-DF11-B380-001A92810A92.root',
                '/store/relval/CMSSW_3_6_0_pre2/Cosmics/RECO/CRAFT09_R_V10_RelVal_col_09-v1/0000/E42E69DB-E626-DF11-9603-001A928116EE.root',
                '/store/relval/CMSSW_3_6_0_pre2/Cosmics/RECO/CRAFT09_R_V10_RelVal_col_09-v1/0000/DC996D28-E726-DF11-AB86-001BFCDBD1BA.root',
                '/store/relval/CMSSW_3_6_0_pre2/Cosmics/RECO/CRAFT09_R_V10_RelVal_col_09-v1/0000/DC7E7F27-E726-DF11-AF51-001A92810A92.root',
                '/store/relval/CMSSW_3_6_0_pre2/Cosmics/RECO/CRAFT09_R_V10_RelVal_col_09-v1/0000/DA6FA4D7-E626-DF11-B604-001A928116BA.root'
                ] );

        else :
            print 'Error: no files for:\n  sample', useSample, '\n  type', dataType, '\n  produced with', sampleFromRelease, '\n  global tag of type ', useGlobalTag    
            errorUserOptions = True

    else :
        print 'Error: Global Tag', useGlobalTag, 'not defined.'    
        errorUserOptions = True
        

elif (useRelValSample == False) and (useLocalFiles == False) :

    # data

    if dataType == 'RAW' : 

        if runNumber == '156508' :
            dataset = '/MinimumBias/Commissioning11-v1/RAW'
            print '   Running on dataset:', dataset, 'with global tag ', useGlobalTag 
    
            readFiles.extend( [
                '/store/data/Commissioning11/MinimumBias/RAW/v1/000/156/508/02BC73A4-042F-E011-97DD-001D09F29146.root' 
                ] );

            secFiles.extend([
                ])

    
        elif runNumber == '137028' :
            dataset = '/Run2010A/ZeroBias/RAW'
            print '   Running on dataset:', dataset, 'with global tag ', useGlobalTag 
    
            readFiles.extend( [
                '/store/data/Run2010A/ZeroBias/RAW/v1/000/137/028/0C88B386-3971-DF11-A163-000423D99896.root' 
                ] );

            secFiles.extend([
                ])

    
        elif runNumber == '123596' :
            dataset = '/Cosmics/BeamCommissioning09-v1/RAW'
            print '   Running on dataset:', dataset, 'with global tag ', useGlobalTag 
    
            readFiles.extend( [
                '/store/data/BeamCommissioning09/Cosmics/RAW/v1/000/123/596/8E21B4C8-74E2-DE11-ABAA-000423D999CA.root' 
                ] );

            secFiles.extend([
                ])

    
        elif runNumber == '116035' :
            dataset = '/Cosmics/Commissioning09-v3/RAW'
            print '   Running on dataset:', dataset, 'with global tag ', useGlobalTag 
    
            readFiles.extend( [                        
                '/store/data/Commissioning09/Cosmics/RAW/v3/000/116/035/34A8317D-76AF-DE11-91DB-000423D98DC4.root'
                ]);                                                                                               

            secFiles.extend([
                ])
        
        elif runNumber == '121560' :
            dataset = '/Cosmics/Commissioning09-v3/RAW'
            print '   Running on dataset:', dataset, 'with global tag ', useGlobalTag 
    
            readFiles.extend( [                        
                '/store/data/BeamCommissioning09/Cosmics/RAW/v1/000/121/560/DC089E4B-5ED4-DE11-A179-000423D98FBC.root'
                ]);                                                                                               

            secFiles.extend([
                ])

        elif runNumber == '127715' :
            dataset = '/Cosmics/Commissioning10-v3/RAW'
            print '   Running on dataset:', dataset, 'with global tag ', useGlobalTag 
    
            readFiles.extend( [                        
                '/store/data/Commissioning10/Cosmics/RAW/v3/000/127/715/FCB12D5F-6C18-DF11-AB4B-000423D174FE.root'
                ]);                                                                                               

            secFiles.extend([
                ])

        elif runNumber == '132440_132439_Cosmics' :
            dataset = '/Cosmics/Commissioning10-v4/RAW'
            print '   Running on dataset:', dataset, 'with global tag ', useGlobalTag 
    
            readFiles.extend( [                        
                '/store/data/Commissioning10/Cosmics/RAW/v4/000/132/440/72DAEFC2-1A3C-DF11-A352-0030487A195C.root',
                '/store/data/Commissioning10/Cosmics/RAW/v4/000/132/439/A689F088-EE3B-DF11-A241-000423D99896.root'
                ]);                                                                                               

            secFiles.extend([
                ])

        elif runNumber == 'Commissioning10-Apr1Skim_Muon_skim-v1' :
            dataset = '/MinimumBias/Commissioning10-Apr1Skim_Muon_skim-v1/RAW-RECO'
            print '   Running on dataset:', dataset, 'run', runNumber, 'with global tag ', useGlobalTag 
    
            readFiles.extend( [                        
                '/store/data/Commissioning10/MinimumBias/RAW-RECO/Apr1Skim_Muon_skim-v1/0139/047E236C-B03E-DF11-8A23-002618FDA204.root',
                '/store/data/Commissioning10/MinimumBias/RAW-RECO/Apr1Skim_Muon_skim-v1/0139/040A2472-C83E-DF11-85C6-002618FDA259.root' 
                ]);                                                                                               

            secFiles.extend([
                ])

        else :
            print 'Error: run', runNumber, 'has no RAW file(s) available.'    
            errorUserOptions = True
            

    elif dataType == 'RECO' : 
        # data
        if runNumber == '137028' :
            dataset = '/Run2010A/ZeroBias/RECO-v2'
            print '   Running on dataset:', dataset, 'run', runNumber, 'with global tag ', useGlobalTag 

            readFiles.extend( [
                '/store/data/Run2010A/ZeroBias/RECO/v2/000/137/028/08BF857D-2471-DF11-9CEB-003048D2BBF0.root'
        
                ] );
        elif runNumber == '123596' :
            dataset = '/Cosmics/BeamCommissioning09-v2/RECO'
            print '   Running on dataset:', dataset, 'run', runNumber, 'with global tag ', useGlobalTag 

            readFiles.extend( [
                '/store/data/BeamCommissioning09/Cosmics/RECO/v2/000/123/596/FC5C3B0F-8AE2-DE11-A905-003048D37456.root'
        
                ] );
        elif runNumber == '127715' :
            dataset = '/Cosmics/Commissioning10-v3/RECO'
            print '   Running on dataset:', dataset, 'run', runNumber, 'with global tag ', useGlobalTag 
    
            readFiles.extend( [                        
                '/store/data/Commissioning10/Cosmics/RECO/v3/000/127/715/261A3141-9F18-DF11-883E-001D09F24493.root'
                ]);                                                                                               

            secFiles.extend([
                ])

        elif runNumber == '132442_132440_MinimumBias_small' :
            dataset = '/MinimumBias/Commissioning10-PromptReco-v7/RECO'
            print '   Running on dataset:', dataset, 'run', runNumber, 'with global tag ', useGlobalTag 
    
            readFiles.extend( [                        
                '/store/data/Commissioning10/MinimumBias/RECO/v7/000/132/442/102D0664-273C-DF11-A013-00304879FA4C.root',
                '/store/data/Commissioning10/MinimumBias/RECO/v7/000/132/440/F4C92A98-163C-DF11-9788-0030487C7392.root',
                '/store/data/Commissioning10/MinimumBias/RECO/v7/000/132/440/F427D642-173C-DF11-A909-0030487C60AE.root',
                '/store/data/Commissioning10/MinimumBias/RECO/v7/000/132/440/E27821C3-0C3C-DF11-9BD9-0030487CD718.root'
                ]);                                                                                               

            secFiles.extend([
                ])

        elif runNumber == 'Commissioning10-Apr1Skim_Muon_skim-v1' :
            dataset = '/MinimumBias/Commissioning10-Apr1Skim_Muon_skim-v1/RAW-RECO'
            print '   Running on dataset:', dataset, 'run', runNumber, 'with global tag ', useGlobalTag 
    
            readFiles.extend( [
                '/store/data/Commissioning10/MinimumBias/RAW-RECO/Apr1Skim_Muon_skim-v1/0140/E0740811-8E40-DF11-AA5E-0026189438ED.root',
                '/store/data/Commissioning10/MinimumBias/RAW-RECO/Apr1Skim_Muon_skim-v1/0139/EE8A4C75-C83E-DF11-9116-002618943983.root',
                '/store/data/Commissioning10/MinimumBias/RAW-RECO/Apr1Skim_Muon_skim-v1/0139/E8557E88-B43E-DF11-9882-00248C0BE005.root',
                '/store/data/Commissioning10/MinimumBias/RAW-RECO/Apr1Skim_Muon_skim-v1/0139/E618AD7F-C83E-DF11-B8BE-002618943915.root'               
                ]);                                                                                               

            secFiles.extend([
                ])

        elif runNumber == 'MinimumBias_Commissioning10-May13thReReco_preproduction-v1_RECO' :
            dataset = '/MinimumBias/Commissioning10-May13thReReco_preproduction-v1/RECO'
            print '   Running on dataset:', dataset, 'run', runNumber, 'with global tag ', useGlobalTag 
    
            readFiles.extend( [
                '/store/data/Commissioning10/MinimumBias/RECO/May13thReReco_preproduction-v1/0141/A07EA5F3-845F-DF11-870E-00261894388A.root',
                '/store/data/Commissioning10/MinimumBias/RECO/May13thReReco_preproduction-v1/0141/208DE4F3-845F-DF11-8CD2-00261894390B.root',
                '/store/data/Commissioning10/MinimumBias/RECO/May13thReReco_preproduction-v1/0139/FEDE3C23-145F-DF11-915A-0018F3D0965E.root',
                '/store/data/Commissioning10/MinimumBias/RECO/May13thReReco_preproduction-v1/0139/FABF11BB-0B5F-DF11-8CF9-00248C0BE013.root'                
                ]);                                                                                               

            secFiles.extend([
                ])
        else :
            print 'Error: run', runNumber, 'has no RECO file available.'    
            errorUserOptions = True
            

    elif dataType == 'FileStream' : 
        # data dat
        readFiles.extend( [
                'file:/lookarea_SM/MWGR_29.00105760.0001.A.storageManager.00.0000.dat'
        
            ] );


    else : 
        print 'Error:', dataType, 'data for',  runNumber, 'not defined.'    
        errorUserOptions = True
        
            
else :
    readFiles.extend( [                        
        'file:/afs/cern.ch/user/g/ghete/scratch0/CmsswTestFiles/LocalFile_source.root'
        ]);                                                                                               

    secFiles.extend([
        ])

    print 'Local file(s)', readFiles
        