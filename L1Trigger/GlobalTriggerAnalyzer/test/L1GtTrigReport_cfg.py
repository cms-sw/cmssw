#
# cfg file to run L1GtTrigReport on GT output file containing 
#    the readout record L1GlobalTriggerReadoutRecord
#    or
#    the lite record L1GlobalTriggerRecord
#
# V M Ghete 2009-03-04


import FWCore.ParameterSet.Config as cms
import sys

# process
process = cms.Process('TestL1Gt')

###################### user choices ######################
# choose (pre)release
useRelease = 'CMSSW_3_1_1'
#useRelease = 'CMSSW_3_1_0'
#useRelease = 'CMSSW_2_2_12'

# choose the type of sample used (True for RelVal, False for data)
useRelValSample = True 
#useRelValSample=False 

if useRelValSample == True :
    
    if useRelease == 'CMSSW_2_2_12' :
        
        useGlobalTag = 'IDEAL_V12'
        #useGlobalTag='STARTUP_V11'
    
    elif useRelease == 'CMSSW_3_1_0' :
         useGlobalTag = 'MC_31X_V1'
         #useGlobalTag = 'STARTUP31X_V1'
    
    elif useRelease == 'CMSSW_3_1_1' :
         #useGlobalTag = 'MC_31X_V2'
         useGlobalTag = 'STARTUP31X_V1'

    # RelVals 
    useSample = 'RelValQCD_Pt_80_120'
    #useSample = 'RelValTTbar'
    #useSample = 'RelValZTT'
    
else :
    # global tag
    
    if useRelease == 'CMSSW_2_2_12' :
        useGlobalTag = 'CRAFT_ALL_V11'
        
    elif useRelease == 'CMSSW_3_1_0' :
        #useGlobalTag = 'CRAFT0831X_V1'
        useGlobalTag = 'GR09_31X_V1P'

    elif useRelease == 'CMSSW_3_1_1' :
        #useGlobalTag = 'CRAFT0831X_V1'
        useGlobalTag = 'GR09_31X_V1P'
    
# data type
#dataType = 'RAW'
dataType = 'RECO'

# explicit choice of the L1 menu. 
# Use:

#    default menu from Global Tag: put l1Menu = ''
#      IDEAL_31X:   L1Menu_MC2009_v1
#      STARTUP_31X: L1Menu_Commissioning2009_v1
l1Menu = ''

#    other available choices (must be compatible with the Global Tag)
#l1Menu = 'L1Menu_Commissioning2009_v0'
#l1Menu = 'L1Menu_MC2009_v0'
#l1Menu = 'L1Menu_startup2_v4'
#l1Menu = 'L1Menu_2008MC_2E30'
#l1Menu = 'myMenu'

#    private menu (must edit the corresponding part in the menu list)  
#    must be compatible with the Global Tag
  
#l1Menu = 'myMenu'

# change to True to use local files
#     the type of file should match the choice of useRelValSample and useGlobalTag
useLocalFiles = False 

###################### end user choices ###################


# number of events to be processed and source file
process.maxEvents = cms.untracked.PSet(
    input=cms.untracked.int32(-1)
)

readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring() 
process.source = cms.Source ('PoolSource', fileNames=readFiles, secondaryFileNames=secFiles)

# type of sample used (True for RelVal, False for data)

if useRelValSample == True :
    if useGlobalTag.count('IDEAL') or useGlobalTag.count('MC') :
        
        if (useRelease == 'CMSSW_2_2_12') and (useSample == 'RelValQCD_Pt_80_120') :
            
            dataset = cms.untracked.vstring('/RelValQCD_Pt_80_120/CMSSW_2_2_10_IDEAL_V12_v1/GEN-SIM-DIGI-RAW-HLTDEBUG')
            readFiles.extend( [
                '/store/relval/CMSSW_2_2_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V12_v1/0003/FC798D81-CB3D-DE11-9FBF-001D09F290BF.root',
                '/store/relval/CMSSW_2_2_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V12_v1/0003/F80D1F07-CA3D-DE11-B1ED-001617DBCF6A.root',
                '/store/relval/CMSSW_2_2_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V12_v1/0003/F63FDCEB-CA3D-DE11-A3AD-0019B9F72BFF.root',
                '/store/relval/CMSSW_2_2_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V12_v1/0003/E6B3876B-CD3D-DE11-9E0D-000423D99F1E.root',
                '/store/relval/CMSSW_2_2_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V12_v1/0003/CE04F0D4-CC3D-DE11-8465-001D09F244DE.root',
                '/store/relval/CMSSW_2_2_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V12_v1/0003/C43129D0-CA3D-DE11-BA1C-001D09F2A49C.root',
                '/store/relval/CMSSW_2_2_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V12_v1/0003/B2F78729-CD3D-DE11-9019-001D09F24448.root',
                '/store/relval/CMSSW_2_2_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V12_v1/0003/9CDE8A75-CB3D-DE11-98B0-001D09F2A690.root',
                '/store/relval/CMSSW_2_2_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V12_v1/0003/96F539A1-CB3D-DE11-9B6B-001D09F25217.root',
                '/store/relval/CMSSW_2_2_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V12_v1/0003/94EB24CC-CB3D-DE11-A4D8-0019DB29C614.root',
                '/store/relval/CMSSW_2_2_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V12_v1/0003/94919798-C83D-DE11-B793-001D09F24F65.root',
                '/store/relval/CMSSW_2_2_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V12_v1/0003/90979075-CD3D-DE11-9D71-001D09F28F0C.root',
                '/store/relval/CMSSW_2_2_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V12_v1/0003/6A2DE1D5-CA3D-DE11-B058-001D09F23A84.root',
                '/store/relval/CMSSW_2_2_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V12_v1/0003/5EE96B31-CB3D-DE11-819A-001D09F25442.root',
                '/store/relval/CMSSW_2_2_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V12_v1/0003/2AC7ADD9-C83D-DE11-B4FF-001D09F2447F.root',
                '/store/relval/CMSSW_2_2_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V12_v1/0003/149A729C-CA3D-DE11-AA49-001617C3B5D8.root',
                '/store/relval/CMSSW_2_2_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V12_v1/0003/142EB962-CA3D-DE11-AAFA-001617C3B6C6.root',
                '/store/relval/CMSSW_2_2_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V12_v1/0003/10C82EA9-043E-DE11-B745-001D09F29524.root',
                '/store/relval/CMSSW_2_2_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V12_v1/0003/08A81316-CC3D-DE11-8500-001D09F2546F.root',
                '/store/relval/CMSSW_2_2_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V12_v1/0003/06C016EB-C93D-DE11-9A07-001D09F23C73.root' 
            ] );

        elif (useRelease == 'CMSSW_3_1_0') and (useSample == 'RelValQCD_Pt_80_120') :
        
            dataset = cms.untracked.vstring('/RelValQCD_Pt_80_120/CMSSW_3_1_0-MC_31X_V1-v1/GEN-SIM-DIGI-RAW-HLTDEBUG')
            readFiles.extend( [
                '/store/relval/CMSSW_3_1_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0002/CC0C544D-DF66-DE11-B3F7-0019DB29C620.root',
                '/store/relval/CMSSW_3_1_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0001/F497B45F-6E66-DE11-BD38-000423D174FE.root',
                '/store/relval/CMSSW_3_1_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0001/EAD57B4C-6866-DE11-B1F6-001D09F28C1E.root',
                '/store/relval/CMSSW_3_1_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0001/CEA3E29A-6A66-DE11-AA09-001D09F24DA8.root',
                '/store/relval/CMSSW_3_1_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0001/CE0E1F78-6D66-DE11-BA57-0019B9F72BFF.root',
                '/store/relval/CMSSW_3_1_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0001/C6C7688D-6C66-DE11-9B55-001D09F24EC0.root',
                '/store/relval/CMSSW_3_1_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0001/C41FF427-6E66-DE11-9E69-001D09F28F11.root',
                '/store/relval/CMSSW_3_1_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0001/BA1F3621-6F66-DE11-8301-000423D992A4.root',
                '/store/relval/CMSSW_3_1_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0001/AACEAEAF-6E66-DE11-8D81-001D09F23A6B.root',
                '/store/relval/CMSSW_3_1_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0001/A6A671F1-6D66-DE11-BA00-001D09F25325.root',
                '/store/relval/CMSSW_3_1_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0001/8A5DCC07-6C66-DE11-9E14-001D09F24448.root',
                '/store/relval/CMSSW_3_1_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0001/669667C8-6E66-DE11-A989-001D09F25393.root',
                '/store/relval/CMSSW_3_1_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0001/5EFC7C08-6666-DE11-B41E-0019B9F581C9.root',
                '/store/relval/CMSSW_3_1_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0001/5CF56885-6F66-DE11-AD32-001D09F2543D.root',
                '/store/relval/CMSSW_3_1_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0001/54A57152-6F66-DE11-A36F-001D09F290BF.root',
                '/store/relval/CMSSW_3_1_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0001/2ADE643E-6D66-DE11-BE7A-001D09F24448.root',
                '/store/relval/CMSSW_3_1_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0001/28483F30-6B66-DE11-B045-001617DBD224.root' 
            ] );

        elif (useRelease == 'CMSSW_3_1_1') and (useSample == 'RelValQCD_Pt_80_120') :
        
            dataset = cms.untracked.vstring('/RelValQCD_Pt_80_120/CMSSW_3_1_1-MC_31X_V2-v1/GEN-SIM-DIGI-RAW-HLTDEBUG')
            readFiles.extend( [
                '/store/relval/CMSSW_3_1_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V2-v1/0002/DCE2FBCF-566B-DE11-8C2E-000423D6006E.root',
                '/store/relval/CMSSW_3_1_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V2-v1/0002/DA37C9FE-576B-DE11-BFD5-000423D6006E.root',
                '/store/relval/CMSSW_3_1_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V2-v1/0002/D4B018E9-556B-DE11-A435-000423D944FC.root',
                '/store/relval/CMSSW_3_1_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V2-v1/0002/D4455EA7-586B-DE11-AE21-001D09F2432B.root',
                '/store/relval/CMSSW_3_1_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V2-v1/0002/CA73E974-546B-DE11-883A-000423D99A8E.root',
                '/store/relval/CMSSW_3_1_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V2-v1/0002/C08535A5-576B-DE11-B4B4-000423D944DC.root',
                '/store/relval/CMSSW_3_1_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V2-v1/0002/BC264E3E-596B-DE11-8E9B-000423D6CA02.root',
                '/store/relval/CMSSW_3_1_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V2-v1/0002/A6247F62-586B-DE11-8D44-0019B9F704D6.root',
                '/store/relval/CMSSW_3_1_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V2-v1/0002/A201691B-D76B-DE11-B1E3-001D09F282F5.root',
                '/store/relval/CMSSW_3_1_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V2-v1/0002/926C28FD-586B-DE11-B3F4-000423D98BE8.root',
                '/store/relval/CMSSW_3_1_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V2-v1/0002/86C42EF7-576B-DE11-B199-001D09F29533.root',
                '/store/relval/CMSSW_3_1_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V2-v1/0002/7EAD824B-566B-DE11-8B6C-000423D991D4.root',
                '/store/relval/CMSSW_3_1_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V2-v1/0002/6AF3F3D8-5C6B-DE11-AC76-001D09F24664.root',
                '/store/relval/CMSSW_3_1_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V2-v1/0002/4EE698A8-576B-DE11-98FF-001D09F2438A.root',
                '/store/relval/CMSSW_3_1_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V2-v1/0002/2E03D513-596B-DE11-890D-001D09F24498.root',
                '/store/relval/CMSSW_3_1_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V2-v1/0002/084648F3-576B-DE11-B1CE-000423D94524.root',
                '/store/relval/CMSSW_3_1_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V2-v1/0002/0090D934-576B-DE11-851B-00304879FA4C.root' 
            ] );



        elif (useRelease == 'CMSSW_2_2_12') and (useSample == 'RelValTTbar') :
            dataset = cms.untracked.vstring('RelValTTbar/CMSSW_2_2_10_IDEAL_V12_v1/GEN-SIM-DIGI-RAW-HLTDEBUG')
            readFiles.extend( [
                '/store/relval/CMSSW_2_2_10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V12_v1/0003/FC0E794B-9F3D-DE11-8969-000423D6C8E6.root',
                '/store/relval/CMSSW_2_2_10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V12_v1/0003/DE8F4018-9E3D-DE11-993A-001D09F2423B.root',
                '/store/relval/CMSSW_2_2_10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V12_v1/0003/D88A54EB-9E3D-DE11-95AC-001617DBD230.root',
                '/store/relval/CMSSW_2_2_10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V12_v1/0003/D0D136FB-033E-DE11-A44E-001D09F28D4A.root',
                '/store/relval/CMSSW_2_2_10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V12_v1/0003/8E1C5431-9E3D-DE11-AF4B-001D09F28F11.root',
                '/store/relval/CMSSW_2_2_10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V12_v1/0003/86FC99FF-9D3D-DE11-92AF-001D09F290BF.root',
                '/store/relval/CMSSW_2_2_10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V12_v1/0003/7EAEEC8E-9E3D-DE11-8BC3-001D09F231C9.root',
                '/store/relval/CMSSW_2_2_10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V12_v1/0003/664A1CAD-9F3D-DE11-95D0-001D09F241B9.root',
                '/store/relval/CMSSW_2_2_10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V12_v1/0003/648417C1-9E3D-DE11-A52F-001D09F24682.root',
                '/store/relval/CMSSW_2_2_10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V12_v1/0003/58F61F49-9E3D-DE11-9B27-001D09F2523A.root',
                '/store/relval/CMSSW_2_2_10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V12_v1/0003/58117FD8-9E3D-DE11-8EEC-001617C3B778.root',
                '/store/relval/CMSSW_2_2_10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V12_v1/0003/3A53E076-9E3D-DE11-B98A-001D09F23A84.root',
                '/store/relval/CMSSW_2_2_10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V12_v1/0003/389C28A6-9E3D-DE11-843E-001D09F2447F.root',
                '/store/relval/CMSSW_2_2_10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V12_v1/0003/305A2B75-9E3D-DE11-BFAB-001D09F2423B.root',
                '/store/relval/CMSSW_2_2_10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V12_v1/0002/EE5B4533-933D-DE11-AD30-001D09F24DA8.root',
                '/store/relval/CMSSW_2_2_10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V12_v1/0002/EC88F7D0-9A3D-DE11-9836-001617E30E28.root',
                '/store/relval/CMSSW_2_2_10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V12_v1/0002/E2A462B0-9D3D-DE11-A2B6-001D09F244BB.root',
                '/store/relval/CMSSW_2_2_10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V12_v1/0002/CC9E91FC-933D-DE11-972F-001D09F25109.root',
                '/store/relval/CMSSW_2_2_10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V12_v1/0002/B845EA9A-9B3D-DE11-A9F9-001617C3B6FE.root',
                '/store/relval/CMSSW_2_2_10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V12_v1/0002/B67E5CE0-9D3D-DE11-83F1-001D09F291D2.root',
                '/store/relval/CMSSW_2_2_10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V12_v1/0002/92912A15-9D3D-DE11-B3C4-001D09F24448.root',
                '/store/relval/CMSSW_2_2_10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V12_v1/0002/749492B7-993D-DE11-9FBF-001617E30F50.root',
                '/store/relval/CMSSW_2_2_10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V12_v1/0002/706DA2E3-923D-DE11-97DA-001D09F241B4.root',
                '/store/relval/CMSSW_2_2_10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V12_v1/0002/6CDD71F8-973D-DE11-A993-001D09F297EF.root',
                '/store/relval/CMSSW_2_2_10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V12_v1/0002/6694F56B-9A3D-DE11-95EA-001D09F291D7.root',
                '/store/relval/CMSSW_2_2_10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V12_v1/0002/2255EDC6-9D3D-DE11-A02F-001D09F24D8A.root' 
            ] );

        elif (useRelease == 'CMSSW_3_1_0') and (useSample == 'RelValTTbar') :
            
            dataset = cms.untracked.vstring('/RelValTTbar/CMSSW_3_1_0-MC_31X_V1-v1/GEN-SIM-DIGI-RAW-HLTDEBUG')
            readFiles.extend( [
                '/store/relval/CMSSW_3_1_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0001/F81AA535-C666-DE11-942A-001D09F24600.root',
                '/store/relval/CMSSW_3_1_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0001/F45A3761-C766-DE11-8274-001D09F24FBA.root',
                '/store/relval/CMSSW_3_1_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0001/ECDD6402-C466-DE11-AD8D-000423D99A8E.root',
                '/store/relval/CMSSW_3_1_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0001/D0B6652D-C266-DE11-A7A6-001D09F24600.root',
                '/store/relval/CMSSW_3_1_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0001/CA895E96-DE66-DE11-8768-001D09F248FD.root',
                '/store/relval/CMSSW_3_1_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0001/B4FF6350-C466-DE11-BB33-001D09F24DA8.root',
                '/store/relval/CMSSW_3_1_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0001/A80A52B0-C266-DE11-8A5A-001D09F25041.root',
                '/store/relval/CMSSW_3_1_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0001/A6C8A82A-C266-DE11-8704-001D09F23A6B.root',
                '/store/relval/CMSSW_3_1_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0001/A6350A56-C866-DE11-B573-001D09F24FBA.root',
                '/store/relval/CMSSW_3_1_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0001/A4C58176-C566-DE11-ADC0-001D09F28D4A.root',
                '/store/relval/CMSSW_3_1_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0001/A2C1AC27-C266-DE11-9667-001D09F2983F.root',
                '/store/relval/CMSSW_3_1_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0001/8AE98AF2-C766-DE11-B315-001D09F26C5C.root',
                '/store/relval/CMSSW_3_1_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0001/88F81419-C966-DE11-8481-001D09F24024.root',
                '/store/relval/CMSSW_3_1_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0001/5EC0F22B-C266-DE11-A2DA-001D09F23A61.root',
                '/store/relval/CMSSW_3_1_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0001/4A872C1B-C366-DE11-A844-001D09F25041.root',
                '/store/relval/CMSSW_3_1_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0001/32CED660-C766-DE11-B873-001D09F28F11.root',
                '/store/relval/CMSSW_3_1_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0001/308CF886-C866-DE11-95C7-001D09F28755.root',
                '/store/relval/CMSSW_3_1_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0001/2A4FA6DE-C466-DE11-A598-000423D99E46.root',
                '/store/relval/CMSSW_3_1_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0001/28327168-C666-DE11-9486-000423D99EEE.root',
                '/store/relval/CMSSW_3_1_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0001/24FF0D62-CB66-DE11-8F1F-001D09F24DA8.root',
                '/store/relval/CMSSW_3_1_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0001/14AEBDFE-C666-DE11-AA23-001D09F28755.root' 
            ] );

        elif (useRelease == 'CMSSW_3_1_1') and (useSample == 'RelValTTbar') :
            
            dataset = cms.untracked.vstring('/RelValTTbar/CMSSW_3_1_1-MC_31X_V2-v1/GEN-SIM-DIGI-RAW-HLTDEBUG')
            readFiles.extend( [
                '/store/relval/CMSSW_3_1_1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V2-v1/0002/EA8E5AF7-576B-DE11-BA98-001D09F24498.root',
                '/store/relval/CMSSW_3_1_1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V2-v1/0002/E8627E8B-5A6B-DE11-A8F4-001D09F2438A.root',
                '/store/relval/CMSSW_3_1_1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V2-v1/0002/D66DD273-5C6B-DE11-A8DB-001D09F290CE.root',
                '/store/relval/CMSSW_3_1_1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V2-v1/0002/CC3232F2-596B-DE11-8C47-0019B9F704D6.root',
                '/store/relval/CMSSW_3_1_1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V2-v1/0002/AAFDF230-5C6B-DE11-BF0A-001D09F24498.root',
                '/store/relval/CMSSW_3_1_1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V2-v1/0002/9A28D939-576B-DE11-811D-000423D944F0.root',
                '/store/relval/CMSSW_3_1_1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V2-v1/0002/94F72FC0-5B6B-DE11-8215-000423D6AF24.root',
                '/store/relval/CMSSW_3_1_1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V2-v1/0002/94710927-5B6B-DE11-92EA-001D09F290CE.root',
                '/store/relval/CMSSW_3_1_1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V2-v1/0002/88D87820-5B6B-DE11-B522-0019B9F704D6.root',
                '/store/relval/CMSSW_3_1_1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V2-v1/0002/7E7FB3BC-E16B-DE11-9374-000423D8F63C.root',
                '/store/relval/CMSSW_3_1_1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V2-v1/0002/7640E138-576B-DE11-B907-000423D99AAE.root',
                '/store/relval/CMSSW_3_1_1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V2-v1/0002/6EC3A9F2-596B-DE11-B800-001D09F290CE.root',
                '/store/relval/CMSSW_3_1_1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V2-v1/0002/6A4E7D34-596B-DE11-BC50-000423D986A8.root',
                '/store/relval/CMSSW_3_1_1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V2-v1/0002/6482606F-586B-DE11-A34C-000423D9880C.root',
                '/store/relval/CMSSW_3_1_1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V2-v1/0002/50B56266-5E6B-DE11-9275-001D09F24664.root',
                '/store/relval/CMSSW_3_1_1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V2-v1/0002/44793DC2-656B-DE11-B11D-000423D6CA72.root',
                '/store/relval/CMSSW_3_1_1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V2-v1/0002/404CFED1-5C6B-DE11-875E-000423D6CA02.root',
                '/store/relval/CMSSW_3_1_1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V2-v1/0002/38212970-586B-DE11-AB61-000423D6CA72.root',
                '/store/relval/CMSSW_3_1_1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V2-v1/0002/2CF0A195-566B-DE11-92A5-000423D6B358.root',
                '/store/relval/CMSSW_3_1_1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V2-v1/0002/0CD87279-5C6B-DE11-8711-000423D98BC4.root',
                '/store/relval/CMSSW_3_1_1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V2-v1/0002/0279970F-606B-DE11-89A8-001D09F2438A.root' 
            ] );


        else :
            print 'Error: no files for sample ', useSample, ', (pre)release ', useRelease, ' and global tag ', useGlobalTag, ' defined.'    
            sys.exit()

        secFiles.extend([
            ])

    elif useGlobalTag.count('STARTUP') :

        if (useRelease == 'CMSSW_2_2_12') and (useSample == 'RelValQCD_Pt_80_120') :

            dataset = cms.untracked.vstring('/RelValQCD_Pt_80_120/CMSSW_2_2_10_STARTUP_V11_v1/GEN-SIM-DIGI-RAW-HLTDEBUG')
            readFiles.extend( [
                '/store/relval/CMSSW_2_2_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V11_v1/0003/1825DE62-043E-DE11-9AB7-001617C3B654.root',
                '/store/relval/CMSSW_2_2_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V11_v1/0002/F2520292-863D-DE11-ACDF-001D09F2538E.root',
                '/store/relval/CMSSW_2_2_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V11_v1/0002/F05F275A-893D-DE11-94B8-001D09F25217.root',
                '/store/relval/CMSSW_2_2_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V11_v1/0002/D2855C5D-8A3D-DE11-8084-001D09F24D8A.root',
                '/store/relval/CMSSW_2_2_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V11_v1/0002/AEAE2C43-893D-DE11-B3C0-001D09F24637.root',
                '/store/relval/CMSSW_2_2_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V11_v1/0002/AE91E67A-873D-DE11-8BD6-001D09F297EF.root',
                '/store/relval/CMSSW_2_2_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V11_v1/0002/AC1958A0-883D-DE11-96AE-001D09F29524.root',
                '/store/relval/CMSSW_2_2_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V11_v1/0002/92706ABC-8B3D-DE11-A1E2-001D09F297EF.root',
                '/store/relval/CMSSW_2_2_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V11_v1/0002/8EAE234C-8B3D-DE11-A87A-0019B9F72CC2.root',
                '/store/relval/CMSSW_2_2_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V11_v1/0002/884EBAF1-873D-DE11-B5B7-001D09F291D2.root',
                '/store/relval/CMSSW_2_2_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V11_v1/0002/68D12010-8C3D-DE11-BA50-001617C3B70E.root',
                '/store/relval/CMSSW_2_2_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V11_v1/0002/68615FA2-8A3D-DE11-85AD-001D09F291D7.root',
                '/store/relval/CMSSW_2_2_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V11_v1/0002/5C7DF99A-883D-DE11-BF37-001D09F2A465.root',
                '/store/relval/CMSSW_2_2_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V11_v1/0002/52ED45CE-893D-DE11-A52E-001D09F24D4E.root',
                '/store/relval/CMSSW_2_2_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V11_v1/0002/3AE45D80-863D-DE11-ADAA-00304879FA4A.root',
                '/store/relval/CMSSW_2_2_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V11_v1/0002/164B8DC2-8B3D-DE11-A1BA-001617C3B654.root',
                '/store/relval/CMSSW_2_2_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V11_v1/0002/14396117-8B3D-DE11-9A10-001D09F291D7.root',
                '/store/relval/CMSSW_2_2_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V11_v1/0002/0EB74EF7-863D-DE11-AD5E-001D09F29597.root',
                '/store/relval/CMSSW_2_2_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V11_v1/0002/026314A2-8A3D-DE11-8CB6-001D09F2447F.root',
                '/store/relval/CMSSW_2_2_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V11_v1/0002/00081940-863D-DE11-8158-0019B9F6C674.root' 
            ] );
        
        elif (useRelease == 'CMSSW_3_1_0') and (useSample == 'RelValQCD_Pt_80_120') :
        
            dataset = cms.untracked.vstring('/RelValQCD_Pt_80_120/CMSSW_3_1_0-STARTUP31X_V1-v1/GEN-SIM-DIGI-RAW-HLTDEBUG')
            readFiles.extend( [
                '/store/relval/CMSSW_3_1_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v1/0002/2C209975-E066-DE11-9A95-001D09F27067.root',
                '/store/relval/CMSSW_3_1_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v1/0001/F8353FEB-3366-DE11-BB74-001D09F26509.root',
                '/store/relval/CMSSW_3_1_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v1/0001/F4E28000-3466-DE11-B243-000423D9997E.root',
                '/store/relval/CMSSW_3_1_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v1/0001/DCD49C28-3466-DE11-A772-001D09F251E0.root',
                '/store/relval/CMSSW_3_1_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v1/0001/CC34BA65-3466-DE11-B3EB-001D09F24D8A.root',
                '/store/relval/CMSSW_3_1_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v1/0001/A83F2050-3466-DE11-A56D-001D09F26509.root',
                '/store/relval/CMSSW_3_1_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v1/0001/A6983B60-3466-DE11-A04F-001D09F2983F.root',
                '/store/relval/CMSSW_3_1_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v1/0001/A03ACDE9-3166-DE11-B484-001D09F29619.root',
                '/store/relval/CMSSW_3_1_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v1/0001/9CFED7BA-3166-DE11-B8DA-001D09F2424A.root',
                '/store/relval/CMSSW_3_1_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v1/0001/761BDD0C-3466-DE11-8808-001D09F2983F.root',
                '/store/relval/CMSSW_3_1_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v1/0001/723AC396-3066-DE11-BA9F-001D09F27067.root',
                '/store/relval/CMSSW_3_1_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v1/0001/6CA51004-3466-DE11-BD6C-001D09F28F25.root',
                '/store/relval/CMSSW_3_1_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v1/0001/664957EA-3166-DE11-B7EA-001D09F2915A.root',
                '/store/relval/CMSSW_3_1_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v1/0001/549B80EA-3166-DE11-B467-000423D6CA6E.root',
                '/store/relval/CMSSW_3_1_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v1/0001/4E98BAB9-3166-DE11-9000-001D09F24EC0.root',
                '/store/relval/CMSSW_3_1_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v1/0001/3ED2AF60-3466-DE11-992C-001D09F28F25.root',
                '/store/relval/CMSSW_3_1_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v1/0001/2869A25E-3166-DE11-9634-001D09F28755.root' 
            ] );

        elif (useRelease == 'CMSSW_3_1_1') and (useSample == 'RelValQCD_Pt_80_120') :
        
            dataset = cms.untracked.vstring('/RelValQCD_Pt_80_120/CMSSW_3_1_0-STARTUP31X_V1-v1/GEN-SIM-DIGI-RAW-HLTDEBUG')
            readFiles.extend( [
                '/store/relval/CMSSW_3_1_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v2/0003/005BDE4E-EF6B-DE11-BDE6-0030487C608C.root',
                '/store/relval/CMSSW_3_1_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v2/0002/EA0A3CD8-E16B-DE11-8377-000423D99AA2.root',
                '/store/relval/CMSSW_3_1_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v2/0002/DC8BA8D0-E16B-DE11-9437-001D09F2910A.root',
                '/store/relval/CMSSW_3_1_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v2/0002/C49B5DBF-D76B-DE11-9D89-000423D98DB4.root',
                '/store/relval/CMSSW_3_1_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v2/0002/A02E9EF1-E16B-DE11-BF0D-001D09F2514F.root',
                #'/store/relval/CMSSW_3_1_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v2/0002/96C440DC-E16B-DE11-971A-001D09F291D2.root',
                #'/store/relval/CMSSW_3_1_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v2/0002/86431DDC-E16B-DE11-96A8-0019B9F709A4.root',
                '/store/relval/CMSSW_3_1_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v2/0002/7C7D43BD-D76B-DE11-9C56-001D09F2546F.root',
                '/store/relval/CMSSW_3_1_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v2/0002/7278E3F0-D66B-DE11-BCA0-001D09F2523A.root',
                '/store/relval/CMSSW_3_1_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v2/0002/6ABFA7D8-E16B-DE11-8615-000423D952C0.root',
                '/store/relval/CMSSW_3_1_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v2/0002/66287807-E26B-DE11-B3FA-000423D951D4.root',
                #'/store/relval/CMSSW_3_1_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v2/0002/6060F450-D66B-DE11-80A7-001D09F24353.root',
                '/store/relval/CMSSW_3_1_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v2/0002/5C14D1E0-E16B-DE11-B099-001D09F292D1.root',
                '/store/relval/CMSSW_3_1_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v2/0002/52CEEEAE-D76B-DE11-B7B6-001D09F232B9.root',
                '/store/relval/CMSSW_3_1_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v2/0002/525E3DDA-D76B-DE11-AC59-000423D6BA18.root',
                #'/store/relval/CMSSW_3_1_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v2/0002/486B2FA9-D76B-DE11-81E6-000423D944F8.root',
                '/store/relval/CMSSW_3_1_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v2/0002/32524B43-D86B-DE11-99E6-000423D6CA42.root' 
            ] );

        elif (useRelease == 'CMSSW_3_1_1') and (useSample == 'RelValTTbar') :
        
            dataset = cms.untracked.vstring('/RelValTTbar/CMSSW_3_1_1-STARTUP31X_V1-v2/GEN-SIM-DIGI-RAW-HLTDEBUG')
            readFiles.extend( [
                '/store/relval/CMSSW_3_1_1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v2/0003/FC835340-F06B-DE11-A5CB-001D09F2960F.root',
                '/store/relval/CMSSW_3_1_1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v2/0003/B0ABE28D-F66B-DE11-A986-001D09F29169.root',
                '/store/relval/CMSSW_3_1_1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v2/0002/E067F836-EC6B-DE11-9EAF-001D09F29619.root',
                '/store/relval/CMSSW_3_1_1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v2/0002/DE306118-EA6B-DE11-837A-000423D9989E.root',
                '/store/relval/CMSSW_3_1_1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v2/0002/D86D78A4-EC6B-DE11-AF14-000423D98804.root',
                '/store/relval/CMSSW_3_1_1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v2/0002/D664972B-E86B-DE11-9B4F-0019B9F707D8.root',
                '/store/relval/CMSSW_3_1_1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v2/0002/CC0640C1-E96B-DE11-B898-000423D6A6F4.root',
                '/store/relval/CMSSW_3_1_1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v2/0002/B8EA0110-EB6B-DE11-ADCF-001D09F2546F.root',
                '/store/relval/CMSSW_3_1_1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v2/0002/B44D29A2-EB6B-DE11-8928-000423D98844.root',
                '/store/relval/CMSSW_3_1_1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v2/0002/AE87EC70-EE6B-DE11-B29E-000423D992A4.root',
                '/store/relval/CMSSW_3_1_1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v2/0002/A22F2693-E76B-DE11-AFD4-001D09F24498.root',
                '/store/relval/CMSSW_3_1_1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v2/0002/90E490E8-E86B-DE11-9712-000423D94E70.root',
                '/store/relval/CMSSW_3_1_1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v2/0002/8AFB2138-ED6B-DE11-AC45-000423D6A6F4.root',
                '/store/relval/CMSSW_3_1_1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v2/0002/8AF7EF42-EB6B-DE11-ABD7-001D09F2546F.root',
                '/store/relval/CMSSW_3_1_1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v2/0002/768A640F-EA6B-DE11-976A-000423D98634.root',
                '/store/relval/CMSSW_3_1_1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v2/0002/6402971C-EF6B-DE11-A368-000423D6CAF2.root',
                '/store/relval/CMSSW_3_1_1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v2/0002/5416A5C0-ED6B-DE11-853C-000423D6CAF2.root',
                '/store/relval/CMSSW_3_1_1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v2/0002/3E166228-EE6B-DE11-BB65-000423D6CA72.root',
                '/store/relval/CMSSW_3_1_1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v2/0002/16AFEC4E-EF6B-DE11-A551-00304879FBB2.root',
                '/store/relval/CMSSW_3_1_1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v2/0002/0406D185-E96B-DE11-B8DA-001D09F2924F.root',
                '/store/relval/CMSSW_3_1_1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v2/0002/02041691-ED6B-DE11-A775-001D09F2AF1E.root' 
            ] );

        elif (useRelease == 'CMSSW_3_1_1') and (useSample == 'RelValZTT') :
            
            dataset = cms.untracked.vstring('/RelValZTT/CMSSW_3_1_1-STARTUP31X_V1-v2/GEN-SIM-DIGI-RAW-HLTDEBUG')
            readFiles.extend( [
                '/store/relval/CMSSW_3_1_1/RelValZTT/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v2/0003/F0D19E45-EF6B-DE11-8BAC-001D09F24D67.root',
                '/store/relval/CMSSW_3_1_1/RelValZTT/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v2/0002/FC59127C-E36B-DE11-B0BE-000423D94908.root',
                '/store/relval/CMSSW_3_1_1/RelValZTT/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v2/0002/EA8183D7-E26B-DE11-B707-001D09F28E80.root',
                '/store/relval/CMSSW_3_1_1/RelValZTT/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v2/0002/E83397D7-E26B-DE11-A60B-001D09F292D1.root',
                '/store/relval/CMSSW_3_1_1/RelValZTT/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v2/0002/B812C1E5-E26B-DE11-9866-000423D6B48C.root',
                '/store/relval/CMSSW_3_1_1/RelValZTT/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v2/0002/7A4FD365-E36B-DE11-866E-000423D8F63C.root',
                '/store/relval/CMSSW_3_1_1/RelValZTT/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v2/0002/6852C06E-E36B-DE11-8658-001D09F2B30B.root',
                '/store/relval/CMSSW_3_1_1/RelValZTT/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v2/0002/5EDC216F-E36B-DE11-BAA8-0019B9F730D2.root',
                '/store/relval/CMSSW_3_1_1/RelValZTT/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v2/0002/48AD5E53-E46B-DE11-BDE9-000423D99CEE.root',
                '/store/relval/CMSSW_3_1_1/RelValZTT/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v2/0002/26FDE17F-E36B-DE11-B13B-001D09F2523A.root',
                '/store/relval/CMSSW_3_1_1/RelValZTT/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v2/0002/269A40F2-E36B-DE11-B6F5-000423D8FA38.root',
                '/store/relval/CMSSW_3_1_1/RelValZTT/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v2/0002/22EC6674-E36B-DE11-95A8-0019B9F72D71.root' 
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

    # data 
    dataset = cms.untracked.vstring('/Cosmics/Commissioning09-v3/RAW')
    readFiles.extend( [
            '/store/data/Commissioning09/Cosmics/RAW/v3/000/105/847/6A699BB9-2072-DE11-995B-001D09F34488.root'
        
        ] );

    secFiles.extend([
        ])


if useLocalFiles :
    readFiles = 'file:/afs/cern.ch/user/g/ghete/scratch0/CmsswTestFiles/testGt_L1GtTrigReport_source.root'


# load and configure modules via Global Tag
# https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideFrontierConditions

process.load('Configuration.StandardSequences.Geometry_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.GlobalTag.globaltag = useGlobalTag+'::All'


# explicit choice of the L1 menu, overwriting the Global Tag menu
#      IDEAL_31X:   L1Menu_MC2009_v1
#      STARTUP_31X: L1Menu_Commissioning2009_v1

if l1Menu != '' :
    process.load('L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMenuConfig_cff')
    process.es_prefer_l1GtParameters = cms.ESPrefer('L1GtTriggerMenuXmlProducer','l1GtTriggerMenuXml')
    
    if useGlobalTag.count('IDEAL') or useGlobalTag.count('MC') :
        if l1Menu == 'L1Menu_MC2009_v0' :
            process.load('L1TriggerConfig.L1GtConfigProducers.Luminosity.lumi1031.L1Menu_MC2009_v0_L1T_Scales_20080922_Imp0_Unprescaled_cff')

        elif l1Menu == 'L1Menu_2008MC_2E30' :
            process.load("L1TriggerConfig.L1GtConfigProducers.Luminosity.lumi1030.L1Menu_2008MC_2E30_Unprescaled_cff")
        else :
            print 'No such L1 menu: ', l1Menu, ' compatible with Global Tag ', useGlobalTag 
        
    elif useGlobalTag.count('STARTUP') :
        if l1Menu == 'L1Menu_Commissioning2009_v0' :       
            process.load('L1TriggerConfig.L1GtConfigProducers.Luminosity.startup.L1Menu_Commissioning2009_v0_L1T_Scales_20080926_startup_Imp0_Unprescaled_cff')
        elif l1Menu == 'myMenu' :
            process.load("L1TriggerConfig.L1GtConfigProducers.Luminosity.lumi1030.L1Menu_2008MC_2E30_Unprescaled_cff")
            #process.load("L1TriggerConfig.L1GtConfigProducers.Luminosity.startup.L1Menu_startup_v3_Unprescaled_cff")
            #process.load("L1TriggerConfig.L1GtConfigProducers.Luminosity.startup.L1Menu_startup_v4_Unprescaled_cff")
            #process.load("L1TriggerConfig.L1GtConfigProducers.Luminosity.startup.L1Menu_startup_v5_Unprescaled_cff")
            #process.load("L1TriggerConfig.L1GtConfigProducers.Luminosity.startup.L1Menu_startup2_v1_Unprescaled_cff")
            #process.load("L1TriggerConfig.L1GtConfigProducers.Luminosity.startup.L1Menu_startup2_v2_Unprescaled_cff")
            #process.load("L1TriggerConfig.L1GtConfigProducers.Luminosity.startup.L1Menu_startup2_v3_Unprescaled_cff")
        else :
            print 'No such L1 menu: ', l1Menu, ' compatible with Global Tag ', useGlobalTag 
            
    elif useGlobalTag.count('CRAFT') :
        if l1Menu == 'L1Menu_Commissioning2009_v0' :       
            process.load('L1TriggerConfig.L1GtConfigProducers.Luminosity.startup.L1Menu_Commissioning2009_v0_L1T_Scales_20080926_startup_Imp0_Unprescaled_cff')
        elif l1Menu == 'myMenu' :
            #process.load("L1TriggerConfig.L1GtConfigProducers.Luminosity.startup.L1Menu_startup_v3_Unprescaled_cff")
            #process.load("L1TriggerConfig.L1GtConfigProducers.Luminosity.startup.L1Menu_startup_v4_Unprescaled_cff")
            #process.load("L1TriggerConfig.L1GtConfigProducers.Luminosity.startup.L1Menu_startup_v5_Unprescaled_cff")
            #process.load("L1TriggerConfig.L1GtConfigProducers.Luminosity.startup.L1Menu_startup2_v1_Unprescaled_cff")
            #process.load("L1TriggerConfig.L1GtConfigProducers.Luminosity.startup.L1Menu_startup2_v2_Unprescaled_cff")
            process.load("L1TriggerConfig.L1GtConfigProducers.Luminosity.startup.L1Menu_startup2_v3_Unprescaled_cff")
        else :
            print 'No such L1 menu: ', l1Menu, ' compatible with Global Tag ', useGlobalTag 
    else :
        print 'Error: Global Tag ', useGlobalTag, ' not defined.'    
        sys.exit()

else :
    print '   Using default L1 trigger menu from Global Tag ', useGlobalTag    
      

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
process.l1GtTrigReport.L1GtRecordInputTag = "simGtDigis"

#process.l1GtTrigReport.PrintVerbosity = 2
#process.l1GtTrigReport.PrintOutput = 1

# for RAW data, run first the RAWTODIGI 
if dataType == 'RAW' :
    process.load('Configuration/StandardSequences/RawToDigi_Data_cff')
    process.l1GtTrigReport.L1GtRecordInputTag = "gtDigis"
    process.p = cms.Path(process.RawToDigi+process.l1GtTrigReport)
    
else :        
    # path to be run for RECO
    process.p = cms.Path(process.l1GtTrigReport)


# Message Logger
process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.debugModules = ['l1GtTrigReport']
process.MessageLogger.categories.append('L1GtTrigReport')

#process.MessageLogger.cerr.threshold = 'DEBUG'
process.MessageLogger.cerr.threshold = 'INFO'
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

