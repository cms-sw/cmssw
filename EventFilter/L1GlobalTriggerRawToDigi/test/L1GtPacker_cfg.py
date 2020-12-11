from __future__ import print_function
#
# cfg file to pack a GT DAQ record
#
# V M Ghete 2009-04-06

import FWCore.ParameterSet.Config as cms

# process
process = cms.Process('TestL1GtPacker')

###################### user choices ######################
# choose (pre)release
useRelease = 'CMSSW_3_1_0'
#useRelease = 'CMSSW_2_2_12'

# choose the type of sample used (True for RelVal, False for data)
useRelValSample = True 
#useRelValSample=False 

if useRelValSample == True :
    
    # 2_2_X
    #useGlobalTag = 'IDEAL_V12'
    #useGlobalTag='STARTUP_V11'
    
    # >= 3_1_0_pre11
    useGlobalTag = 'MC_31X_V1'
    #useGlobalTag = 'STARTUP31X_V1'

    # RelVals 
    #useSample = 'RelValQCD_Pt_80_120'
    useSample = 'RelValTTbar'
    #useSample = 'RelValZTT'
    
else :
    # < 3_1_0_pre11
    #useGlobalTag = 'CRAFT_ALL_V11'

    # >= 3_1_0_pre11
    useGlobalTag = 'CRAFT0831X_V1'
    #useGlobalTag = 'GR09_31X_V1P'
    #useGlobalTag = 'GR09_31X_V1H'
    

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
        
        if (useRelease == 'CMSSW_3_1_0') and (useSample == 'RelValQCD_Pt_80_120') :
        
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

        elif (useRelease == 'CMSSW_2_2_12') and (useSample == 'RelValQCD_Pt_80_120') :
            
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


        else :
            print('Error: no files for sample ', useSample, ', (pre)release ', useRelease, ' and global tag ', useGlobalTag, ' defined.')    
            sys.exit()

        secFiles.extend([
            ])

    elif useGlobalTag.count('STARTUP') :

        if (useRelease == 'CMSSW_3_1_0') and (useSample == 'RelValQCD_Pt_80_120') :
        
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

                    
        elif (useRelease == 'CMSSW_2_2_12') and (useSample == 'RelValQCD_Pt_80_120') :

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


        else :
            print('Error: no files for sample ', useSample, ', (pre)release ', useRelease, ' and global tag ', useGlobalTag, ' defined.')    
            sys.exit()

                
        secFiles.extend([
            ])
        
    else :
        print('Error: Global Tag ', useGlobalTag, ' not defined.')    
        sys.exit()

else : 

    # data
    dataset = '/Cosmics/Commissioning09-v1/RAW'
    print('   Running on set: '+ dataset)    
    
    readFiles.extend( [
       '/store/data/Commissioning09/Cosmics/RAW/v1/000/076/966/00BD9A1F-B908-DE11-8B2C-000423D94A04.root',
       '/store/data/Commissioning09/Cosmics/RAW/v1/000/076/966/025E8B48-B608-DE11-A0EE-00161757BF42.root',
       '/store/data/Commissioning09/Cosmics/RAW/v1/000/076/966/027AA271-D208-DE11-9A7F-001617DBD5AC.root',
       '/store/data/Commissioning09/Cosmics/RAW/v1/000/076/966/04281D2F-D108-DE11-9A27-000423D944DC.root',
       '/store/data/Commissioning09/Cosmics/RAW/v1/000/076/966/065B0C1C-C008-DE11-A32B-001617E30F48.root',
       '/store/data/Commissioning09/Cosmics/RAW/v1/000/076/966/08B1054B-BD08-DE11-AF8B-001617C3B78C.root',
       '/store/data/Commissioning09/Cosmics/RAW/v1/000/076/966/0C055C33-D108-DE11-B678-001617C3B73A.root',
       '/store/data/Commissioning09/Cosmics/RAW/v1/000/076/966/0E480977-D208-DE11-BA78-001617C3B6E2.root',
       '/store/data/Commissioning09/Cosmics/RAW/v1/000/076/966/0E79251B-B908-DE11-83FF-000423D99CEE.root',
       '/store/data/Commissioning09/Cosmics/RAW/v1/000/076/966/101B8CA0-B508-DE11-B614-000423D99160.root',
       '/store/data/Commissioning09/Cosmics/RAW/v1/000/076/966/12C62C71-BF08-DE11-A48C-000423D99614.root',
       '/store/data/Commissioning09/Cosmics/RAW/v1/000/076/966/16A77E08-B008-DE11-9121-000423D8F63C.root'
        ]);

    secFiles.extend([
        ])

if useLocalFiles :
    readFiles = 'file:/afs/cern.ch/user/g/ghete/scratch0/CmsswTestFiles/testGt_L1GtPacker_source.root'


# load and configure modules via Global Tag
# https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideFrontierConditions

process.load('Configuration.StandardSequences.Geometry_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.GlobalTag.globaltag = useGlobalTag+'::All'
#process.load('L1Trigger.Configuration.L1Trigger_FakeConditions_cff')

# L1 GT/GMT packer
process.load("EventFilter.L1GlobalTriggerRawToDigi.l1GtPack_cfi")

# input tag for GT readout collection: 
# input tag for GMT readout collection: 
#     source        = hardware record

if useRelValSample == True :
    daqGtInputTag = 'simGtDigis'
    muGmtInputTag = 'simGmtDigis'
else :
    daqGtInputTag = 'l1GtUnpack'
    muGmtInputTag = 'l1GtUnpack'

process.l1GtPack.DaqGtInputTag = daqGtInputTag
process.l1GtPack.MuGmtInputTag = muGmtInputTag

# mask for active boards (actually 16 bits)
#      if bit is zero, the corresponding board will not be packed
#      default: no board masked: ActiveBoardsMask = 0xFFFF

# no board masked (default)
#process.l1GtPack.ActiveBoardsMask = 0xFFFF
    
# GTFE only in the record
#process.l1GtPack.ActiveBoardsMask = 0x0000

# GTFE + FDL 
#process.l1GtPack.ActiveBoardsMask = 0x0001
     
# GTFE + GMT 
#process.l1GtPack.ActiveBoardsMask = 0x0100

# GTFE + FDL + GMT 
#process.l1GtPack.ActiveBoardsMask = 0x0101

# set it to verbose
process.l1GtPack.Verbosity = cms.untracked.int32(1)

# path to be run

if useRelValSample == True :
    process.p = cms.Path(process.l1GtPack)
else :
    process.p = cms.Path(process.l1GtPack) # FIXME unpack first raw data

# Message Logger
process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.debugModules = ['l1GtPack']
process.MessageLogger.cerr.enable = False
process.MessageLogger.files.L1GtPacker = cms.untracked.PSet(
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

process.outputL1GtPack = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('L1GtPacker.root'),
    # keep only unpacked data in the ROOT file
    outputCommands = cms.untracked.vstring('drop *', 
        'keep *_l1GtPack_*_*')
)

process.outpath = cms.EndPath(process.outputL1GtPack)
