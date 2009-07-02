#
# cfg file to run on L1 GT output file, with GCT and GMT EDM products included
#

import FWCore.ParameterSet.Config as cms

# process
process = cms.Process('TestHLTLevel1GTSeed')

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
    
    # >= 3_1_0
    #useGlobalTag = 'MC_31X_V1'
    useGlobalTag = 'STARTUP31X_V1'

    # RelVals 
    useSample = 'RelValQCD_Pt_80_120'
    
else :
    # < 2_2_X
    useGlobalTag = 'CRAFT_ALL_V11'

    # >= 3_1_0
    #useGlobalTag = 'CRAFT0831X_V1'
    #useGlobalTag = 'GR09_31X_V1P'
    #useGlobalTag = 'GR09_31X_V1H'

# explicit choice of the L1 menu. 
# Use:

#    default menu from Global Tag: put l1Menu = ''
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

        else :
            print 'Error: no files for sample ', useSample, ', (pre)release ', useRelease, ' and global tag ', useGlobalTag, ' defined.'    
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
            print 'Error: no files for sample ', useSample, ', (pre)release ', useRelease, ' and global tag ', useGlobalTag, ' defined.'    
            sys.exit()

                
        secFiles.extend([
            ])
        
    else :
        print 'Error: Global Tag ', useGlobalTag, ' not defined.'    
        sys.exit()

else : 

    # CRAFT data FIXME
    dataset = ''
    
    readFiles.extend([
        ]);

    secFiles.extend([
        ])


if useLocalFiles :
    readFiles = 'file:/afs/cern.ch/user/g/ghete/scratch0/CmsswTestFiles/test_HLTLevel1GTSeed_source.root'


# load and configure modules via Global Tag
# https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideFrontierConditions

process.load('Configuration.StandardSequences.Geometry_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.GlobalTag.globaltag = useGlobalTag+'::All'


# explicit choice of the L1 menu, overwriting the Global Tag menu

if l1Menu != '' :
    process.load('L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMenuConfig_cff')
    process.es_prefer_l1GtParameters = cms.ESPrefer('L1GtTriggerMenuXmlProducer','l1GtTriggerMenuXml')
    
    if useGlobalTag.count('IDEAL') or useGlobalTag.count('MC') :
        if l1Menu == 'L1Menu_MC2009_v2' :
            process.load('L1TriggerConfig.L1GtConfigProducers.Luminosity.lumi1031.L1Menu_MC2009_v2_L1T_Scales_20080922_Imp0_Unprescaled_cff')

        elif l1Menu == 'L1Menu_2008MC_2E30' :
            process.load("L1TriggerConfig.L1GtConfigProducers.Luminosity.lumi1030.L1Menu_2008MC_2E30_Unprescaled_cff")
        else :
            print 'No such L1 menu: ', l1Menu, ' compatible with Global Tag ', useGlobalTag 
        
    elif useGlobalTag.count('STARTUP') :
        if l1Menu == 'L1Menu_Commissioning2009_v0' :       
            process.load('L1TriggerConfig.L1GtConfigProducers.Luminosity.startup.L1Menu_Commissioning2009_v0_L1T_Scales_20080926_startup_Imp0_Unprescaled_cff')
        elif l1Menu == 'myMenu' :
            #process.load("L1TriggerConfig.L1GtConfigProducers.Luminosity.lumi1030.L1Menu_2008MC_2E30_Unprescaled_cff")
            #process.load("L1TriggerConfig.L1GtConfigProducers.Luminosity.startup.L1Menu_startup_v3_Unprescaled_cff")
            #process.load("L1TriggerConfig.L1GtConfigProducers.Luminosity.startup.L1Menu_startup_v4_Unprescaled_cff")
            #process.load("L1TriggerConfig.L1GtConfigProducers.Luminosity.startup.L1Menu_startup_v5_Unprescaled_cff")
            #process.load("L1TriggerConfig.L1GtConfigProducers.Luminosity.startup.L1Menu_startup2_v1_Unprescaled_cff")
            #process.load("L1TriggerConfig.L1GtConfigProducers.Luminosity.startup.L1Menu_startup2_v2_Unprescaled_cff")
            #process.load("L1TriggerConfig.L1GtConfigProducers.Luminosity.startup.L1Menu_startup2_v3_Unprescaled_cff")
            process.load("L1TriggerConfig.L1GtConfigProducers.Luminosity.startup.L1Menu_startup2_v4_Unprescaled_cff")
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
      

# Global Trigger emulator to produce the trigger object maps

import L1Trigger.GlobalTrigger.gtDigis_cfi
process.hltL1GtObjectMap = L1Trigger.GlobalTrigger.gtDigis_cfi.gtDigis.clone()


# input tag for GMT readout collection: 
#     gmtDigis = GMT emulator (default)
#     l1GtUnpack     = GT unpacker (common GT/GMT unpacker)
process.hltL1GtObjectMap.GmtInputTag = 'hltGtDigis'

# input tag for GCT readout collections: 
#     gctDigis = GCT emulator (default) 
process.hltL1GtObjectMap.GctInputTag = 'hltGctDigis'

# input tag for CASTOR record 
#     castorL1Digis =  CASTOR
#process.hltL1GtObjectMap.CastorInputTag = cms.InputTag("castorL1Digis")
    
# technical triggers: a vector of input tags, one tag per each technical 
# trigger producer 
# 
# by default: empty vector
    
# Example:
# TechnicalTriggersInputTags = cms.VInputTag(cms.InputTag('aTechTrigDigis'), 
#                                            cms.InputTag('anotherTechTriggerDigis')),
process.hltL1GtObjectMap.TechnicalTriggersInputTags = cms.VInputTag(cms.InputTag('bscTrigger'))

# logical flag to produce the L1 GT DAQ readout record
#     if true, produce the record (default)
process.hltL1GtObjectMap.ProduceL1GtDaqRecord = False
    
# logical flag to produce the L1 GT EVM readout record
#     if true, produce the record (default)
process.hltL1GtObjectMap.ProduceL1GtEvmRecord = False

# logical flag to produce the L1 GT object map record
#     if true, produce the record (default)
#process.hltL1GtObjectMap.ProduceL1GtObjectMapRecord = False

# logical flag to write the PSB content in the  L1 GT DAQ record
#     if true, write the PSB content in the record (default)
process.hltL1GtObjectMap.WritePsbL1GtDaqRecord = False

# logical flag to read the technical trigger records
#     if true, it will read via getMany the available records (default)
process.hltL1GtObjectMap.ReadTechnicalTriggerRecords = True

# number of "bunch crossing in the event" (BxInEvent) to be emulated
# symmetric around L1Accept (BxInEvent = 0):
#    1 (BxInEvent = 0); 3 (F 0 1) (standard record); 5 (E F 0 1 2) (debug record)
# even numbers (except 0) "rounded" to the nearest lower odd number
# negative value: emulate TotalBxInEvent as given in EventSetup  
process.hltL1GtObjectMap.EmulateBxInEvent = 1

# length of BST record (in bytes) from parameter set
# negative value: take the value from EventSetup      
process.hltL1GtObjectMap.BstLengthBytes = -1

# L1 Extra
process.load('L1Trigger.Configuration.L1Extra_cff')

# replacing arguments for L1Extra
process.l1extraParticles.muonSource = cms.InputTag('hltGtDigis')
process.l1extraParticles.isolatedEmSource = cms.InputTag('hltGctDigis', 'isoEm')
process.l1extraParticles.nonIsolatedEmSource = cms.InputTag('hltGctDigis', 'nonIsoEm')
process.l1extraParticles.centralJetSource = cms.InputTag('hltGctDigis', 'cenJets')
process.l1extraParticles.forwardJetSource = cms.InputTag('hltGctDigis', 'forJets')
process.l1extraParticles.tauJetSource = cms.InputTag('hltGctDigis', 'tauJets')
process.l1extraParticles.etTotalSource = cms.InputTag('hltGctDigis')
process.l1extraParticles.etHadSource = cms.InputTag('hltGctDigis')
process.l1extraParticles.etMissSource = cms.InputTag('hltGctDigis')
process.l1extraParticles.htMissSource = cms.InputTag("hltGctDigis")
process.l1extraParticles.hfRingEtSumsSource = cms.InputTag("hltGctDigis")
process.l1extraParticles.hfRingBitCountsSource = cms.InputTag("hltGctDigis")
process.l1extraParticles.ignoreHtMiss = cms.bool(True)

# this module
process.load('HLTrigger.HLTfilters.hltLevel1GTSeed_cfi')
 
# replacing arguments for hltLevel1GTSeed

# seeding done via technical trigger bits, if value is 'True';
# default: false (seeding via physics algorithms)
#process.hltLevel1GTSeed.L1TechTriggerSeeding = True

# seeding done with aliases for physics algorithms
#process.hltLevel1GTSeed.L1UseAliasesForSeeding = cms.bool(False)

# logical expression for the required L1 algorithms;
# the algorithms are specified by name
# allowed operators: 'AND', 'OR', 'NOT', '(', ')'
#
# by convention, 'L1GlobalDecision' logical expression means global decision
# 

logExpressionNumber = 2

if logExpressionNumber == 0 :
    process.hltLevel1GTSeed.L1SeedsLogicalExpression = \
        'L1_SingleJet50'                                     # 0
elif logExpressionNumber == 1 :        
    process.hltLevel1GTSeed.L1SeedsLogicalExpression = \
        'L1_SingleJet50U'                                    # 1
elif logExpressionNumber == 2 :        
    process.hltLevel1GTSeed.L1SeedsLogicalExpression = \
        'L1_HTT200 OR L1_ETM20 OR L1_ETT60 OR L1_SingleEG15' # 2
elif logExpressionNumber == 3 :        
    process.hltLevel1GTSeed.L1SeedsLogicalExpression = \
        'L1_HTT250 OR L1_HTT300 OR L1_SingleEG15'            # 3
elif logExpressionNumber == 4 :        
    process.hltLevel1GTSeed.L1SeedsLogicalExpression = \
        'NOT L1_SingleEG15'                                  # 4
elif logExpressionNumber == 5 :        
    process.hltLevel1GTSeed.L1SeedsLogicalExpression = \
        'L1_ZeroBias'                                        # 5
elif logExpressionNumber == 6 :        
    process.hltLevel1GTSeed.L1SeedsLogicalExpression = \
        'L1_SingleEG15 AND L1_HTT300'                        # 6
elif logExpressionNumber == 7 :        
    process.hltLevel1GTSeed.L1SeedsLogicalExpression = \
        'L1_SingleEG15 AND (L1_HTT300 AND NOT L1_SingleMu7)' # 7
elif logExpressionNumber == 8 :        
    process.hltLevel1GTSeed.L1SeedsLogicalExpression = \
        '(L1_SingleEG15 OR L1_QuadJet40) AND (L1_HTT300 AND NOT L1_SingleMu7)'  # 8
elif logExpressionNumber == 9 :        
    process.hltLevel1GTSeed.L1SeedsLogicalExpression = \
        '(L1_SingleEG15 OR L1_QuadJet40) AND ((L1_HTT300 AND NOT L1_SingleMu7) OR NOT L1_SingleMu20)' # 9
elif logExpressionNumber == 10 :        
    process.hltLevel1GTSeed.L1SeedsLogicalExpression = \
        'L1_SingleJet6U'                                     # 10
elif logExpressionNumber == 11 :        
    process.hltLevel1GTSeed.L1SeedsLogicalExpression = \
        'L1_SingleJet6U'                                     # 11
elif logExpressionNumber == 12 :        
    process.hltLevel1GTSeed.L1SeedsLogicalExpression = \
        'L1_IsoEG10_Jet6U_ForJet6U'                          # 12
elif logExpressionNumber == 13 :        
    # for technical triggers, one specifies by bit number        
    process.hltLevel1GTSeed.L1SeedsLogicalExpression = \
        '1 AND 15 AND NOT (29 OR 55)'                        # 13
elif logExpressionNumber == 14 :
    process.hltLevel1GTSeed.L1SeedsLogicalExpression = \
        '(L1_SingleEG15 OR L1_QuadJet6U) AND ((L1_HTT200 AND NOT L1_SingleMu7) OR NOT L1_SingleMu20)' # 14
else :
    print 'Error: no logical expression defined'    

    
# InputTag for the L1 Global Trigger DAQ readout record
#   GT Emulator = gtDigis
#   GT Unpacker = l1GtUnpack
#
#   cloned GT unpacker in HLT = gtDigis
#
process.hltLevel1GTSeed.L1GtReadoutRecordTag = cms.InputTag('hltGtDigis')
    
# InputTag for L1 Global Trigger object maps
#   only the emulator produces the object maps
#   GT Emulator = gtDigis
#
#   cloned GT emulator in HLT = l1GtObjectMap
#
process.hltLevel1GTSeed.L1GtObjectMapTag = cms.InputTag('hltL1GtObjectMap')
    
# InputTag for L1 particle collections (except muon)
#   L1 Extra = l1extraParticles
#
#process.hltLevel1GTSeed.L1CollectionsTag = cms.InputTag('l1extraParticles')
    
# InputTag for L1 muon collection
#process.hltLevel1GTSeed.L1MuonCollectionTag = cms.InputTag('l1extraParticles')
    
# saveTags for AOD book-keeping
#process.hltLevel1GTSeed.saveTags = cms.untracked.bool(True)


# path to be run
process.p = cms.Path(process.hltL1GtObjectMap * process.L1Extra * process.hltLevel1GTSeed)

# services

# Message Logger
process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.debugModules = ['hltLevel1GTSeed']
process.MessageLogger.categories.append('HLTLevel1GTSeed')
#process.MessageLogger.destinations = ['cerr']

process.MessageLogger.cerr.threshold = 'DEBUG'
#process.MessageLogger.cerr.threshold = 'INFO'
#process.MessageLogger.cerr.threshold = 'WARNING'
#process.MessageLogger.cerr.threshold = 'ERROR'

process.MessageLogger.cerr.DEBUG = cms.untracked.PSet( limit = cms.untracked.int32(0) )
process.MessageLogger.cerr.INFO = cms.untracked.PSet( limit = cms.untracked.int32(0) )
process.MessageLogger.cerr.WARNING = cms.untracked.PSet( limit = cms.untracked.int32(0) )
process.MessageLogger.cerr.ERROR = cms.untracked.PSet( limit = cms.untracked.int32(0) )

process.MessageLogger.cerr.HLTLevel1GTSeed = cms.untracked.PSet( limit = cms.untracked.int32(-1) )


