from __future__ import print_function
#
# cfg file to run on L1 GT output file, with GCT and GMT EDM products included
#

import FWCore.ParameterSet.Config as cms

# process
process = cms.Process('TestHLTLevel1GTSeed')

###################### user choices ######################
# choose (pre)release
useRelease = 'CMSSW_3_3_2'
#useRelease = 'CMSSW_2_2_12'

# choose the type of sample used (True for RelVal, False for data)
useRelValSample = True 
#useRelValSample=False 

if useRelValSample == True :
    
    # 2_2_X
    #useGlobalTag = 'IDEAL_V12'
    #useGlobalTag='STARTUP_V11'
    
    # >= 3_3_2
    #useGlobalTag = 'MC_31X_V9'
    useGlobalTag = 'STARTUP31X_V8'

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
    input=cms.untracked.int32(50)
)

readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring() 
process.source = cms.Source ('PoolSource', fileNames=readFiles, secondaryFileNames=secFiles)

# type of sample used (True for RelVal, False for data)

if useRelValSample == True :
    if useGlobalTag.count('IDEAL') or useGlobalTag.count('MC') :
        
        if (useRelease == 'CMSSW_3_3_2') and (useSample == 'RelValQCD_Pt_80_120') :
        
            dataset = cms.untracked.vstring('/RelValQCD_Pt_80_120/CMSSW_3_3_1-MC_31X_V9-v3/GEN-SIM-DIGI-RAW-HLTDEBUG')
            readFiles.extend( [
                '/store/relval/CMSSW_3_3_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v3/0003/56AFD3DD-9CC1-DE11-8FFD-0030487A18A4.root',
                '/store/relval/CMSSW_3_3_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v3/0002/E88CFADA-91C0-DE11-96DF-001617C3B77C.root',
                '/store/relval/CMSSW_3_3_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v3/0002/D8E1ADED-8BC0-DE11-82D5-003048D2BE12.root',
                '/store/relval/CMSSW_3_3_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v3/0002/CA13F138-8EC0-DE11-9F23-000423D9863C.root',
                '/store/relval/CMSSW_3_3_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v3/0002/B6C534D3-8FC0-DE11-9F50-0030487D1BCC.root',
                '/store/relval/CMSSW_3_3_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v3/0002/A462E332-8DC0-DE11-BB9B-003048D37514.root',
                '/store/relval/CMSSW_3_3_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v3/0002/9020564B-94C0-DE11-969E-003048D2C108.root',
                '/store/relval/CMSSW_3_3_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v3/0002/86997D62-8CC0-DE11-A87A-003048D375AA.root',
                '/store/relval/CMSSW_3_3_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v3/0002/563FAA76-8AC0-DE11-A5ED-000423D99614.root',
                '/store/relval/CMSSW_3_3_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v3/0002/4C14A1F9-90C0-DE11-899A-000423D94A20.root',
                '/store/relval/CMSSW_3_3_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v3/0002/3A0B5458-90C0-DE11-BF4A-000423DD2F34.root',
                '/store/relval/CMSSW_3_3_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v3/0002/38DDDF74-8BC0-DE11-BB04-000423D98DD4.root',
                '/store/relval/CMSSW_3_3_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v3/0002/22829BDB-8EC0-DE11-8096-001617C3B70E.root',
                '/store/relval/CMSSW_3_3_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v3/0002/1A002063-8FC0-DE11-B210-0030487D1BCC.root',
                '/store/relval/CMSSW_3_3_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v3/0002/1675A79E-92C0-DE11-A828-003048D2C0F0.root',
                '/store/relval/CMSSW_3_3_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v3/0002/0A6FE359-8AC0-DE11-971E-000423D98BC4.root',
                '/store/relval/CMSSW_3_3_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v3/0002/067C1188-89C0-DE11-8FDF-001617E30CC8.root' 
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
            print('Error: no files for sample ', useSample, ', (pre)release ', useRelease, ' and global tag ', useGlobalTag, ' defined.')    
            sys.exit()

        secFiles.extend([
            ])

    elif useGlobalTag.count('STARTUP') :

        if (useRelease == 'CMSSW_3_3_2') and (useSample == 'RelValQCD_Pt_80_120') :
        
            dataset = cms.untracked.vstring('/RelValQCD_Pt_80_120/CMSSW_3_3_1-STARTUP31X_V8-v3/GEN-SIM-DIGI-RAW-HLTDEBUG')
            readFiles.extend( [
                '/store/relval/CMSSW_3_3_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V8-v3/0003/9014D885-9CC1-DE11-91C7-0030487A3232.root',
                '/store/relval/CMSSW_3_3_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V8-v3/0002/F2E3DBC2-67C0-DE11-BCF6-000423D33970.root',
                '/store/relval/CMSSW_3_3_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V8-v3/0002/E6FC5D14-6EC0-DE11-9263-000423D99394.root',
                '/store/relval/CMSSW_3_3_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V8-v3/0002/E6871FE8-66C0-DE11-8C41-000423D6BA18.root',
                '/store/relval/CMSSW_3_3_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V8-v3/0002/D8D07B2B-64C0-DE11-BBE9-003048D37580.root',
                '/store/relval/CMSSW_3_3_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V8-v3/0002/A6BDBA47-67C0-DE11-9647-001617DBD230.root',
                '/store/relval/CMSSW_3_3_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V8-v3/0002/A4A7B978-63C0-DE11-A83C-001617E30F48.root',
                '/store/relval/CMSSW_3_3_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V8-v3/0002/A0155A61-66C0-DE11-885D-0030486733D8.root',
                '/store/relval/CMSSW_3_3_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V8-v3/0002/90204993-68C0-DE11-992A-000423D98800.root',
                '/store/relval/CMSSW_3_3_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V8-v3/0002/8AEF6F45-62C0-DE11-82A8-000423D99A8E.root',
                '/store/relval/CMSSW_3_3_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V8-v3/0002/86EEBC99-61C0-DE11-A997-0030486780B8.root',
                '/store/relval/CMSSW_3_3_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V8-v3/0002/842D2780-60C0-DE11-A1A0-000423D985B0.root',
                '/store/relval/CMSSW_3_3_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V8-v3/0002/747C8377-64C0-DE11-AED5-003048D374F2.root',
                '/store/relval/CMSSW_3_3_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V8-v3/0002/702C5A3A-65C0-DE11-AAC8-000423D98950.root',
                '/store/relval/CMSSW_3_3_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V8-v3/0002/30E041B9-62C0-DE11-80A6-000423D6CA42.root',
                '/store/relval/CMSSW_3_3_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V8-v3/0002/2EB6FA51-6AC0-DE11-B374-003048D373AE.root',
                '/store/relval/CMSSW_3_3_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V8-v3/0002/00757FB9-65C0-DE11-B46A-001617DBCF6A.root'
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
            print('No such L1 menu: ', l1Menu, ' compatible with Global Tag ', useGlobalTag) 
        
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
            print('No such L1 menu: ', l1Menu, ' compatible with Global Tag ', useGlobalTag) 
            
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
            print('No such L1 menu: ', l1Menu, ' compatible with Global Tag ', useGlobalTag) 
    else :
        print('Error: Global Tag ', useGlobalTag, ' not defined.')    
        sys.exit()

else :
    print('   Using default L1 trigger menu from Global Tag ', useGlobalTag)    
      

# Global Trigger emulator to produce the trigger object maps

import L1Trigger.GlobalTrigger.gtDigis_cfi
process.hltL1GtObjectMap = L1Trigger.GlobalTrigger.gtDigis_cfi.gtDigis.clone()


# input tag for GMT readout collection: 
#     gmtDigis = GMT emulator (default)
#     l1GtUnpack     = GT unpacker (common GT/GMT unpacker)
process.hltL1GtObjectMap.GmtInputTag = 'gtDigis'

# input tag for GCT readout collections: 
#     gctDigis = GCT emulator (default) 
process.hltL1GtObjectMap.GctInputTag = 'gctDigis'

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
process.l1extraParticles.muonSource = cms.InputTag('gtDigis')
process.l1extraParticles.isolatedEmSource = cms.InputTag('gctDigis', 'isoEm')
process.l1extraParticles.nonIsolatedEmSource = cms.InputTag('gctDigis', 'nonIsoEm')
process.l1extraParticles.centralJetSource = cms.InputTag('gctDigis', 'cenJets')
process.l1extraParticles.forwardJetSource = cms.InputTag('gctDigis', 'forJets')
process.l1extraParticles.tauJetSource = cms.InputTag('gctDigis', 'tauJets')
process.l1extraParticles.etTotalSource = cms.InputTag('gctDigis')
process.l1extraParticles.etHadSource = cms.InputTag('gctDigis')
process.l1extraParticles.etMissSource = cms.InputTag('gctDigis')
process.l1extraParticles.htMissSource = cms.InputTag("gctDigis")
process.l1extraParticles.hfRingEtSumsSource = cms.InputTag("gctDigis")
process.l1extraParticles.hfRingBitCountsSource = cms.InputTag("gctDigis")
process.l1extraParticles.ignoreHtMiss = cms.bool(False)
process.l1extraParticles.centralBxOnly = cms.bool(False)

# this module
process.load('HLTrigger.HLTfilters.hltLevel1GTSeed_cfi')
 
# replacing arguments for hltLevel1GTSeed

# default: true
#    seeding done via L1 trigger object maps, with objects that fired 
#    only objects from the central BxInEvent (L1A) are used
# if false:
#    seeding is done ignoring if a L1 object fired or not, 
#    adding all L1EXtra objects corresponding to the object types 
#    used in all conditions from the algorithms in logical expression 
#    for a given number of BxInEvent
process.hltLevel1GTSeed.L1UseL1TriggerObjectMaps = cms.bool(False)
#
# option used forL1UseL1TriggerObjectMaps = False only
# number of BxInEvent: 1: L1A=0; 3: -1, L1A=0, 1; 5: -2, -1, L1A=0, 1, 2
process.hltLevel1GTSeed.L1NrBxInEvent = cms.int32(3)

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

logExpressionNumber = 7

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
        'L1_SingleEG15 AND (L1_HTT100 OR L1_SingleMu0) OR L1_SingleJet6U' # 7
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
    print('Error: no logical expression defined')    

    
# InputTag for the L1 Global Trigger DAQ readout record
#   GT Emulator = gtDigis
#   GT Unpacker = l1GtUnpack
#
#   cloned GT unpacker in HLT = gtDigis
#
process.hltLevel1GTSeed.L1GtReadoutRecordTag = cms.InputTag('gtDigis')
    
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
    
# saveTagsfor AOD book-keeping
#process.hltLevel1GTSeed.saveTags = cms.bool( True )


# path to be run
process.p = cms.Path(process.hltL1GtObjectMap * process.L1Extra * process.hltLevel1GTSeed)

# services

# Message Logger
process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.debugModules = ['hltLevel1GTSeed']
process.MessageLogger.HLTLevel1GTSeed=dict()

process.MessageLogger.cerr.threshold = 'DEBUG'
#process.MessageLogger.cerr.threshold = 'INFO'
#process.MessageLogger.cerr.threshold = 'WARNING'
#process.MessageLogger.cerr.threshold = 'ERROR'

process.MessageLogger.cerr.DEBUG = cms.untracked.PSet( limit = cms.untracked.int32(0) )
process.MessageLogger.cerr.INFO = cms.untracked.PSet( limit = cms.untracked.int32(0) )
process.MessageLogger.cerr.WARNING = cms.untracked.PSet( limit = cms.untracked.int32(0) )
process.MessageLogger.cerr.ERROR = cms.untracked.PSet( limit = cms.untracked.int32(0) )

process.MessageLogger.cerr.HLTLevel1GTSeed = cms.untracked.PSet( limit = cms.untracked.int32(-1) )


