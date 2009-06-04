#
# cfg file to run L1GtTrigReport on GT output file containing 
#    the readout record L1GlobalTriggerReadoutRecord
#    or
#    the lite record L1GlobalTriggerRecord
#
# V M Ghete 2009-03-04


import FWCore.ParameterSet.Config as cms

# process
process = cms.Process('TestL1Gt')

###################### user choices ######################

# choose the type of sample used (True for RelVal, False for data)
useRelValSample = True 
#useRelValSample=False 

if useRelValSample == True :
    useGlobalTag = 'IDEAL_31X'
    #useGlobalTag='STARTUP_31X'
    #useGlobalTag = 'IDEAL_V12' # 22X
    #useGlobalTag='STARTUP_V11'
else :
    useGlobalTag = 'CRAFT_ALL_V11'

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
    if useGlobalTag.count('IDEAL') :

        dataset = cms.untracked.vstring('/RelValQCD_Pt_80_120/CMSSW_3_1_0_pre8_IDEAL_31X_v1/GEN-SIM-DIGI-RAW-HLTDEBUG')
        readFiles.extend( [
            '/store/relval/CMSSW_3_1_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0006/20130BFF-DA4D-DE11-B7BC-000423D6C8E6.root',
            '/store/relval/CMSSW_3_1_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0005/F46B2CCE-954D-DE11-A4DC-001D09F2AD7F.root',
            '/store/relval/CMSSW_3_1_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0005/C47AA4AA-A34D-DE11-974E-001D09F241B4.root',
            '/store/relval/CMSSW_3_1_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0005/BCC57AE2-934D-DE11-9D87-001D09F2527B.root',
            '/store/relval/CMSSW_3_1_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0005/A45D3F48-A34D-DE11-BD7C-001D09F25393.root',
            '/store/relval/CMSSW_3_1_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0005/9A2593C5-914D-DE11-A449-001D09F25041.root',
            '/store/relval/CMSSW_3_1_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0005/84DBBB73-984D-DE11-B35C-000423D6CA6E.root',
            '/store/relval/CMSSW_3_1_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0005/7C060FCA-9E4D-DE11-B289-001617E30F50.root',
            '/store/relval/CMSSW_3_1_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0005/6EDD6527-9F4D-DE11-B2FF-001D09F251FE.root',
            '/store/relval/CMSSW_3_1_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0005/6E200660-A14D-DE11-8602-001D09F253FC.root',
            '/store/relval/CMSSW_3_1_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0005/4AC16B83-9B4D-DE11-B992-001D09F251FE.root',
            '/store/relval/CMSSW_3_1_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0005/48B99AC1-A14D-DE11-97E3-001D09F2906A.root',
            '/store/relval/CMSSW_3_1_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0005/3C7B93A8-8E4D-DE11-8AA0-001D09F251B8.root',
            '/store/relval/CMSSW_3_1_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0005/28669EB2-9D4D-DE11-A51E-001D09F291D2.root',
            '/store/relval/CMSSW_3_1_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0005/182F7A26-A44D-DE11-9251-0019DB29C614.root',
            '/store/relval/CMSSW_3_1_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0005/0638B1A4-A24D-DE11-A1F2-001617C3B778.root',
            '/store/relval/CMSSW_3_1_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0005/0000D2AC-A34D-DE11-A869-000423D98AF0.root' 
        ] );

        #dataset = cms.untracked.vstring('/RelValQCD_Pt_80_120/CMSSW_3_1_0_pre6_IDEAL_31X_v1/GEN-SIM-DIGI-RAW-HLTDEBUG')
        #readFiles.extend( [
        #    '/store/relval/CMSSW_3_1_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0002/F8A7CE85-DA32-DE11-9BCF-001617C3B6E2.root',
        #    '/store/relval/CMSSW_3_1_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0002/DA8F0DD9-E232-DE11-B38E-001617DBCF6A.root',
        #    '/store/relval/CMSSW_3_1_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0002/AC307E58-E932-DE11-B81A-000423D6C8EE.root',
        #    '/store/relval/CMSSW_3_1_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0002/A2F7A41F-DA32-DE11-9104-001617C3B6CC.root',
        #    '/store/relval/CMSSW_3_1_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0002/863B5F6F-E432-DE11-B136-000423D94908.root',
        #    '/store/relval/CMSSW_3_1_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0002/706894A4-F432-DE11-81CD-000423D99AAA.root',
        #    '/store/relval/CMSSW_3_1_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0002/64E7EFA4-F332-DE11-8759-001617DBD288.root',
        #    '/store/relval/CMSSW_3_1_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0002/6258726F-EC32-DE11-85A5-000423D9870C.root',
        #    '/store/relval/CMSSW_3_1_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0002/5E484B7A-1733-DE11-82B4-000423D95030.root',
        #    '/store/relval/CMSSW_3_1_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0002/4E7CE05D-F332-DE11-9E6C-000423D944FC.root',
        #    '/store/relval/CMSSW_3_1_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0002/42C0E81D-DA32-DE11-B51A-001617C3B6C6.root',
        #    '/store/relval/CMSSW_3_1_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0002/3CE247BB-DA32-DE11-9AE6-001617E30E28.root',
        #    '/store/relval/CMSSW_3_1_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0002/38CD0D02-F132-DE11-8D5D-000423D9970C.root',
        #    '/store/relval/CMSSW_3_1_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0002/36B44188-F232-DE11-858F-00161757BF42.root',
        #    '/store/relval/CMSSW_3_1_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0002/2C4658FC-DA32-DE11-8251-000423D98C20.root',
        #    '/store/relval/CMSSW_3_1_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0002/2C2C0C6D-DA32-DE11-A380-001617C3B6C6.root',
        #    '/store/relval/CMSSW_3_1_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0002/10056DBD-DF32-DE11-9777-001617E30E2C.root'
        #]);

        #dataset = cms.untracked.vstring('/RelValQCD_Pt_80_120/CMSSW_2_2_10_IDEAL_V12_v1/GEN-SIM-DIGI-RAW-HLTDEBUG')
        #readFiles.extend( [
        #    '/store/relval/CMSSW_2_2_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V12_v1/0003/FC798D81-CB3D-DE11-9FBF-001D09F290BF.root',
        #    '/store/relval/CMSSW_2_2_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V12_v1/0003/F80D1F07-CA3D-DE11-B1ED-001617DBCF6A.root',
        #    '/store/relval/CMSSW_2_2_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V12_v1/0003/F63FDCEB-CA3D-DE11-A3AD-0019B9F72BFF.root',
        #    '/store/relval/CMSSW_2_2_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V12_v1/0003/E6B3876B-CD3D-DE11-9E0D-000423D99F1E.root',
        #    '/store/relval/CMSSW_2_2_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V12_v1/0003/CE04F0D4-CC3D-DE11-8465-001D09F244DE.root',
        #    '/store/relval/CMSSW_2_2_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V12_v1/0003/C43129D0-CA3D-DE11-BA1C-001D09F2A49C.root',
        #    '/store/relval/CMSSW_2_2_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V12_v1/0003/B2F78729-CD3D-DE11-9019-001D09F24448.root',
        #    '/store/relval/CMSSW_2_2_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V12_v1/0003/9CDE8A75-CB3D-DE11-98B0-001D09F2A690.root',
        #    '/store/relval/CMSSW_2_2_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V12_v1/0003/96F539A1-CB3D-DE11-9B6B-001D09F25217.root',
        #    '/store/relval/CMSSW_2_2_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V12_v1/0003/94EB24CC-CB3D-DE11-A4D8-0019DB29C614.root',
        #    '/store/relval/CMSSW_2_2_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V12_v1/0003/94919798-C83D-DE11-B793-001D09F24F65.root',
        #    '/store/relval/CMSSW_2_2_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V12_v1/0003/90979075-CD3D-DE11-9D71-001D09F28F0C.root',
        #    '/store/relval/CMSSW_2_2_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V12_v1/0003/6A2DE1D5-CA3D-DE11-B058-001D09F23A84.root',
        #    '/store/relval/CMSSW_2_2_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V12_v1/0003/5EE96B31-CB3D-DE11-819A-001D09F25442.root',
        #    '/store/relval/CMSSW_2_2_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V12_v1/0003/2AC7ADD9-C83D-DE11-B4FF-001D09F2447F.root',
        #    '/store/relval/CMSSW_2_2_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V12_v1/0003/149A729C-CA3D-DE11-AA49-001617C3B5D8.root',
        #    '/store/relval/CMSSW_2_2_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V12_v1/0003/142EB962-CA3D-DE11-AAFA-001617C3B6C6.root',
        #    '/store/relval/CMSSW_2_2_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V12_v1/0003/10C82EA9-043E-DE11-B745-001D09F29524.root',
        #    '/store/relval/CMSSW_2_2_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V12_v1/0003/08A81316-CC3D-DE11-8500-001D09F2546F.root',
        #    '/store/relval/CMSSW_2_2_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V12_v1/0003/06C016EB-C93D-DE11-9A07-001D09F23C73.root' 
        #] );

        #dataset = cms.untracked.vstring('/RelValTTbar/CMSSW_3_1_0_pre9_IDEAL_31X_v1/GEN-SIM-DIGI-RAW-HLTDEBUG')
        #readFiles.extend( [
        #    '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0007/EA6C6D8A-5E4F-DE11-BD34-0030487C6062.root',
        #    '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0007/E8BC81A9-F64E-DE11-8727-000423D99AAA.root',
        #    '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0007/E2AA7EE2-F64E-DE11-A06D-001617C3B6CE.root',
        #    '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0007/C4CB7C79-5E4F-DE11-9EA2-001D09F28F0C.root',
        #    '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0007/B691AE7B-5E4F-DE11-ABF5-001D09F24637.root',
        #    '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0007/A4118E73-F64E-DE11-AE57-000423D6B358.root',
        #    '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0007/8C610683-5E4F-DE11-817D-0030487A1FEC.root',
        #    '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0007/8A108B79-5E4F-DE11-8CD7-001D09F253D4.root',
        #    '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0007/7AE06281-6E4F-DE11-8C85-000423D99896.root',
        #    '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0007/6A2FE089-5E4F-DE11-9BA6-000423D9863C.root',
        #    '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0007/56845B82-5E4F-DE11-B375-000423D6CAF2.root',
        #    '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0007/4824EE85-5E4F-DE11-8F3A-000423D98804.root',
        #    '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0007/44A7AB85-5E4F-DE11-B80A-001D09F24600.root',
        #    '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0007/40302A87-5E4F-DE11-A997-001D09F2AD7F.root',
        #    '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0007/3CADAA7E-5E4F-DE11-B5C7-0019B9F730D2.root',
        #    '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0007/3C7EC0B0-5E4F-DE11-AFB3-001D09F253D4.root',
        #    '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0007/36A89B8D-F64E-DE11-91FF-001617C3B778.root',
        #    '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0007/369D3C85-5E4F-DE11-A2F7-001D09F25456.root',
        #    '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0007/2E42BD9B-524F-DE11-A1CF-000423D6C8EE.root',
        #    '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0007/2ABD6A7A-5E4F-DE11-8F5F-001D09F242EA.root',
        #    '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0007/24C60B86-5E4F-DE11-8C87-000423D6CA42.root' 
        #] );


        secFiles.extend([
            ])

    elif useGlobalTag.count('STARTUP') :

        #dataset = cms.untracked.vstring('/RelValQCD_Pt_80_120/CMSSW_3_1_0_pre8_STARTUP_31X_v1/GEN-SIM-DIGI-RAW-HLTDEBUG')        
        #readFiles.extend( [
        #    '/store/relval/CMSSW_3_1_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0006/7C2BE1B8-DB4D-DE11-8A6D-001D09F24664.root',
        #    '/store/relval/CMSSW_3_1_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0005/FA7B9921-564D-DE11-8BC3-001D09F231B0.root',
        #    '/store/relval/CMSSW_3_1_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0005/F076816C-504D-DE11-A394-001D09F253C0.root',
        #    '/store/relval/CMSSW_3_1_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0005/EEBCD418-584D-DE11-BEBB-001D09F251FE.root',
        #    '/store/relval/CMSSW_3_1_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0005/EEA9EC0B-514D-DE11-A0B4-001D09F2B30B.root',
        #    '/store/relval/CMSSW_3_1_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0005/D0E77935-574D-DE11-A716-0019B9F705A3.root',
        #    '/store/relval/CMSSW_3_1_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0005/C2882148-4B4D-DE11-8483-001D09F253FC.root',
        #    '/store/relval/CMSSW_3_1_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0005/B4F8E75A-514D-DE11-AD58-001D09F241B9.root',
        #    '/store/relval/CMSSW_3_1_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0005/9ACBB3BC-564D-DE11-9EB0-001617E30E2C.root',
        #    '/store/relval/CMSSW_3_1_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0005/94373D30-4E4D-DE11-871F-000423D6B42C.root',
        #    '/store/relval/CMSSW_3_1_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0005/8E5CC1E3-514D-DE11-BD44-001D09F251BD.root',
        #    '/store/relval/CMSSW_3_1_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0005/8827D200-554D-DE11-B7C0-001617E30D0A.root',
        #    '/store/relval/CMSSW_3_1_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0005/5CE0A773-574D-DE11-AE55-000423D6CA02.root',
        #    '/store/relval/CMSSW_3_1_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0005/4A5870B5-5A4D-DE11-9038-001D09F2424A.root',
        #    '/store/relval/CMSSW_3_1_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0005/405C0068-504D-DE11-B38A-001D09F241B4.root',
        #    '/store/relval/CMSSW_3_1_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0005/2E8FA2FF-4F4D-DE11-884F-001D09F29849.root',
        #    '/store/relval/CMSSW_3_1_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0005/1C8EA945-524D-DE11-A6A7-001D09F24691.root' 
        #] );

        #dataset = cms.untracked.vstring('/RelValQCD_Pt_80_120/CMSSW_3_1_0_pre6_STARTUP_31X_v1/GEN-SIM-DIGI-RAW-HLTDEBUG')        
        #readFiles.extend([
        #    '/store/relval/CMSSW_3_1_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0002/FE009142-7032-DE11-92E7-001617E30D06.root',
        #    '/store/relval/CMSSW_3_1_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0002/ECF1AB4F-6E32-DE11-B245-001617DBD288.root',
        #    '/store/relval/CMSSW_3_1_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0002/EC7B074F-7032-DE11-90BC-0019DB29C620.root',
        #    '/store/relval/CMSSW_3_1_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0002/EA6374E4-6E32-DE11-B43C-000423D99AAA.root',
        #    '/store/relval/CMSSW_3_1_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0002/C44F29A7-1733-DE11-B06C-000423D99614.root',
        #    '/store/relval/CMSSW_3_1_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0002/C2B436EA-6E32-DE11-9909-001617C3B6CC.root',
        #    '/store/relval/CMSSW_3_1_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0002/B6E022DA-6D32-DE11-9E42-0019DB29C620.root',
        #    '/store/relval/CMSSW_3_1_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0002/9AC6F5E6-8732-DE11-AC07-001617C3B77C.root',
        #    '/store/relval/CMSSW_3_1_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0002/987CFDE4-8732-DE11-983E-000423D944FC.root',
        #    '/store/relval/CMSSW_3_1_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0002/9642BBEF-8732-DE11-A0E9-000423D996C8.root',
        #    '/store/relval/CMSSW_3_1_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0002/648F6143-6E32-DE11-BEF7-0016177CA7A0.root',
        #    '/store/relval/CMSSW_3_1_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0002/600C1B20-6E32-DE11-AB30-0016177CA7A0.root',
        #    '/store/relval/CMSSW_3_1_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0002/58862D55-6E32-DE11-A088-000423D98844.root',
        #    '/store/relval/CMSSW_3_1_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0002/524489E6-8732-DE11-BF23-000423D60FF6.root',
        #    '/store/relval/CMSSW_3_1_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0002/46854355-7032-DE11-BB01-001617C3B6E2.root',
        #    '/store/relval/CMSSW_3_1_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0002/1A971B55-8A32-DE11-AF0D-000423D944DC.root',
        #    '/store/relval/CMSSW_3_1_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0002/12FB614A-7032-DE11-8627-001617E30F56.root'         
        #]);


        #dataset = cms.untracked.vstring('/RelValQCD_Pt_80_120/CMSSW_2_2_10_STARTUP_V11_v1/GEN-SIM-DIGI-RAW-HLTDEBUG')
        #readFiles.extend( [
        #    '/store/relval/CMSSW_2_2_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V11_v1/0003/1825DE62-043E-DE11-9AB7-001617C3B654.root',
        #    '/store/relval/CMSSW_2_2_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V11_v1/0002/F2520292-863D-DE11-ACDF-001D09F2538E.root',
        #    '/store/relval/CMSSW_2_2_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V11_v1/0002/F05F275A-893D-DE11-94B8-001D09F25217.root',
        #    '/store/relval/CMSSW_2_2_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V11_v1/0002/D2855C5D-8A3D-DE11-8084-001D09F24D8A.root',
        #    '/store/relval/CMSSW_2_2_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V11_v1/0002/AEAE2C43-893D-DE11-B3C0-001D09F24637.root',
        #    '/store/relval/CMSSW_2_2_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V11_v1/0002/AE91E67A-873D-DE11-8BD6-001D09F297EF.root',
        #    '/store/relval/CMSSW_2_2_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V11_v1/0002/AC1958A0-883D-DE11-96AE-001D09F29524.root',
        #    '/store/relval/CMSSW_2_2_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V11_v1/0002/92706ABC-8B3D-DE11-A1E2-001D09F297EF.root',
        #    '/store/relval/CMSSW_2_2_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V11_v1/0002/8EAE234C-8B3D-DE11-A87A-0019B9F72CC2.root',
        #    '/store/relval/CMSSW_2_2_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V11_v1/0002/884EBAF1-873D-DE11-B5B7-001D09F291D2.root',
        #    '/store/relval/CMSSW_2_2_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V11_v1/0002/68D12010-8C3D-DE11-BA50-001617C3B70E.root',
        #    '/store/relval/CMSSW_2_2_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V11_v1/0002/68615FA2-8A3D-DE11-85AD-001D09F291D7.root',
        #    '/store/relval/CMSSW_2_2_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V11_v1/0002/5C7DF99A-883D-DE11-BF37-001D09F2A465.root',
        #    '/store/relval/CMSSW_2_2_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V11_v1/0002/52ED45CE-893D-DE11-A52E-001D09F24D4E.root',
        #    '/store/relval/CMSSW_2_2_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V11_v1/0002/3AE45D80-863D-DE11-ADAA-00304879FA4A.root',
        #    '/store/relval/CMSSW_2_2_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V11_v1/0002/164B8DC2-8B3D-DE11-A1BA-001617C3B654.root',
        #    '/store/relval/CMSSW_2_2_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V11_v1/0002/14396117-8B3D-DE11-9A10-001D09F291D7.root',
        #    '/store/relval/CMSSW_2_2_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V11_v1/0002/0EB74EF7-863D-DE11-AD5E-001D09F29597.root',
        #    '/store/relval/CMSSW_2_2_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V11_v1/0002/026314A2-8A3D-DE11-8CB6-001D09F2447F.root',
        #    '/store/relval/CMSSW_2_2_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V11_v1/0002/00081940-863D-DE11-8158-0019B9F6C674.root' 
        #] );

        dataset = cms.untracked.vstring('/RelValTTbar/CMSSW_3_1_0_pre9_STARTUP_31X_v1/GEN-SIM-DIGI-RAW-HLTDEBUG')
        readFiles.extend( [
            '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0006/F20698E4-834E-DE11-AD06-001D09F2441B.root',
            '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0006/ECCD5B61-874E-DE11-9485-001D09F250AF.root',
            '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0006/E82E2C72-674E-DE11-9D8E-001617C3B706.root',
            '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0006/E063E7B0-6D4E-DE11-A29E-001D09F253FC.root',
            '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0006/BEC3F2B7-7C4E-DE11-96D4-001D09F2437B.root',
            '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0006/A495AF09-6B4E-DE11-8E11-001D09F248F8.root',
            '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0006/943F56B1-7D4E-DE11-90D1-001D09F2AF1E.root',
            '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0006/92B7EF93-684E-DE11-A1F0-000423D6C8EE.root',
            '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0006/86936088-7E4E-DE11-9A2A-001D09F24EE3.root',
            '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0006/7667DB33-674E-DE11-B5FF-001D09F29597.root',
            '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0006/665E47B7-724E-DE11-B705-001D09F2AF96.root',
            '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0006/60E23679-664E-DE11-A436-001D09F25442.root',
            '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0006/58E8888D-704E-DE11-BD30-001D09F27003.root',
            '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0006/50034FD3-6F4E-DE11-A525-001D09F23A34.root',
            '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0006/44D03C92-6B4E-DE11-9EB7-001617C3B6FE.root',
            '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0006/44290C09-814E-DE11-9AB6-001D09F27003.root',
            '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0006/32457AF8-784E-DE11-AD7A-001D09F23A07.root',
            '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0006/10BDFCF9-6C4E-DE11-8A57-001D09F29619.root',
            '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0006/10070AB8-754E-DE11-9B66-001D09F2514F.root',
            '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0006/021AB859-6C4E-DE11-AAB2-001D09F253D4.root',
            '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0000/FAC5D36B-224E-DE11-A638-0018F3D09688.root',
            '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0000/F0AE1C5B-2E4E-DE11-87AD-0018F3D0960C.root',
            '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0000/EEDA31B7-294E-DE11-8A0F-001A92971B62.root',
            '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0000/EEA77716-2B4E-DE11-AD91-0018F3D09642.root',
            '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0000/ECBC6A61-204E-DE11-BBBB-003048679236.root',
            '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0000/EAE27C79-2C4E-DE11-8826-001A928116B0.root',
            '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0000/EA3FC178-2C4E-DE11-8D7C-0018F3D096E8.root',
            '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0000/E4840369-2E4E-DE11-AAB7-0013D4C3BAFA.root',
            '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0000/E02F8963-204E-DE11-9C37-003048D15E02.root',
            '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0000/DC16D029-214E-DE11-860D-0030486790BE.root',
            '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0000/D4F6E7AE-294E-DE11-8CAC-001BFCDBD160.root',
            '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0000/CAB0D74C-2D4E-DE11-AA50-0018F3D09702.root',
            '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0000/C42946B1-2D4E-DE11-9C3E-0018F3D095F6.root',
            '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0000/ACED7BAC-2D4E-DE11-ACFB-0018F3D096E8.root',
            '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0000/ACB4B857-2E4E-DE11-95D0-003048678B74.root',
            '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0000/A8D4E4B7-2B4E-DE11-A6A1-0018F3D0966C.root',
            '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0000/98847871-2A4E-DE11-9874-001BFCDBD11E.root',
            '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0000/94761BAF-1F4E-DE11-9155-003048678ADA.root',
            '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0000/90CE1802-214E-DE11-AE32-003048D15D22.root',
            '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0000/8C04AA1B-2B4E-DE11-839D-001A9281171C.root',
            '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0000/8A915ABA-2D4E-DE11-9D43-001731AF692F.root',
            '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0000/84710B49-2D4E-DE11-81E6-0018F3D095F8.root',
            '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0000/80E67C4B-204E-DE11-A363-003048678FDE.root',
            '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0000/644402F8-5F4F-DE11-92A8-00304867926C.root',
            '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0000/506ECA80-224E-DE11-9647-0018F3D096EE.root',
            '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0000/40BA330F-2B4E-DE11-8F09-001BFCDBD100.root',
            '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0000/3E09458C-224E-DE11-BF30-001A92971B36.root',
            '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0000/2A8B4160-204E-DE11-B066-003048D15CC0.root',
            '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0000/268A207A-2C4E-DE11-9015-001A92971ACE.root',
            '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0000/20FE4F45-2D4E-DE11-9CEC-001A928116B0.root',
            '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0000/1282FB16-214E-DE11-9A8B-003048678FDE.root',
            '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0000/10662B7C-2C4E-DE11-8504-001A92971B80.root',
            '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0000/0E56B627-214E-DE11-B804-001A92971BDA.root',
            '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0000/08133658-2D4E-DE11-B64B-0013D4C3BAFA.root',
            '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0000/06A6E5BC-2D4E-DE11-8958-001731AF6873.root',
            '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0000/0038EF82-2C4E-DE11-B565-001731AF65F7.root' 
    ] );


        
        
        secFiles.extend([
            ])
    else :
        print 'Error: Global Tag ', useGlobalTag, ' not defined.'    

else : 

    # CRAFT data FIXME when Global Tag available (if data compatible)
    dataset = ''
    
    readFiles.extend([
        ]);

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
    
    if useGlobalTag.count('IDEAL') :
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
        
else :
    print 'Using default L1 trigger menu from Global Tag ', useGlobalTag    
      

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
    
# path to be run
process.p = cms.Path(process.l1GtTrigReport)


# Message Logger
process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.debugModules = ['l1GtTrigReport']
process.MessageLogger.categories.append('L1GtTrigReport')
process.MessageLogger.destinations = ['L1GtTrigReport']
process.MessageLogger.L1GtTrigReport = cms.untracked.PSet(
    #threshold=cms.untracked.string('DEBUG'),
    threshold = cms.untracked.string('INFO'),
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
    default=cms.untracked.PSet(
        limit=cms.untracked.int32(-1)  
    ),
    L1GtTrigReport=cms.untracked.PSet(
        limit=cms.untracked.int32(-1)  
    )
)

# summary
process.options = cms.untracked.PSet(
    wantSummary=cms.untracked.bool(True)
)

