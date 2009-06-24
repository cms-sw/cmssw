# gctValidationTest_cfg.py
#
# G Heath 23/09/08
#
#

import FWCore.ParameterSet.Config as cms

# The top-level process
process = cms.Process("TEST")

startupConfig = bool(True)
qcdData = bool(False)

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 100

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('gctValidationPlots.root')
)
 
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring() 
process.source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)

if startupConfig:
    if qcdData:

        readFiles.extend( [
               '/store/relval/CMSSW_3_1_0_pre10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0008/FA9DDC90-9A57-DE11-8E89-001D09F2426D.root',
               '/store/relval/CMSSW_3_1_0_pre10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0008/FA5DD45B-A457-DE11-9D23-0019B9F72D71.root',
               '/store/relval/CMSSW_3_1_0_pre10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0008/F8C8EFB7-9657-DE11-B4C1-000423D94494.root',
               '/store/relval/CMSSW_3_1_0_pre10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0008/ECCF3F1D-9157-DE11-B1B6-001D09F24691.root',
               '/store/relval/CMSSW_3_1_0_pre10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0008/E0926969-9657-DE11-B7FA-001D09F23944.root',
               '/store/relval/CMSSW_3_1_0_pre10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0008/B822D825-9A57-DE11-A7F6-001D09F24303.root',
               '/store/relval/CMSSW_3_1_0_pre10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0008/AA5EEE21-A057-DE11-BE67-000423D991D4.root',
               '/store/relval/CMSSW_3_1_0_pre10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0008/A0CD1CAF-9357-DE11-A46E-001D09F2960F.root',
               '/store/relval/CMSSW_3_1_0_pre10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0008/8E39D976-9257-DE11-9F41-0030487C6062.root',
               '/store/relval/CMSSW_3_1_0_pre10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0008/72F0CDD7-9257-DE11-97EE-0030487A3232.root',
               '/store/relval/CMSSW_3_1_0_pre10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0008/709E22D1-A757-DE11-B19A-001617C3B77C.root',
               '/store/relval/CMSSW_3_1_0_pre10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0008/706A7B92-9457-DE11-BD6F-0019B9F70468.root',
               '/store/relval/CMSSW_3_1_0_pre10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0008/6E7B80FB-9E57-DE11-8F23-001617C3B65A.root',
               '/store/relval/CMSSW_3_1_0_pre10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0008/2EE5F6DB-A557-DE11-AC90-0019B9F704D6.root',
               '/store/relval/CMSSW_3_1_0_pre10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0008/204107E2-0458-DE11-949F-001D09F25208.root',
               '/store/relval/CMSSW_3_1_0_pre10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0008/182FAEFA-9B57-DE11-A996-001D09F24D8A.root',
               '/store/relval/CMSSW_3_1_0_pre10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0008/0491604D-9457-DE11-98DC-0019B9F705A3.root',
               '/store/relval/CMSSW_3_1_0_pre10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0001/E67CAF12-3C5A-DE11-8014-001A92811700.root',
               '/store/relval/CMSSW_3_1_0_pre10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0001/BAFA3153-F757-DE11-A8C3-00304875AA6F.root',
               '/store/relval/CMSSW_3_1_0_pre10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0001/BA04FF79-2F59-DE11-9DEB-00304875A9C5.root',
               '/store/relval/CMSSW_3_1_0_pre10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0001/AAE2853B-6959-DE11-957B-001A928116E0.root',
               '/store/relval/CMSSW_3_1_0_pre10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0001/A2393669-185A-DE11-B36E-003048678FB8.root',
               '/store/relval/CMSSW_3_1_0_pre10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0001/9869D52F-2F58-DE11-8BD1-001A92811748.root',
               '/store/relval/CMSSW_3_1_0_pre10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0001/92B6E527-9859-DE11-A753-001731A28BE1.root',
               '/store/relval/CMSSW_3_1_0_pre10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0001/8437187A-0B58-DE11-ABB0-0018F3D09698.root',
               '/store/relval/CMSSW_3_1_0_pre10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0001/5A4D8C82-EE57-DE11-BFE3-001A92810AB8.root',
               '/store/relval/CMSSW_3_1_0_pre10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0001/56AB125F-5859-DE11-B0AD-0018F3D09648.root',
               '/store/relval/CMSSW_3_1_0_pre10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0001/4E5127FA-0E5A-DE11-81F4-0018F3D09670.root',
               '/store/relval/CMSSW_3_1_0_pre10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0001/44681A09-8359-DE11-BC80-0030486792AC.root',
               '/store/relval/CMSSW_3_1_0_pre10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0001/1091F5EF-F557-DE11-83CB-001A928116E8.root' ] );

    else:

        readFiles.extend( [
               '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0009/E8C7007E-1158-DE11-96E0-001D09F252DA.root',
               '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0008/C064EAF3-EC57-DE11-A281-001D09F23174.root',
               '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0008/B2A831C2-EC57-DE11-8C57-001D09F24047.root',
               '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0008/B269212B-F157-DE11-AF19-000423D98EC4.root',
               '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0008/9C8C2AA3-EF57-DE11-8620-0030487C6062.root',
               '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0008/928B34E2-ED57-DE11-ADFD-001D09F24D8A.root',
               '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0008/8CFC6198-EF57-DE11-8EEE-001D09F24763.root',
               '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0008/882189B6-ED57-DE11-B866-001617E30D12.root',
               '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0008/881635CD-EF57-DE11-9FD5-001D09F29146.root',
               '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0008/86BFF7D7-EE57-DE11-9AC8-001D09F24763.root',
               '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0008/76BB517B-F157-DE11-83DD-001D09F2905B.root',
               '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0008/62123202-ED57-DE11-9940-000423D98800.root',
               '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0008/56DDEC87-F257-DE11-AEB4-000423D98834.root',
               '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0008/50163A1D-F157-DE11-B69A-001D09F24DA8.root',
               '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0008/3A79D5D5-0458-DE11-9CEF-001D09F26509.root',
               '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0008/32051996-EB57-DE11-874F-001D09F25438.root',
               '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0008/2E704134-F057-DE11-8CB0-000423D996B4.root',
               '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0008/18C99F55-EC57-DE11-8DCF-001D09F2915A.root',
               '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0008/18374144-EE57-DE11-B9E4-0030487D0D3A.root',
               '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0008/14A881A0-EE57-DE11-BB0B-001D09F2503C.root',
               '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0008/144581B6-F057-DE11-9F1C-000423D992A4.root' ] );

else:
    if qcdData:

        readFiles.extend( [
               '/store/relval/CMSSW_3_1_0_pre10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0008/FCF5E972-4257-DE11-8176-001D09F24489.root',
               '/store/relval/CMSSW_3_1_0_pre10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0008/E67E97E4-4157-DE11-A081-001D09F248FD.root',
               '/store/relval/CMSSW_3_1_0_pre10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0008/D88F8CB1-4057-DE11-A8C5-001D09F25208.root',
               '/store/relval/CMSSW_3_1_0_pre10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0008/D04D3EB6-4257-DE11-8396-001D09F24259.root',
               '/store/relval/CMSSW_3_1_0_pre10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0008/B48D500C-4357-DE11-8351-001D09F23A3E.root',
               '/store/relval/CMSSW_3_1_0_pre10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0008/B2A85936-4257-DE11-BF64-001D09F25325.root',
               '/store/relval/CMSSW_3_1_0_pre10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0008/AAFB32E9-4257-DE11-B320-001D09F24EAC.root',
               '/store/relval/CMSSW_3_1_0_pre10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0008/8C68554C-4157-DE11-9D04-001D09F24EC0.root',
               '/store/relval/CMSSW_3_1_0_pre10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0008/6867F8C2-4257-DE11-B267-001D09F28C1E.root',
               '/store/relval/CMSSW_3_1_0_pre10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0008/520A3EB4-4257-DE11-8860-001D09F23A3E.root',
               '/store/relval/CMSSW_3_1_0_pre10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0008/4CEDD4D7-4157-DE11-8820-001D09F252F3.root',
               '/store/relval/CMSSW_3_1_0_pre10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0008/2E21341C-4257-DE11-9A60-001D09F2424A.root',
               '/store/relval/CMSSW_3_1_0_pre10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0008/1E94E0D4-4257-DE11-B049-001D09F250AF.root',
               '/store/relval/CMSSW_3_1_0_pre10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0008/187220D2-0458-DE11-B440-001D09F24682.root',
               '/store/relval/CMSSW_3_1_0_pre10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0008/10F15E8B-4157-DE11-A0D4-001D09F252F3.root',
               '/store/relval/CMSSW_3_1_0_pre10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0008/0C42A60F-4257-DE11-AAEB-000423D985E4.root',
               '/store/relval/CMSSW_3_1_0_pre10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0008/00819765-4057-DE11-B071-001D09F251E0.root' ] );

    else:

        readFiles.extend( [
               '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0008/E4C24E91-CD57-DE11-90DE-001D09F2532F.root',
               '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0008/E469F214-C857-DE11-A950-001D09F252E9.root',
               '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0008/DE489DF6-C657-DE11-ABCA-001D09F28D54.root',
               '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0008/D4CBC4D7-CC57-DE11-924E-001D09F29619.root',
               '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0008/CA9880CE-AC57-DE11-B891-001D09F25208.root',
               '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0008/B67BD00F-F257-DE11-95E5-001D09F2B2CF.root',
               '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0008/B66C0FD8-CC57-DE11-9593-001D09F297EF.root',
               '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0008/B0863555-CA57-DE11-B198-000423D94534.root',
               '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0008/98DD9243-B357-DE11-82B4-00304879FBB2.root',
               '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0008/906F7F83-C557-DE11-BF54-000423D99AAA.root',
               '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0008/4EC4DD37-CD57-DE11-8F88-0019B9F705A3.root',
               '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0008/4ADD5FD0-BB57-DE11-B8F0-001617DBD5AC.root',
               '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0008/4A2A45E1-AA57-DE11-82AA-0030487A1990.root',
               '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0008/404DF239-C357-DE11-849D-001D09F241F0.root',
               '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0008/3CB939FA-B757-DE11-BFB5-001D09F24448.root',
               '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0008/36647208-C757-DE11-94B1-001D09F24763.root',
               '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0008/3447A45D-CD57-DE11-9396-001D09F24489.root',
               '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0008/124FFE86-CD57-DE11-92D7-001D09F24259.root',
               '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0008/124CD9B7-C457-DE11-87D4-001D09F25393.root',
               '/store/relval/CMSSW_3_1_0_pre10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0008/081FA023-C457-DE11-BAC8-0019B9F704D6.root' ] );



secFiles.extend( (
               ) )

# Copied from:
# L1Trigger/Configuration/test/L1EmulatorFromRaw_cfg.py
#
# standard includes
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
if startupConfig:
    process.GlobalTag.globaltag = 'STARTUP_31X::All'
else:
    process.GlobalTag.globaltag = 'IDEAL_31X::All'

# unpack raw data
process.load("Configuration.StandardSequences.RawToDigi_cff")

# run trigger primitive generation on unpacked digis, then central L1
process.load("L1Trigger.Configuration.CaloTriggerPrimitives_cff")
process.load("L1Trigger.Configuration.SimL1Emulator_cff")

# set the new input tags after RawToDigi for the TPG producers
process.simEcalTriggerPrimitiveDigis.Label = 'ecalDigis'
process.simHcalTriggerPrimitiveDigis.inputLabel = cms.VInputTag(cms.InputTag('hcalDigis'), 
                                                                cms.InputTag('hcalDigis'))
#
process.simDtTriggerPrimitiveDigis.digiTag = 'muonDTDigis'
#
process.simCscTriggerPrimitiveDigis.CSCComparatorDigiProducer = cms.InputTag('muonCSCDigis',
                                                                             'MuonCSCComparatorDigi')
process.simCscTriggerPrimitiveDigis.CSCWireDigiProducer = cms.InputTag('muonCSCDigis',
                                                                       'MuonCSCWireDigi')
#
process.simRpcTriggerDigis.label = 'muonRPCDigis'

#
process.simGctDigis.writeInternalData = True

# GCT validation
process.load("L1Trigger.GlobalCaloTrigger.l1GctValidation_cfi")


## process.p = cms.Path(
##     process.ecalDigis
##     *process.hcalDigis
##     *process.muonDTDigis
##     *process.muonCSCDigis
##     *process.muonRPCDigis
##     *process.CaloTriggerPrimitives
##     *process.SimL1Emulator
##     *process.l1GctValidation
## )
process.p = cms.Path(
    process.ecalDigis
    *process.hcalDigis
    *process.muonDTDigis
    *process.muonCSCDigis
    *process.muonRPCDigis
    *process.CaloTriggerPrimitives
    *process.simRctDigis
    *process.simGctDigis
    *process.l1GctValidation
)





