# gctValidationTest_cfg.py
#
# G Heath 23/09/08
#
#

import FWCore.ParameterSet.Config as cms

# The top-level process
process = cms.Process("TEST")

startupConfig = bool(True)
qcdData = bool(True)

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
               '/store/relval/CMSSW_3_1_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v1/0001/2869A25E-3166-DE11-9634-001D09F28755.root' ] );

    else:

        readFiles.extend( [
               '/store/relval/CMSSW_3_1_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v1/0001/F472473A-9966-DE11-A190-000423D6A6F4.root',
               '/store/relval/CMSSW_3_1_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v1/0001/EACE4CF8-9766-DE11-BF04-000423D99896.root',
               '/store/relval/CMSSW_3_1_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v1/0001/E00B438B-9566-DE11-80DF-001D09F24763.root',
               '/store/relval/CMSSW_3_1_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v1/0001/DAC06FF6-8F66-DE11-8718-001D09F23174.root',
               '/store/relval/CMSSW_3_1_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v1/0001/D09AE612-9666-DE11-BAF6-001D09F2924F.root',
               '/store/relval/CMSSW_3_1_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v1/0001/AEB24742-9266-DE11-A673-001D09F2516D.root',
               '/store/relval/CMSSW_3_1_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v1/0001/9AD52C86-9666-DE11-BBA1-001D09F248FD.root',
               '/store/relval/CMSSW_3_1_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v1/0001/7CA66E6B-9466-DE11-8635-0019DB29C614.root',
               '/store/relval/CMSSW_3_1_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v1/0001/74121AB2-9366-DE11-B322-000423D98844.root',
               '/store/relval/CMSSW_3_1_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v1/0001/70A5735F-9966-DE11-AE0F-0019B9F72F97.root',
               '/store/relval/CMSSW_3_1_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v1/0001/6681E6EF-9666-DE11-B54C-001D09F24F1F.root',
               '/store/relval/CMSSW_3_1_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v1/0001/5CED3A4A-9066-DE11-B817-001D09F26509.root',
               '/store/relval/CMSSW_3_1_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v1/0001/5621975A-9B66-DE11-962A-001D09F251E0.root',
               '/store/relval/CMSSW_3_1_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v1/0001/529EB6C9-8E66-DE11-94BE-001D09F2437B.root',
               '/store/relval/CMSSW_3_1_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v1/0001/4E055049-9066-DE11-86A8-001D09F24D8A.root',
               '/store/relval/CMSSW_3_1_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v1/0001/48ACEEEC-9866-DE11-9293-001D09F29538.root',
               '/store/relval/CMSSW_3_1_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v1/0001/44E0A4A9-9166-DE11-A20A-001D09F29321.root',
               '/store/relval/CMSSW_3_1_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v1/0001/28F921C6-9166-DE11-A899-001D09F25325.root',
               '/store/relval/CMSSW_3_1_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v1/0001/20D964E0-9866-DE11-AA34-001D09F248F8.root',
               '/store/relval/CMSSW_3_1_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V1-v1/0001/084BB2C6-9266-DE11-A1F8-000423D33970.root' ] );

else:
    if qcdData:

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
               '/store/relval/CMSSW_3_1_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0001/28483F30-6B66-DE11-B045-001617DBD224.root' ] );

    else:

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
               '/store/relval/CMSSW_3_1_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0001/14AEBDFE-C666-DE11-AA23-001D09F28755.root' ] );



secFiles.extend( (
               ) )

# Copied from:
# L1Trigger/Configuration/test/L1EmulatorFromRaw_cfg.py
#
# standard includes
process.load("Configuration.StandardSequences.GeometryDB_cff")
process.load("L1Trigger.GlobalCaloTrigger.test.gctConfig_cff")

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





