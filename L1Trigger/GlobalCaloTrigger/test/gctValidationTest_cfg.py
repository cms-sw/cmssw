# gctValidationTest_cfg.py
#
# G Heath 23/09/08
#
#

import FWCore.ParameterSet.Config as cms

# The top-level process
process = cms.Process("TEST")

process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('gctValidationPlots.root')
)
 
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring() 
process.source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)
readFiles.extend( [
       '/store/relval/CMSSW_3_1_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0006/7C2BE1B8-DB4D-DE11-8A6D-001D09F24664.root',
       '/store/relval/CMSSW_3_1_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0005/FA7B9921-564D-DE11-8BC3-001D09F231B0.root',
       '/store/relval/CMSSW_3_1_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0005/F076816C-504D-DE11-A394-001D09F253C0.root',
       '/store/relval/CMSSW_3_1_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0005/EEBCD418-584D-DE11-BEBB-001D09F251FE.root',
       '/store/relval/CMSSW_3_1_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0005/EEA9EC0B-514D-DE11-A0B4-001D09F2B30B.root',
       '/store/relval/CMSSW_3_1_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0005/D0E77935-574D-DE11-A716-0019B9F705A3.root',
       '/store/relval/CMSSW_3_1_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0005/C2882148-4B4D-DE11-8483-001D09F253FC.root',
       '/store/relval/CMSSW_3_1_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0005/B4F8E75A-514D-DE11-AD58-001D09F241B9.root',
       '/store/relval/CMSSW_3_1_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0005/9ACBB3BC-564D-DE11-9EB0-001617E30E2C.root',
       '/store/relval/CMSSW_3_1_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0005/94373D30-4E4D-DE11-871F-000423D6B42C.root',
       '/store/relval/CMSSW_3_1_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0005/8E5CC1E3-514D-DE11-BD44-001D09F251BD.root',
       '/store/relval/CMSSW_3_1_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0005/8827D200-554D-DE11-B7C0-001617E30D0A.root',
       '/store/relval/CMSSW_3_1_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0005/5CE0A773-574D-DE11-AE55-000423D6CA02.root',
       '/store/relval/CMSSW_3_1_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0005/4A5870B5-5A4D-DE11-9038-001D09F2424A.root',
       '/store/relval/CMSSW_3_1_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0005/405C0068-504D-DE11-B38A-001D09F241B4.root',
       '/store/relval/CMSSW_3_1_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0005/2E8FA2FF-4F4D-DE11-884F-001D09F29849.root',
       '/store/relval/CMSSW_3_1_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_31X_v1/0005/1C8EA945-524D-DE11-A6A7-001D09F24691.root' ] );



secFiles.extend( (
               ) )

# Copied from:
# L1Trigger/Configuration/test/L1EmulatorFromRaw_cfg.py
#
# standard includes
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'STARTUP_31X::All'


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





