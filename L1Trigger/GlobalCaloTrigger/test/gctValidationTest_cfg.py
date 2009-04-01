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
 
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100) )
readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring() 
process.source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)
readFiles.extend( [
       '/store/relval/CMSSW_3_1_0_pre4/RelValTTbar/GEN-SIM-RECO/IDEAL_30X_v1/0003/00E48100-3A16-DE11-A693-001617DBCF6A.root',
       '/store/relval/CMSSW_3_1_0_pre4/RelValTTbar/GEN-SIM-RECO/IDEAL_30X_v1/0003/12C01897-4616-DE11-8AA7-000423D98B5C.root',
       '/store/relval/CMSSW_3_1_0_pre4/RelValTTbar/GEN-SIM-RECO/IDEAL_30X_v1/0003/289FC85A-4216-DE11-ACEE-000423D98844.root',
       '/store/relval/CMSSW_3_1_0_pre4/RelValTTbar/GEN-SIM-RECO/IDEAL_30X_v1/0003/6ED9476F-4C16-DE11-8BFC-001617C3B76A.root',
       '/store/relval/CMSSW_3_1_0_pre4/RelValTTbar/GEN-SIM-RECO/IDEAL_30X_v1/0003/76E8D7B2-5216-DE11-8A7A-000423D174FE.root',
       '/store/relval/CMSSW_3_1_0_pre4/RelValTTbar/GEN-SIM-RECO/IDEAL_30X_v1/0003/B0D94AFE-3616-DE11-BFD5-000423D9880C.root',
       '/store/relval/CMSSW_3_1_0_pre4/RelValTTbar/GEN-SIM-RECO/IDEAL_30X_v1/0003/BCE77A07-AC16-DE11-80B9-000423D986A8.root' ] );


secFiles.extend( (
               ) )

# Copied from:
# L1Trigger/Configuration/test/L1EmulatorFromRaw_cfg.py
#
# standard includes
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'IDEAL_30X::All'


# unpack raw data
process.load("Configuration.StandardSequences.RawToDigi_cff")

# run trigger primitive generation on unpacked digis, then central L1
process.load("L1Trigger.Configuration.CaloTriggerPrimitives_cff")
process.load("L1Trigger.Configuration.SimL1Emulator_cff")
process.simEcalTriggerPrimitiveDigis.Label = 'ecalDigis'
process.simHcalTriggerPrimitiveDigis.inputLabel = 'hcalDigis'
process.simDtTriggerPrimitiveDigis.digiTag = 'muonDTDigis'
process.simCscTriggerPrimitiveDigis.CSCComparatorDigiProducer = cms.InputTag("muonCSCDigis","MuonCSCComparatorDigi")
process.simCscTriggerPrimitiveDigis.CSCWireDigiProducer = cms.InputTag("muonCSCDigis","MuonCSCWireDigi")
process.simRpcTriggerDigis.label = 'muonRPCDigis'

# L1 configuration
process.load('L1Trigger.Configuration.L1DummyConfig_cff')

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





