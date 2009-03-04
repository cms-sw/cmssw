import FWCore.ParameterSet.Config as cms

process = cms.Process("L1")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.source = cms.Source(
    "PoolSource",
    fileNames = cms.untracked.vstring(
       '/store/relval/CMSSW_3_1_0_pre2/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0000/04A55351-8803-DE11-A538-001617E30D0A.root',
               '/store/relval/CMSSW_3_1_0_pre2/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0000/0604D932-8903-DE11-B939-001D09F282F5.root',
               '/store/relval/CMSSW_3_1_0_pre2/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0000/4E7E148B-8803-DE11-80C3-000423D99660.root',
               '/store/relval/CMSSW_3_1_0_pre2/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0000/6CAF4DC2-8703-DE11-83BB-000423D6B358.root',
               '/store/relval/CMSSW_3_1_0_pre2/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0000/6E33757C-8903-DE11-8B80-000423D952C0.root',
               '/store/relval/CMSSW_3_1_0_pre2/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0000/80EBB40E-8903-DE11-9A25-0019B9F6C674.root',
               '/store/relval/CMSSW_3_1_0_pre2/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0000/8AD23071-8703-DE11-A266-000423D99AAE.root',
               '/store/relval/CMSSW_3_1_0_pre2/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0000/92746211-8903-DE11-938A-001D09F24FEC.root',
               '/store/relval/CMSSW_3_1_0_pre2/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0000/B0ABF9A8-8703-DE11-BEAB-000423D9939C.root',
               '/store/relval/CMSSW_3_1_0_pre2/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0000/B8D31D12-8903-DE11-9483-001D09F24D4E.root',
               '/store/relval/CMSSW_3_1_0_pre2/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0000/CC7DB17F-8803-DE11-A2E1-000423D6C8E6.root',
               '/store/relval/CMSSW_3_1_0_pre2/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0001/22CAF860-DB03-DE11-842E-000423D9989E.root'
       
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(3000)
)

# standard includes
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = "IDEAL_30X::All"


# unpack raw data
process.load("Configuration.StandardSequences.RawToDigi_cff")

# run trigger primitive generation on unpacked digis, then central L1
process.load("L1Trigger.Configuration.CaloTriggerPrimitives_cff")

process.simEcalTriggerPrimitiveDigis.Label = 'ecalDigis'
process.simHcalTriggerPrimitiveDigis.inputLabel = 'hcalDigis'


process.load("L1TriggerConfig.RCTConfigProducers.L1RCTConfig_cff")
process.load("L1Trigger.RegionalCaloTrigger.rctDigis_cfi")

process.rctDigis.ecalDigisLabel = 'simEcalTriggerPrimitiveDigis'
process.rctDigis.hcalDigisLabel = 'simHcalTriggerPrimitiveDigis'

process.L1Analysis = cms.EDAnalyzer("L1RCTTestAnalyzer",
    hcalDigisLabel = cms.InputTag("simHcalTriggerPrimitiveDigis"),
    showEmCands = cms.untracked.bool(True),
    ecalDigisLabel = cms.InputTag("simEcalTriggerPrimitiveDigis"),
    rctDigisLabel = cms.InputTag("rctDigis"),
    showRegionSums = cms.untracked.bool(True)
)


process.TFileService = cms.Service("TFileService",
                                 fileName = cms.string("histo.root"),
                                 closeFileFast = cms.untracked.bool(True)
                             )



# L1 configuration
process.load('L1Trigger.Configuration.L1DummyConfig_cff')


process.rctDigis.useCorrectionsLindsey = cms.bool(False)

process.p = cms.Path(
    process.ecalDigis
    *process.hcalDigis
    *process.rctDigis
    *process.L1Analysis
)


