import FWCore.ParameterSet.Config as cms

process = cms.Process("L1")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.source = cms.Source(
    "PoolSource",
    fileNames = cms.untracked.vstring(
    'file:/scratch/bachtis/test.root'
       
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(3000)
)

# standard includes
process.load("Configuration.StandardSequences.GeometryDB_cff")
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

process.rctDigis.ecalDigis =cms.VInputTag( cms.InputTag('simEcalTriggerPrimitiveDigis'))
process.rctDigis.hcalDigis = cms.VInputTag(cms.InputTag('simHcalTriggerPrimitiveDigis'))

process.L1Analysis = cms.EDAnalyzer("L1RCTTestAnalyzer",
    hcalDigisLabel = cms.InputTag("simHcalTriggerPrimitiveDigis"),
    showEmCands = cms.untracked.bool(False),
    ecalDigisLabel = cms.InputTag("simEcalTriggerPrimitiveDigis"),
    rctDigisLabel = cms.InputTag("rctDigis"),
    showRegionSums = cms.untracked.bool(False)
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


