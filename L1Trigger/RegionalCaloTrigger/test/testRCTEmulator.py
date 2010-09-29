import FWCore.ParameterSet.Config as cms

process = cms.Process("L1")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.source = cms.Source(
    "PoolSource",
    fileNames = cms.untracked.vstring(
        '/store/data/Run2010B/CommissioningNoBeam/RAW/v1/000/146/527/CA5D36A5-D0C7-DF11-A027-0019B9F4A1D7.root',
#        '/store/data/Run2010B/CommissioningNoBeam/RAW/v1/000/146/529/08B9E39D-D7C7-DF11-9B6F-0030487CAF0E.root',
        
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

# standard includes
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'GR10_P_V9::All'


# unpack raw data
process.load("Configuration.StandardSequences.RawToDigi_Data_cff")


process.rctEmulDigis = cms.EDProducer("L1RCTProducer",
                                  hcalDigis = cms.VInputTag(cms.InputTag("hcalTriggerPrimitiveDigis")),
                                  useDebugTpgScales = cms.bool(True),
                                  useEcal = cms.bool(True),
                                  useHcal = cms.bool(True),
                                  ecalDigis = cms.VInputTag(cms.InputTag("ecalTriggerPrimitiveDigis")),
                                  BunchCrossings = cms.vint32(0)
                              )

process.rctEmulDigis.hcalDigis = cms.VInputTag(cms.InputTag("hcalDigis"))
process.rctEmulDigis.ecalDigis = cms.VInputTag(cms.InputTag("ecalDigis:EcalTriggerPrimitives"))


process.p = cms.Path(
    process.RawToDigi*
    process.rctEmulDigis
)


