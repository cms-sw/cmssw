import FWCore.ParameterSet.Config as cms

process = cms.Process("recHitProd")
# initialize  MessageLogger
process.load("FWCore.MessageLogger.MessageLogger_cfi")

# Read Intercalibrations from offline DB (v2 for 0_8_x)
process.load("Configuration.EcalTB.readIntercalibrationsFromAscii2006_v0_cff")

process.source = cms.Source("PoolSource",
    maxEvents = cms.untracked.int32(-1),
    debugFlag = cms.untracked.bool(False),
    debugVebosity = cms.untracked.uint32(1),
    fileNames = cms.untracked.vstring('file:/u1/meridian/data/h4/2006/h4b.00013247.A.0.0.root')
)

process.ecal2006TBRecHit = cms.EDProducer("EcalRecHitProducer",
    EEuncalibRecHitCollection = cms.string(''),
    uncalibRecHitProducer = cms.string('ecal2006TBWeightUncalibRecHit'),
    EBuncalibRecHitCollection = cms.string('EcalUncalibRecHitsEB'),
    EBrechitCollection = cms.string('EcalRecHitsEB'),
    EErechitCollection = cms.string('')
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('tbhits.root'),
    outputCommands = cms.untracked.vstring('drop *', 
        'keep EcalRecHitsSorted_*_*_*', 
        'keep EcalTBHodoscopeRecInfo_*_*_*', 
        'keep EcalTBEventHeader_*_*_*', 
        'keep EcalTBTDCRecInfo_*_*_*')
)

process.p = cms.Path(process.ecal2006TBRecHit)
process.ep = cms.EndPath(process.out)

