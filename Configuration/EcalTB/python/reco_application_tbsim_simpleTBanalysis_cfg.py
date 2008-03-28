import FWCore.ParameterSet.Config as cms

process = cms.Process("simpleAnalysis")
# initialize  MessageLogger
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.source = cms.Source("PoolSource",
    maxEvents = cms.untracked.int32(-1),
    fileNames = cms.untracked.vstring('file:ECALH4TB_detsim_hits.root')
)

process.simpleTBanalysis = cms.EDAnalyzer("EcalSimpleTBAnalyzer",
    rootfile = cms.untracked.string('ecal_tbsim_SimpleTBAnalysis.root'),
    eventHeaderProducer = cms.string('SimEcalEventHeader'),
    hitProducer = cms.string('ecalTBSimWeightUncalibRecHit'),
    digiCollection = cms.string(''),
    tdcRecInfoCollection = cms.string('EcalTBTDCRecInfo'),
    digiProducer = cms.string('ecaldigi'),
    hitCollection = cms.string('EcalUncalibRecHitsEB'),
    hodoRecInfoProducer = cms.string('ecalTBSimHodoscopeReconstructor'),
    eventHeaderCollection = cms.string(''),
    hodoRecInfoCollection = cms.string('EcalTBHodoscopeRecInfo'),
    tdcRecInfoProducer = cms.string('ecalTBSimTDCReconstructor')
)

process.p = cms.Path(process.simpleTBanalysis)

