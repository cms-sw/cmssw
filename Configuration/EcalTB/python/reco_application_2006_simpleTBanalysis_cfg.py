import FWCore.ParameterSet.Config as cms

process = cms.Process("simpleAnalysis")
# initialize  MessageLogger
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.source = cms.Source("PoolSource",
    maxEvents = cms.untracked.int32(500),
    #	untracked vstring fileNames = {'rfio:/castor/cern.ch/cms/archive/ecal/h4tb.pool-cmssw-SM12/h4b.00015217.A.0.0.root'}
    fileNames = cms.untracked.vstring('file:hits.root')
)

process.simpleTBanalysis = cms.EDAnalyzer("EcalSimpleTBAnalyzer",
    rootfile = cms.untracked.string('ecalSimpleTBAnalysis.root'),
    eventHeaderProducer = cms.string('ecalTBunpack'),
    hitProducer = cms.string('ecal2006TBWeightUncalibRecHit'),
    digiCollection = cms.string(''),
    tdcRecInfoCollection = cms.string('EcalTBTDCRecInfo'),
    digiProducer = cms.string('ecalTBunpack'),
    hitCollection = cms.string('EcalUncalibRecHitsEB'),
    hodoRecInfoProducer = cms.string('ecal2006TBHodoscopeReconstructor'),
    eventHeaderCollection = cms.string(''),
    hodoRecInfoCollection = cms.string('EcalTBHodoscopeRecInfo'),
    tdcRecInfoProducer = cms.string('ecal2006TBTDCReconstructor')
)

process.p = cms.Path(process.simpleTBanalysis)

