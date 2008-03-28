import FWCore.ParameterSet.Config as cms

process = cms.Process("ProcessOne")
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/archive/ecal/h4tb.pool-cmssw-SM06/h4b.00016481.A.0.0.root'),
    #untracked int32 maxEvents = 10
    isBinary = cms.untracked.bool(True)
)

process.myCnaPackage = cms.EDAnalyzer("EcalCorrelatedNoisePedestalRunAnalyzer",
    verbosity = cms.untracked.uint32(0),
    eventHeaderCollection = cms.string(''),
    #Getting Event Header
    # string eventHeaderProducer = "ecalEBunpacker"
    eventHeaderProducer = cms.string('ecalTBunpack')
)

process.p = cms.Path(process.myCnaPackage)

