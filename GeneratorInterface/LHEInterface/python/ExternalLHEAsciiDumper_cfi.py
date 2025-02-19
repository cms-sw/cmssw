import FWCore.ParameterSet.Config as cms

externalLHEAsciiDumper = cms.EDAnalyzer('ExternalLHEAsciiDumper',
    lheProduct = cms.InputTag('externalLHEProducer','LHEScriptOutput'),
    lheFileName = cms.string('ascii_dump.lhe')
)
