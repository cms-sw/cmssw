import FWCore.ParameterSet.Config as cms

process = cms.Process("GETUNCALRECHIT")
#untracked PSet maxEvents = {untracked int32 input = -1}
process.load("EventFilter.EcalRawToDigiDev.EcalUnpackerMapping_cfi")

process.load("EventFilter.EcalRawToDigiDev.EcalUnpackerData_cfi")

# for neighbor navigation
process.load("Geometry.CaloEventSetup.CaloTopology_cfi")

process.load("Geometry.EcalCommonData.EcalOnly_cfi")

process.load("Geometry.CaloEventSetup.CaloGeometry_cff")

import RecoLocalCalo.EcalRecProducers.ecalMaxSampleUncalibRecHit_cfi
process.ecalUncalibHit = RecoLocalCalo.EcalRecProducers.ecalMaxSampleUncalibRecHit_cfi.ecalMaxSampleUncalibRecHit.clone()
process.load("CaloOnlineTools.EcalTools.ecalMipGraphs_cfi")

process.source = cms.Source("PoolSource",
    skipEvents = cms.untracked.uint32(0),
    #untracked vstring fileNames = {'file:/data/scooper/data/grea07/40792E58-B757-DC11-8AB2-001617E30F46.root'}
    fileNames = cms.untracked.vstring('file:/data/scooper/data/gren07/P5_Co.00029485.A.0.0.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(150)
)
process.src1 = cms.ESSource("EcalTrivialConditionRetriever",
    jittWeights = cms.untracked.vdouble(0.04, 0.04, 0.04, 0.0, 1.32, 
        -0.05, -0.5, -0.5, -0.4, 0.0),
    pedWeights = cms.untracked.vdouble(0.333, 0.333, 0.333, 0.0, 0.0, 
        0.0, 0.0, 0.0, 0.0, 0.0),
    amplWeights = cms.untracked.vdouble(-0.333, -0.333, -0.333, 0.0, 0.0, 
        1.0, 0.0, 0.0, 0.0, 0.0)
)

process.dumpEv = cms.EDAnalyzer("EventContentAnalyzer")

process.MessageLogger = cms.Service("MessageLogger",
    suppressInfo = cms.untracked.vstring('ecalEBunpacker'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO')
    ),
    categories = cms.untracked.vstring('EcalMipGraphs'),
    destinations = cms.untracked.vstring('cout')
)

process.p = cms.Path(process.ecalEBunpacker*process.ecalUncalibHit*process.ecalMipGraphs)
process.ecalUncalibHit.EBdigiCollection = cms.InputTag("ecalEBunpacker","ebDigis")
process.ecalUncalibHit.EEdigiCollection = cms.InputTag("ecalEBunpacker","eeDigis")
process.ecalMipGraphs.seedCry = 18637

