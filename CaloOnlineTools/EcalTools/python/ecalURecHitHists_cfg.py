import FWCore.ParameterSet.Config as cms

process = cms.Process("GETUNCALRECHIT")
process.load("EventFilter.EcalRawToDigi.EcalUnpackerMapping_cfi")
process.load("EventFilter.EcalRawToDigi.EcalUnpackerData_cfi")

process.load("Geometry.EcalMapping.EcalMapping_cfi")
process.load("Geometry.EcalMapping.EcalMappingRecord_cfi")

import RecoLocalCalo.EcalRecProducers.ecalFixedAlphaBetaFitUncalibRecHit_cfi
process.ecalUncalibHit = RecoLocalCalo.EcalRecProducers.ecalFixedAlphaBetaFitUncalibRecHit_cfi.ecalFixedAlphaBetaFitUncalibRecHit.clone()

process.load("CaloOnlineTools.EcalTools.ecalURecHitHists_cfi")

process.source = cms.Source("PoolSource",
    skipEvents = cms.untracked.uint32(0),
    #untracked vstring fileNames = {'file:/data/scooper/data/grea07/40792E58-B757-DC11-8AB2-001617E30F46.root'}
    #fileNames = cms.untracked.vstring('file:/data/scooper/data/gren07/P5_Co.00029485.A.0.0.root')
    fileNames = cms.untracked.vstring('file:/data/scooper/data/cruzet3/7E738216-584D-DD11-9209-000423D6AF24.root')
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

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        enable = cms.untracked.bool(True),
        threshold = cms.untracked.string('INFO')
    ),
    suppressInfo = cms.untracked.vstring('ecalEBunpacker')
)

process.p = cms.Path(process.ecalEBunpacker*process.ecalUncalibHit*process.ecalURecHitHists)
process.ecalUncalibHit.EBdigiCollection = cms.InputTag("ecalEBunpacker","ebDigis")
process.ecalUncalibHit.EEdigiCollection = cms.InputTag("ecalEBunpacker","eeDigis")

