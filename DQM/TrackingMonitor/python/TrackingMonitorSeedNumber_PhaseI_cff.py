import FWCore.ParameterSet.Config as cms

#-------------------------------------------------
# Tracking Monitor 
#-------------------------------------------------
import DQM.TrackingMonitor.TrackingMonitorSeed_cfi

Phase1TrackMonStep0 = DQM.TrackingMonitor.TrackingMonitorSeed_cfi.TrackMonSeed.clone()
Phase1TrackMonStep0.TrackProducer = cms.InputTag("generalTracks")
Phase1TrackMonStep0.SeedProducer  = cms.InputTag("initialStepSeeds")
Phase1TrackMonStep0.TCProducer    = cms.InputTag("initialStepTrackCandidates")
Phase1TrackMonStep0.AlgoName      = cms.string('iter0')
Phase1TrackMonStep0.TkSeedSizeBin = cms.int32(100) # could be 50 ?
Phase1TrackMonStep0.TkSeedSizeMax = cms.double(5000)                         
Phase1TrackMonStep0.TkSeedSizeMin = cms.double(0)
Phase1TrackMonStep0.NClusPxBin    = cms.int32(100)
Phase1TrackMonStep0.NClusPxMax    = cms.double(20000)
Phase1TrackMonStep0.ClusterLabels = cms.vstring('Pix')

Phase1TrackMonStep1 = DQM.TrackingMonitor.TrackingMonitorSeed_cfi.TrackMonSeed.clone()
Phase1TrackMonStep1.TrackProducer = cms.InputTag("generalTracks")
Phase1TrackMonStep1.SeedProducer  = cms.InputTag("highPtTripletStepSeeds")
Phase1TrackMonStep1.TCProducer    = cms.InputTag("highPtTripletStepTrackCandidates")
Phase1TrackMonStep1.AlgoName      = cms.string('iter1')
Phase1TrackMonStep1.TkSeedSizeBin = cms.int32(100)
Phase1TrackMonStep1.TkSeedSizeMax = cms.double(30000)                         
Phase1TrackMonStep1.TkSeedSizeMin = cms.double(0)
Phase1TrackMonStep1.NClusPxBin    = cms.int32(100)
Phase1TrackMonStep1.NClusPxMax    = cms.double(20000)
Phase1TrackMonStep1.ClusterLabels = cms.vstring('Pix')

Phase1TrackMonStep2 = DQM.TrackingMonitor.TrackingMonitorSeed_cfi.TrackMonSeed.clone()
Phase1TrackMonStep2.TrackProducer = cms.InputTag("generalTracks")
Phase1TrackMonStep2.SeedProducer  = cms.InputTag("lowPtTripletStepSeeds")
Phase1TrackMonStep2.TCProducer    = cms.InputTag("lowPtTripletStepTrackCandidates")
Phase1TrackMonStep2.AlgoName      = cms.string('iter2')
Phase1TrackMonStep2.TkSeedSizeBin = cms.int32(100)
Phase1TrackMonStep2.TkSeedSizeMax = cms.double(30000)                         
Phase1TrackMonStep2.TkSeedSizeMin = cms.double(0)
Phase1TrackMonStep2.NClusPxBin    = cms.int32(100)
Phase1TrackMonStep2.NClusPxMax    = cms.double(20000)
Phase1TrackMonStep2.ClusterLabels = cms.vstring('Pix')

Phase1TrackMonStep3 = DQM.TrackingMonitor.TrackingMonitorSeed_cfi.TrackMonSeed.clone()
Phase1TrackMonStep3.TrackProducer = cms.InputTag("generalTracks")
Phase1TrackMonStep3.SeedProducer  = cms.InputTag("pixelPairStepSeeds")
Phase1TrackMonStep3.TCProducer    = cms.InputTag("pixelPairStepTrackCandidates")
Phase1TrackMonStep3.AlgoName      = cms.string('iter3')
Phase1TrackMonStep3.TkSeedSizeBin = cms.int32(400)
Phase1TrackMonStep3.TkSeedSizeMax = cms.double(100000)                         
Phase1TrackMonStep3.TkSeedSizeMin = cms.double(0)
Phase1TrackMonStep3.NClusPxBin    = cms.int32(100)
Phase1TrackMonStep3.NClusPxMax    = cms.double(20000)
Phase1TrackMonStep3.ClusterLabels = cms.vstring('Pix')

Phase1TrackMonStep4 = DQM.TrackingMonitor.TrackingMonitorSeed_cfi.TrackMonSeed.clone()
Phase1TrackMonStep4.TrackProducer = cms.InputTag("generalTracks")
Phase1TrackMonStep4.SeedProducer  = cms.InputTag("mixedTripletStepSeeds")
Phase1TrackMonStep4.TCProducer    = cms.InputTag("mixedTripletStepTrackCandidates")
Phase1TrackMonStep4.AlgoName      = cms.string('iter4')
Phase1TrackMonStep4.TkSeedSizeBin = cms.int32(400)
Phase1TrackMonStep4.TkSeedSizeMax = cms.double(200000)                         
Phase1TrackMonStep4.TkSeedSizeMin = cms.double(0)
Phase1TrackMonStep4.NClusStrBin   = cms.int32(500)
Phase1TrackMonStep4.NClusStrMax   = cms.double(100000)
Phase1TrackMonStep4.ClusterLabels = cms.vstring('Tot')

# out of the box
trackMonIterativeTrackingPhaseI = cms.Sequence(
     Phase1TrackMonStep0
    * Phase1TrackMonStep1
    * Phase1TrackMonStep2
    * Phase1TrackMonStep3
    * Phase1TrackMonStep4
)



# all paths
trkmon = cms.Sequence(
      trackMonIterativeTrackingPhaseI
)

