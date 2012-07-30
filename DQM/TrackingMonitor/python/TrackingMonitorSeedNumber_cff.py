import FWCore.ParameterSet.Config as cms

#-------------------------------------------------
# Tracking Monitor 
#-------------------------------------------------
import DQM.TrackingMonitor.TrackingMonitorSeed_cfi

TrackMonStep0 = DQM.TrackingMonitor.TrackingMonitorSeed_cfi.TrackMonSeed.clone()
TrackMonStep0.TrackProducer = cms.InputTag("generalTracks")
TrackMonStep0.SeedProducer  = cms.InputTag("initialStepSeeds")
TrackMonStep0.TCProducer    = cms.InputTag("initialStepTrackCandidates")
TrackMonStep0.AlgoName      = cms.string('iter0')

TrackMonStep0.TkSeedSizeBin = cms.int32(100)
TrackMonStep0.TkSeedSizeMax = cms.double(5000)                         
TrackMonStep0.TkSeedSizeMin = cms.double(0)
TrackMonStep0.NClusPxBin = cms.int32(100)
TrackMonStep0.NClusPxMax = cms.double(20000)
TrackMonStep0.ClusterLabels = cms.vstring('Pix')

TrackMonStep1 = DQM.TrackingMonitor.TrackingMonitorSeed_cfi.TrackMonSeed.clone()
TrackMonStep1.TrackProducer = cms.InputTag("generalTracks")
TrackMonStep1.SeedProducer  = cms.InputTag("lowPtTripletStepSeeds")
TrackMonStep1.TCProducer    = cms.InputTag("lowPtTripletStepTrackCandidates")
TrackMonStep1.AlgoName      = cms.string('iter1')
TrackMonStep1.TkSeedSizeBin = cms.int32(200)
TrackMonStep1.TkSeedSizeMax = cms.double(10000)                         
TrackMonStep1.TkSeedSizeMin = cms.double(0)
TrackMonStep1.ClusterLabels = cms.vstring('Pix')

TrackMonStep2 = DQM.TrackingMonitor.TrackingMonitorSeed_cfi.TrackMonSeed.clone()
TrackMonStep2.TrackProducer = cms.InputTag("generalTracks")
TrackMonStep2.SeedProducer  = cms.InputTag("pixelPairStepSeeds")
TrackMonStep2.TCProducer    = cms.InputTag("pixelPairStepTrackCandidates")
TrackMonStep2.AlgoName      = cms.string('iter2')
TrackMonStep2.TkSeedSizeBin = cms.int32(400)
TrackMonStep2.TkSeedSizeMax = cms.double(100000)                         
TrackMonStep2.TkSeedSizeMin = cms.double(0)
TrackMonStep2.ClusterLabels = cms.vstring('Pix')

TrackMonStep3 = DQM.TrackingMonitor.TrackingMonitorSeed_cfi.TrackMonSeed.clone()
TrackMonStep3.TrackProducer = cms.InputTag("generalTracks")
TrackMonStep3.SeedProducer  = cms.InputTag("detachedTripletStepSeeds")
TrackMonStep3.TCProducer    = cms.InputTag("detachedTripletStepTrackCandidates")
TrackMonStep3.AlgoName      = cms.string('iter3')
TrackMonStep3.TkSeedSizeBin = cms.int32(200)
TrackMonStep3.TkSeedSizeMax = cms.double(20000)                         
TrackMonStep3.TkSeedSizeMin = cms.double(0)
TrackMonStep3.ClusterLabels = cms.vstring('Pix')

TrackMonStep4 = DQM.TrackingMonitor.TrackingMonitorSeed_cfi.TrackMonSeed.clone()
TrackMonStep4.TrackProducer = cms.InputTag("generalTracks")
TrackMonStep4.SeedProducer  = cms.InputTag("mixedTripletStepSeeds")
TrackMonStep4.TCProducer    = cms.InputTag("mixedTripletStepTrackCandidates")
TrackMonStep4.AlgoName      = cms.string('iter4')
TrackMonStep4.TkSeedSizeBin = cms.int32(400)
TrackMonStep4.TkSeedSizeMax = cms.double(200000)                         
TrackMonStep4.TkSeedSizeMin = cms.double(0)
TrackMonStep4.ClusterLabels = cms.vstring('Tot')

TrackMonStep5 = DQM.TrackingMonitor.TrackingMonitorSeed_cfi.TrackMonSeed.clone()
TrackMonStep5.TrackProducer = cms.InputTag("generalTracks")
TrackMonStep5.SeedProducer  = cms.InputTag("pixelLessStepSeeds")
TrackMonStep5.TCProducer    = cms.InputTag("pixelLessStepTrackCandidates")
TrackMonStep5.AlgoName      = cms.string('iter5')
TrackMonStep5.TkSeedSizeBin = cms.int32(400)
TrackMonStep5.TkSeedSizeMax = cms.double(200000)                         
TrackMonStep5.TkSeedSizeMin = cms.double(0)
TrackMonStep5.ClusterLabels = cms.vstring('Strip')

TrackMonStep6 = DQM.TrackingMonitor.TrackingMonitorSeed_cfi.TrackMonSeed.clone()
TrackMonStep6.TrackProducer = cms.InputTag("generalTracks")
TrackMonStep6.SeedProducer  = cms.InputTag("tobTecStepSeeds")
TrackMonStep6.TCProducer    = cms.InputTag("tobTecStepTrackCandidates")
TrackMonStep6.AlgoName      = cms.string('iter6')
TrackMonStep6.TkSeedSizeBin = cms.int32(400)
TrackMonStep6.TkSeedSizeMax = cms.double(100000)                         
TrackMonStep6.TkSeedSizeMin = cms.double(0)
TrackMonStep6.ClusterLabels = cms.vstring('Strip')

# out of the box
trkmonootb = cms.Sequence(
     TrackMonStep0
    * TrackMonStep1
    * TrackMonStep2
    * TrackMonStep3
    * TrackMonStep4
    * TrackMonStep5
    * TrackMonStep6
)



# all paths
trkmon = cms.Sequence(
      trkmonootb
)

