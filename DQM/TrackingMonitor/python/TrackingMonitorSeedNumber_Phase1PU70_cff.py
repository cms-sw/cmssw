import FWCore.ParameterSet.Config as cms

#-------------------------------------------------
# Tracking Monitor 
#-------------------------------------------------
import DQM.TrackingMonitor.TrackingMonitorSeed_cfi

Phase1Pu70TrackMonStep0 = DQM.TrackingMonitor.TrackingMonitorSeed_cfi.TrackMonSeed.clone()
Phase1Pu70TrackMonStep0.TrackProducer = cms.InputTag("generalTracks")
Phase1Pu70TrackMonStep0.SeedProducer  = cms.InputTag("initialStepSeeds")
Phase1Pu70TrackMonStep0.TCProducer    = cms.InputTag("initialStepTrackCandidates")
Phase1Pu70TrackMonStep0.AlgoName      = cms.string('iter0')
Phase1Pu70TrackMonStep0.TkSeedSizeBin = cms.int32(100) # could be 50 ?
Phase1Pu70TrackMonStep0.TkSeedSizeMax = cms.double(5000)                         
Phase1Pu70TrackMonStep0.TkSeedSizeMin = cms.double(0)
Phase1Pu70TrackMonStep0.NClusPxBin    = cms.int32(100)
Phase1Pu70TrackMonStep0.NClusPxMax    = cms.double(20000)
Phase1Pu70TrackMonStep0.ClusterLabels = cms.vstring('Pix')

Phase1Pu70TrackMonStep1 = DQM.TrackingMonitor.TrackingMonitorSeed_cfi.TrackMonSeed.clone()
Phase1Pu70TrackMonStep1.TrackProducer = cms.InputTag("generalTracks")
Phase1Pu70TrackMonStep1.SeedProducer  = cms.InputTag("highPtTripletStepSeeds")
Phase1Pu70TrackMonStep1.TCProducer    = cms.InputTag("highPtTripletStepTrackCandidates")
Phase1Pu70TrackMonStep1.AlgoName      = cms.string('iter1')
Phase1Pu70TrackMonStep1.TkSeedSizeBin = cms.int32(100)
Phase1Pu70TrackMonStep1.TkSeedSizeMax = cms.double(30000)                         
Phase1Pu70TrackMonStep1.TkSeedSizeMin = cms.double(0)
Phase1Pu70TrackMonStep1.NClusPxBin    = cms.int32(100)
Phase1Pu70TrackMonStep1.NClusPxMax    = cms.double(20000)
Phase1Pu70TrackMonStep1.ClusterLabels = cms.vstring('Pix')

Phase1Pu70TrackMonStep2 = DQM.TrackingMonitor.TrackingMonitorSeed_cfi.TrackMonSeed.clone()
Phase1Pu70TrackMonStep2.TrackProducer = cms.InputTag("generalTracks")
Phase1Pu70TrackMonStep2.SeedProducer  = cms.InputTag("lowPtQuadStepSeeds")
Phase1Pu70TrackMonStep2.TCProducer    = cms.InputTag("lowPtQuadStepTrackCandidates")
Phase1Pu70TrackMonStep2.AlgoName      = cms.string('iter2')
Phase1Pu70TrackMonStep2.TkSeedSizeBin = cms.int32(100)
Phase1Pu70TrackMonStep2.TkSeedSizeMax = cms.double(30000)                         
Phase1Pu70TrackMonStep2.TkSeedSizeMin = cms.double(0)
Phase1Pu70TrackMonStep2.NClusPxBin    = cms.int32(100)
Phase1Pu70TrackMonStep2.NClusPxMax    = cms.double(20000)
Phase1Pu70TrackMonStep2.ClusterLabels = cms.vstring('Pix')

Phase1Pu70TrackMonStep3 = DQM.TrackingMonitor.TrackingMonitorSeed_cfi.TrackMonSeed.clone()
Phase1Pu70TrackMonStep3.TrackProducer = cms.InputTag("generalTracks")
Phase1Pu70TrackMonStep3.SeedProducer  = cms.InputTag("lowPtTripletStepSeeds")
Phase1Pu70TrackMonStep3.TCProducer    = cms.InputTag("lowPtTripletStepTrackCandidates")
Phase1Pu70TrackMonStep3.AlgoName      = cms.string('iter3')
Phase1Pu70TrackMonStep3.TkSeedSizeBin = cms.int32(400)
Phase1Pu70TrackMonStep3.TkSeedSizeMax = cms.double(100000)                         
Phase1Pu70TrackMonStep3.TkSeedSizeMin = cms.double(0)
Phase1Pu70TrackMonStep3.NClusPxBin    = cms.int32(100)
Phase1Pu70TrackMonStep3.NClusPxMax    = cms.double(20000)
Phase1Pu70TrackMonStep3.ClusterLabels = cms.vstring('Pix')

Phase1Pu70TrackMonStep4 = DQM.TrackingMonitor.TrackingMonitorSeed_cfi.TrackMonSeed.clone()
Phase1Pu70TrackMonStep4.TrackProducer = cms.InputTag("generalTracks")
Phase1Pu70TrackMonStep4.SeedProducer  = cms.InputTag("detachedQuadStepSeeds")
Phase1Pu70TrackMonStep4.TCProducer    = cms.InputTag("detachedQuadStepTrackCandidates")
Phase1Pu70TrackMonStep4.AlgoName      = cms.string('iter4')
Phase1Pu70TrackMonStep4.TkSeedSizeBin = cms.int32(400)
Phase1Pu70TrackMonStep4.TkSeedSizeMax = cms.double(200000)                         
Phase1Pu70TrackMonStep4.TkSeedSizeMin = cms.double(0)
Phase1Pu70TrackMonStep4.NClusStrBin   = cms.int32(500)
Phase1Pu70TrackMonStep4.NClusStrMax   = cms.double(100000)
Phase1Pu70TrackMonStep4.ClusterLabels = cms.vstring('Tot')

Phase1Pu70TrackMonStep5 = DQM.TrackingMonitor.TrackingMonitorSeed_cfi.TrackMonSeed.clone()
Phase1Pu70TrackMonStep5.TrackProducer = cms.InputTag("generalTracks")
Phase1Pu70TrackMonStep5.SeedProducer  = cms.InputTag("mixedTripletStepSeeds")
Phase1Pu70TrackMonStep5.TCProducer    = cms.InputTag("mixedTripletStepTrackCandidates")
Phase1Pu70TrackMonStep5.AlgoName      = cms.string('iter5')
Phase1Pu70TrackMonStep5.TkSeedSizeBin = cms.int32(400)
Phase1Pu70TrackMonStep5.TkSeedSizeMax = cms.double(200000)                         
Phase1Pu70TrackMonStep5.TkSeedSizeMin = cms.double(0)
Phase1Pu70TrackMonStep5.NClusStrBin   = cms.int32(500)
Phase1Pu70TrackMonStep5.NClusStrMax   = cms.double(100000)
Phase1Pu70TrackMonStep5.ClusterLabels = cms.vstring('Tot')

Phase1Pu70TrackMonStep6 = DQM.TrackingMonitor.TrackingMonitorSeed_cfi.TrackMonSeed.clone()
Phase1Pu70TrackMonStep6.TrackProducer = cms.InputTag("generalTracks")
Phase1Pu70TrackMonStep6.SeedProducer  = cms.InputTag("pixelPairStepSeeds")
Phase1Pu70TrackMonStep6.TCProducer    = cms.InputTag("pixelPairStepTrackCandidates")
Phase1Pu70TrackMonStep6.AlgoName      = cms.string('iter6')
Phase1Pu70TrackMonStep6.TkSeedSizeBin = cms.int32(400)
Phase1Pu70TrackMonStep6.TkSeedSizeMax = cms.double(200000)                         
Phase1Pu70TrackMonStep6.TkSeedSizeMin = cms.double(0)
Phase1Pu70TrackMonStep6.NClusStrBin   = cms.int32(500)
Phase1Pu70TrackMonStep6.NClusStrMax   = cms.double(100000)
Phase1Pu70TrackMonStep6.ClusterLabels = cms.vstring('Pix')

Phase1Pu70TrackMonStep7 = DQM.TrackingMonitor.TrackingMonitorSeed_cfi.TrackMonSeed.clone()
Phase1Pu70TrackMonStep7.TrackProducer = cms.InputTag("generalTracks")
Phase1Pu70TrackMonStep7.SeedProducer  = cms.InputTag("tobTecStepSeeds")
Phase1Pu70TrackMonStep7.TCProducer    = cms.InputTag("tobTecStepTrackCandidates")
Phase1Pu70TrackMonStep7.AlgoName      = cms.string('iter7')
Phase1Pu70TrackMonStep7.TkSeedSizeBin = cms.int32(400)
Phase1Pu70TrackMonStep7.TkSeedSizeMax = cms.double(100000)                         
Phase1Pu70TrackMonStep7.TkSeedSizeMin = cms.double(0)
Phase1Pu70TrackMonStep7.NClusStrBin   = cms.int32(500)
Phase1Pu70TrackMonStep7.NClusStrMax   = cms.double(100000)
Phase1Pu70TrackMonStep7.ClusterLabels = cms.vstring('Strip')

Phase1Pu70TrackMonStep9 = DQM.TrackingMonitor.TrackingMonitorSeed_cfi.TrackMonSeed.clone()
Phase1Pu70TrackMonStep9.TrackProducer = cms.InputTag("generalTracks")
Phase1Pu70TrackMonStep9.SeedProducer  = cms.InputTag("muonSeededSeedsInOut")
Phase1Pu70TrackMonStep9.TCProducer    = cms.InputTag("muonSeededTrackCandidatesInOut")
Phase1Pu70TrackMonStep9.AlgoName      = cms.string('iter9')
Phase1Pu70TrackMonStep9.TkSeedSizeBin = cms.int32(400)
Phase1Pu70TrackMonStep9.TkSeedSizeMax = cms.double(100000)                         
Phase1Pu70TrackMonStep9.TkSeedSizeMin = cms.double(0)
Phase1Pu70TrackMonStep9.NClusStrBin   = cms.int32(500)
Phase1Pu70TrackMonStep9.NClusStrMax   = cms.double(100000)
Phase1Pu70TrackMonStep9.ClusterLabels = cms.vstring('Tot')

Phase1Pu70TrackMonStep10 = DQM.TrackingMonitor.TrackingMonitorSeed_cfi.TrackMonSeed.clone()
Phase1Pu70TrackMonStep10.TrackProducer = cms.InputTag("generalTracks")
Phase1Pu70TrackMonStep10.SeedProducer  = cms.InputTag("muonSeededSeedsOutIn")
Phase1Pu70TrackMonStep10.TCProducer    = cms.InputTag("muonSeededTrackCandidatesOutIn")
Phase1Pu70TrackMonStep10.AlgoName      = cms.string('iter10')
Phase1Pu70TrackMonStep10.TkSeedSizeBin = cms.int32(400)
Phase1Pu70TrackMonStep10.TkSeedSizeMax = cms.double(100000)                         
Phase1Pu70TrackMonStep10.TkSeedSizeMin = cms.double(0)
Phase1Pu70TrackMonStep10.NClusStrBin   = cms.int32(500)
Phase1Pu70TrackMonStep10.NClusStrMax   = cms.double(100000)
Phase1Pu70TrackMonStep10.ClusterLabels = cms.vstring('Tot')

# out of the box
trackMonIterativeTrackingPhase1PU70 = cms.Sequence(
     Phase1Pu70TrackMonStep0
    * Phase1Pu70TrackMonStep1
    * Phase1Pu70TrackMonStep2
    * Phase1Pu70TrackMonStep3
    * Phase1Pu70TrackMonStep4
    * Phase1Pu70TrackMonStep5
    * Phase1Pu70TrackMonStep6
    * Phase1Pu70TrackMonStep7
    * Phase1Pu70TrackMonStep9
    * Phase1Pu70TrackMonStep10
)



# all paths
trkmon = cms.Sequence(
      trackMonIterativeTrackingPhase1PU70
)

