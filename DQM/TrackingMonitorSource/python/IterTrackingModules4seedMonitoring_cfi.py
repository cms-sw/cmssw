import FWCore.ParameterSet.Config as cms

selectedIterTrackingStep = []

seedInputTag      = {}
trackCandInputTag = {}
trackSeedSizeBin  = {}
trackSeedSizeMin  = {}
trackSeedSizeMax  = {}
clusterLabel      = {}
clusterBin        = {}
clusterMax        = {}

seedInputTag     ['iter0'] = cms.InputTag("initialStepSeeds")
trackCandInputTag['iter0'] = cms.InputTag("initialStepSeeds")
trackSeedSizeBin ['iter0'] = cms.int32(100) # could be 50 ? 
trackSeedSizeMin ['iter0'] = cms.double(0)                  
trackSeedSizeMax ['iter0'] = cms.double(5000)               
clusterLabel     ['iter0'] = cms.vstring('Pix')
clusterBin       ['iter0'] = cms.int32(100)
clusterMax       ['iter0'] = cms.double(20000)

seedInputTag     ['iter1'] = cms.InputTag("lowPtTripletStepSeeds")
trackCandInputTag['iter1'] = cms.InputTag("lowPtTripletStepTrackCandidates")
trackSeedSizeBin ['iter1'] = cms.int32(100)
trackSeedSizeMin ['iter1'] = cms.double(0)
trackSeedSizeMax ['iter1'] = cms.double(30000)                         
clusterLabel     ['iter1'] = cms.vstring('Pix')
clusterBin       ['iter1'] = cms.int32(100)
clusterMax       ['iter1'] = cms.double(20000)

seedInputTag     ['iter2'] = cms.InputTag("pixelPairStepSeeds")
trackCandInputTag['iter2'] = cms.InputTag("pixelPairStepTrackCandidates")
trackSeedSizeBin ['iter2'] = cms.int32(400)
trackSeedSizeMin ['iter2'] = cms.double(0)
trackSeedSizeMax ['iter2'] = cms.double(100000)                         
clusterLabel     ['iter2'] = cms.vstring('Pix')
clusterBin       ['iter2'] = cms.int32(100)
clusterMax       ['iter2'] = cms.double(20000)

seedInputTag     ['iter3'] = cms.InputTag("detachedTripletStepSeeds")
trackCandInputTag['iter3'] = cms.InputTag("detachedTripletStepTrackCandidates")
trackSeedSizeBin ['iter3'] = cms.int32(100)
trackSeedSizeMin ['iter3'] = cms.double(0)
trackSeedSizeMax ['iter3'] = cms.double(30000)                         
clusterLabel     ['iter3'] = cms.vstring('Pix')
clusterBin       ['iter3'] = cms.int32(100)
clusterMax       ['iter3'] = cms.double(20000)

seedInputTag     ['iter4'] = cms.InputTag("mixedTripletStepSeeds")
trackCandInputTag['iter4'] = cms.InputTag("mixedTripletStepTrackCandidates")
trackSeedSizeBin ['iter4'] = cms.int32(400)
trackSeedSizeMin ['iter4'] = cms.double(0)
trackSeedSizeMax ['iter4'] = cms.double(200000)                         
clusterLabel     ['iter4'] = cms.vstring('Tot')
clusterBin       ['iter4'] = cms.int32(500)
clusterMax       ['iter4'] = cms.double(100000)

seedInputTag     ['iter5'] = cms.InputTag("pixelLessStepSeeds")
trackCandInputTag['iter5'] = cms.InputTag("pixelLessStepTrackCandidates")
trackSeedSizeBin ['iter5'] = cms.int32(400)
trackSeedSizeMin ['iter5'] = cms.double(0)
trackSeedSizeMax ['iter5'] = cms.double(200000)                         
clusterLabel     ['iter5'] = cms.vstring('Strip')
clusterBin       ['iter5'] = cms.int32(500)
clusterMax       ['iter5'] = cms.double(100000)

seedInputTag     ['iter6'] = cms.InputTag("tobTecStepSeeds")
trackCandInputTag['iter6'] = cms.InputTag("tobTecStepTrackCandidates")
trackSeedSizeBin ['iter6'] = cms.int32(400)
trackSeedSizeMin ['iter6'] = cms.double(0)
trackSeedSizeMax ['iter6'] = cms.double(100000)                         
clusterLabel     ['iter6'] = cms.vstring('Strip')
clusterBin       ['iter6'] = cms.int32(500)
clusterMax       ['iter6'] = cms.double(100000)

seedInputTag     ['iter9'] = cms.InputTag("muonSeededSeedsInOut")
trackCandInputTag['iter9'] = cms.InputTag("muonSeededTrackCandidatesInOut")
trackSeedSizeBin ['iter9'] = cms.int32(15)
trackSeedSizeMin ['iter9'] = cms.double(-0.5)                         
trackSeedSizeMax ['iter9'] = cms.double(14.5)
clusterLabel     ['iter9'] = cms.vstring('Strip')
clusterBin       ['iter9'] = cms.int32(500)
clusterMax       ['iter9'] = cms.double(100000)

seedInputTag     ['iter10'] = cms.InputTag("muonSeededSeedsOutIn")
trackCandInputTag['iter10'] = cms.InputTag("muonSeededTrackCandidatesOutIn")
trackSeedSizeBin ['iter10'] = cms.int32(15)
trackSeedSizeMin ['iter10'] = cms.double(-0.5)                         
trackSeedSizeMax ['iter10'] = cms.double(14.5)
clusterLabel     ['iter10'] = cms.vstring('Strip')
clusterBin       ['iter10'] = cms.int32(500)
clusterMax       ['iter10'] = cms.double(100000)

selectedIterTrackingStep.extend( ['iter0']  )
selectedIterTrackingStep.extend( ['iter1']  )
selectedIterTrackingStep.extend( ['iter2']  )
selectedIterTrackingStep.extend( ['iter3']  )
selectedIterTrackingStep.extend( ['iter4']  )
selectedIterTrackingStep.extend( ['iter5']  )
selectedIterTrackingStep.extend( ['iter6']  )
selectedIterTrackingStep.extend( ['iter9']  )
selectedIterTrackingStep.extend( ['iter10'] )
