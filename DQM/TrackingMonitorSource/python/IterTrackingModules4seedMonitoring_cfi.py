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

seedInputTag     ['initialStep'] = cms.InputTag("initialStepSeeds")
trackCandInputTag['initialStep'] = cms.InputTag("initialStepSeeds")
trackSeedSizeBin ['initialStep'] = cms.int32(100) # could be 50 ? 
trackSeedSizeMin ['initialStep'] = cms.double(0)                  
trackSeedSizeMax ['initialStep'] = cms.double(5000)               
clusterLabel     ['initialStep'] = cms.vstring('Pix')
clusterBin       ['initialStep'] = cms.int32(100)
clusterMax       ['initialStep'] = cms.double(20000)

seedInputTag     ['lowPtTripletStep'] = cms.InputTag("lowPtTripletStepSeeds")
trackCandInputTag['lowPtTripletStep'] = cms.InputTag("lowPtTripletStepTrackCandidates")
trackSeedSizeBin ['lowPtTripletStep'] = cms.int32(100)
trackSeedSizeMin ['lowPtTripletStep'] = cms.double(0)
trackSeedSizeMax ['lowPtTripletStep'] = cms.double(30000)                         
clusterLabel     ['lowPtTripletStep'] = cms.vstring('Pix')
clusterBin       ['lowPtTripletStep'] = cms.int32(100)
clusterMax       ['lowPtTripletStep'] = cms.double(20000)

seedInputTag     ['pixelPairStep'] = cms.InputTag("pixelPairStepSeeds")
trackCandInputTag['pixelPairStep'] = cms.InputTag("pixelPairStepTrackCandidates")
trackSeedSizeBin ['pixelPairStep'] = cms.int32(400)
trackSeedSizeMin ['pixelPairStep'] = cms.double(0)
trackSeedSizeMax ['pixelPairStep'] = cms.double(100000)                         
clusterLabel     ['pixelPairStep'] = cms.vstring('Pix')
clusterBin       ['pixelPairStep'] = cms.int32(100)
clusterMax       ['pixelPairStep'] = cms.double(20000)

seedInputTag     ['detachedTripletStep'] = cms.InputTag("detachedTripletStepSeeds")
trackCandInputTag['detachedTripletStep'] = cms.InputTag("detachedTripletStepTrackCandidates")
trackSeedSizeBin ['detachedTripletStep'] = cms.int32(100)
trackSeedSizeMin ['detachedTripletStep'] = cms.double(0)
trackSeedSizeMax ['detachedTripletStep'] = cms.double(30000)                         
clusterLabel     ['detachedTripletStep'] = cms.vstring('Pix')
clusterBin       ['detachedTripletStep'] = cms.int32(100)
clusterMax       ['detachedTripletStep'] = cms.double(20000)

seedInputTag     ['mixedTripletStep'] = cms.InputTag("mixedTripletStepSeeds")
trackCandInputTag['mixedTripletStep'] = cms.InputTag("mixedTripletStepTrackCandidates")
trackSeedSizeBin ['mixedTripletStep'] = cms.int32(400)
trackSeedSizeMin ['mixedTripletStep'] = cms.double(0)
trackSeedSizeMax ['mixedTripletStep'] = cms.double(200000)                         
clusterLabel     ['mixedTripletStep'] = cms.vstring('Tot')
clusterBin       ['mixedTripletStep'] = cms.int32(500)
clusterMax       ['mixedTripletStep'] = cms.double(100000)

seedInputTag     ['pixelLessStep'] = cms.InputTag("pixelLessStepSeeds")
trackCandInputTag['pixelLessStep'] = cms.InputTag("pixelLessStepTrackCandidates")
trackSeedSizeBin ['pixelLessStep'] = cms.int32(400)
trackSeedSizeMin ['pixelLessStep'] = cms.double(0)
trackSeedSizeMax ['pixelLessStep'] = cms.double(200000)                         
clusterLabel     ['pixelLessStep'] = cms.vstring('Strip')
clusterBin       ['pixelLessStep'] = cms.int32(500)
clusterMax       ['pixelLessStep'] = cms.double(100000)

seedInputTag     ['tobTecStep'] = cms.InputTag("tobTecStepSeeds")
trackCandInputTag['tobTecStep'] = cms.InputTag("tobTecStepTrackCandidates")
trackSeedSizeBin ['tobTecStep'] = cms.int32(400)
trackSeedSizeMin ['tobTecStep'] = cms.double(0)
trackSeedSizeMax ['tobTecStep'] = cms.double(100000)                         
clusterLabel     ['tobTecStep'] = cms.vstring('Strip')
clusterBin       ['tobTecStep'] = cms.int32(500)
clusterMax       ['tobTecStep'] = cms.double(100000)

seedInputTag     ['muonSeededStepInOut'] = cms.InputTag("muonSeededSeedsInOut")
trackCandInputTag['muonSeededStepInOut'] = cms.InputTag("muonSeededTrackCandidatesInOut")
trackSeedSizeBin ['muonSeededStepInOut'] = cms.int32(15)
trackSeedSizeMin ['muonSeededStepInOut'] = cms.double(-0.5)                         
trackSeedSizeMax ['muonSeededStepInOut'] = cms.double(14.5)
clusterLabel     ['muonSeededStepInOut'] = cms.vstring('Strip')
clusterBin       ['muonSeededStepInOut'] = cms.int32(500)
clusterMax       ['muonSeededStepInOut'] = cms.double(100000)

seedInputTag     ['muonSeededStepOutIn'] = cms.InputTag("muonSeededSeedsOutIn")
trackCandInputTag['muonSeededStepOutIn'] = cms.InputTag("muonSeededTrackCandidatesOutIn")
trackSeedSizeBin ['muonSeededStepOutIn'] = cms.int32(15)
trackSeedSizeMin ['muonSeededStepOutIn'] = cms.double(-0.5)                         
trackSeedSizeMax ['muonSeededStepOutIn'] = cms.double(14.5)
clusterLabel     ['muonSeededStepOutIn'] = cms.vstring('Strip')
clusterBin       ['muonSeededStepOutIn'] = cms.int32(500)
clusterMax       ['muonSeededStepOutIn'] = cms.double(100000)

selectedIterTrackingStep.extend( ['initialStep']  )
selectedIterTrackingStep.extend( ['lowPtTripletStep']  )
selectedIterTrackingStep.extend( ['pixelPairStep']  )
selectedIterTrackingStep.extend( ['detachedTripletStep']  )
selectedIterTrackingStep.extend( ['mixedTripletStep']  )
selectedIterTrackingStep.extend( ['pixelLessStep']  )
selectedIterTrackingStep.extend( ['tobTecStep']  )
selectedIterTrackingStep.extend( ['muonSeededStepInOut']  )
selectedIterTrackingStep.extend( ['muonSeededStepOutIn'] )
