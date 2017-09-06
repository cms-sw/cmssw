import FWCore.ParameterSet.Config as cms
import RecoTracker.IterativeTracking.iterativeTkConfig as _cfg

seedInputTag      = {}
trackCandInputTag = {}
trackSeedSizeBin  = {}
trackSeedSizeMin  = {}
trackSeedSizeMax  = {}
TCSizeMax         = {}
clusterLabel      = {}
clusterBin        = {}
clusterMax        = {}
regionLabel       = {}
regionCandidateLabel = {}

seedInputTag     ['initialStep'] = cms.InputTag("initialStepSeeds")
trackCandInputTag['initialStep'] = cms.InputTag("initialStepTrackCandidates")
trackSeedSizeBin ['initialStep'] = cms.int32(100) # could be 50 ? 
trackSeedSizeMin ['initialStep'] = cms.double(0)                  
trackSeedSizeMax ['initialStep'] = cms.double(5000)               
clusterLabel     ['initialStep'] = cms.vstring('Pix')
clusterBin       ['initialStep'] = cms.int32(100)
clusterMax       ['initialStep'] = cms.double(20000)

seedInputTag     ['highPtTripletStep'] = cms.InputTag("highPtTripletStepSeeds")
trackCandInputTag['highPtTripletStep'] = cms.InputTag("highPtTripletStepTrackCandidates")
trackSeedSizeBin ['highPtTripletStep'] = cms.int32(100)
trackSeedSizeMin ['highPtTripletStep'] = cms.double(0)
trackSeedSizeMax ['highPtTripletStep'] = cms.double(30000)
clusterLabel     ['highPtTripletStep'] = cms.vstring('Pix')
clusterBin       ['highPtTripletStep'] = cms.int32(100)
clusterMax       ['highPtTripletStep'] = cms.double(20000)

seedInputTag     ['lowPtQuadStep'] = cms.InputTag("lowPtQuadStepSeeds")
trackCandInputTag['lowPtQuadStep'] = cms.InputTag("lowPtQuadStepTrackCandidates")
trackSeedSizeBin ['lowPtQuadStep'] = cms.int32(100)
trackSeedSizeMin ['lowPtQuadStep'] = cms.double(0)
trackSeedSizeMax ['lowPtQuadStep'] = cms.double(10000)
clusterLabel     ['lowPtQuadStep'] = cms.vstring('Pix')
clusterBin       ['lowPtQuadStep'] = cms.int32(100)
clusterMax       ['lowPtQuadStep'] = cms.double(20000)

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
trackSeedSizeMax ['pixelPairStep'] = cms.double(10000)                         
TCSizeMax        ['pixelPairStep'] = cms.double(199.5)
clusterLabel     ['pixelPairStep'] = cms.vstring('Pix')
clusterBin       ['pixelPairStep'] = cms.int32(100)
clusterMax       ['pixelPairStep'] = cms.double(20000)

seedInputTag     ['detachedQuadStep'] = cms.InputTag("detachedQuadStepSeeds")
trackCandInputTag['detachedQuadStep'] = cms.InputTag("detachedQuadStepTrackCandidates")
trackSeedSizeBin ['detachedQuadStep'] = cms.int32(100)
trackSeedSizeMin ['detachedQuadStep'] = cms.double(0)
trackSeedSizeMax ['detachedQuadStep'] = cms.double(10000)
TCSizeMax        ['detachedQuadStep'] = cms.double(199.5)
clusterLabel     ['detachedQuadStep'] = cms.vstring('Pix')
clusterBin       ['detachedQuadStep'] = cms.int32(100)
clusterMax       ['detachedQuadStep'] = cms.double(20000)

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
trackSeedSizeBin ['mixedTripletStep'] = cms.int32(200)
trackSeedSizeMin ['mixedTripletStep'] = cms.double(0)
trackSeedSizeMax ['mixedTripletStep'] = cms.double(10000)                         
TCSizeMax        ['mixedTripletStep'] = cms.double(199.5)
clusterLabel     ['mixedTripletStep'] = cms.vstring('Tot')
clusterBin       ['mixedTripletStep'] = cms.int32(100)
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
TCSizeMax        ['tobTecStep'] = cms.double(199.5)
clusterLabel     ['tobTecStep'] = cms.vstring('Strip')
clusterBin       ['tobTecStep'] = cms.int32(100)
clusterMax       ['tobTecStep'] = cms.double(100000)

seedInputTag     ['muonSeededStepInOut'] = cms.InputTag("muonSeededSeedsInOut")
trackCandInputTag['muonSeededStepInOut'] = cms.InputTag("muonSeededTrackCandidatesInOut")
trackSeedSizeBin ['muonSeededStepInOut'] = cms.int32(30)
trackSeedSizeMin ['muonSeededStepInOut'] = cms.double(-0.5)                         
trackSeedSizeMax ['muonSeededStepInOut'] = cms.double(29.5)
TCSizeMax        ['muonSeededStepInOut'] = cms.double(199.5)
clusterLabel     ['muonSeededStepInOut'] = cms.vstring('Strip')
clusterBin       ['muonSeededStepInOut'] = cms.int32(100)
clusterMax       ['muonSeededStepInOut'] = cms.double(100000)

seedInputTag     ['muonSeededStepOutIn'] = cms.InputTag("muonSeededSeedsOutIn")
trackCandInputTag['muonSeededStepOutIn'] = cms.InputTag("muonSeededTrackCandidatesOutIn")
trackSeedSizeBin ['muonSeededStepOutIn'] = cms.int32(30)
trackSeedSizeMin ['muonSeededStepOutIn'] = cms.double(-0.5)                         
trackSeedSizeMax ['muonSeededStepOutIn'] = cms.double(29.5)
TCSizeMax        ['muonSeededStepOutIn'] = cms.double(199.5)
clusterLabel     ['muonSeededStepOutIn'] = cms.vstring('Strip')
clusterBin       ['muonSeededStepOutIn'] = cms.int32(100)
clusterMax       ['muonSeededStepOutIn'] = cms.double(100000)

seedInputTag     ['muonSeededStepOutInDisplaced'] = cms.InputTag("muonSeededSeedsOutInDisplaced")
trackCandInputTag['muonSeededStepOutInDisplaced'] = cms.InputTag("muonSeededTrackCandidatesOutInDisplacedg")
trackSeedSizeBin ['muonSeededStepOutInDisplaced'] = cms.int32(30)
trackSeedSizeMin ['muonSeededStepOutInDisplaced'] = cms.double(-0.5)                         
trackSeedSizeMax ['muonSeededStepOutInDisplaced'] = cms.double(29.5)
TCSizeMax        ['muonSeededStepOutInDisplaced'] = cms.double(199.5)
clusterLabel     ['muonSeededStepOutInDisplaced'] = cms.vstring('Strip')
clusterBin       ['muonSeededStepOutInDisplaced'] = cms.int32(100)
clusterMax       ['muonSeededStepOutInDisplaced'] = cms.double(100000)

seedInputTag     ['jetCoreRegionalStep'] = cms.InputTag("jetCoreRegionalStepSeeds")
trackCandInputTag['jetCoreRegionalStep'] = cms.InputTag("jetCoreRegionalStepTrackCandidates")
trackSeedSizeBin ['jetCoreRegionalStep'] = cms.int32(100)
trackSeedSizeMin ['jetCoreRegionalStep'] = cms.double(-0.5)                         
trackSeedSizeMax ['jetCoreRegionalStep'] = cms.double(199.5)
clusterLabel     ['jetCoreRegionalStep'] = cms.vstring('Tot')
clusterBin       ['jetCoreRegionalStep'] = cms.int32(100)
clusterMax       ['jetCoreRegionalStep'] = cms.double(100000)
regionLabel      ['jetCoreRegionalStep'] = "jetCoreRegionalStepTrackingRegions"
regionCandidateLabel['jetCoreRegionalStep'] = "jetsForCoreTracking"

for _eraName, _postfix, _era in _cfg.allEras():
    locals()["selectedIterTrackingStep"+_postfix] = _cfg.iterationAlgos(_postfix)
#selectedIterTrackingStep.append('muonSeededStepOutInDisplaced')

