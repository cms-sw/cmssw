import FWCore.ParameterSet.Config as cms

PseudoBayesPattern = cms.PSet(pattern_filename = cms.FileInPath("L1Trigger/DTTriggerPhase2/data/PseudoBayesPatterns_uncorrelated_v0.root"),
                              debug = cms.untracked.bool(False),
                              #Minimum number of layers hit (total). Together with the two parameters under this it means 4+4, 4+3 or 3+3
                              minNLayerHits   = cms.int32(3),
                              #Minimum number of hits in the most hit superlayer
                              minSingleSLHitsMax = cms.int32(3),
                              #Minimum number of hits in the less hit superlayer
                              minSingleSLHitsMin = cms.int32(0),
                              #By default pattern width is 1, 0 can be considered (harder fits but, lower efficiency of high quality), 2 is the absolute limit unless we have extremely bent muons somehow
                              allowedVariance = cms.int32(1),
                              #If true, it will provide all candidate sets with the same hits of the same quality (with lateralities defined). If false only the leading one (with its lateralities).
                              allowDuplicates = cms.bool(False),
                              #Also provide best estimates for the lateralities
                              setLateralities = cms.bool(True),
                              #Allow for uncorrelated patterns searching 
                              allowUncorrelatedPatterns = cms.bool(True),
                              #If uncorrelated, minimum hits 
                              minUncorrelatedHits = cms.int32(3),
                              #DTPrimitives are saved in the appropriate element of the muonPath array
                              saveOnPlace = cms.bool(True),
                              #Maximum number of muonpaths created per final match
                              maxPathsPerMatch = cms.int32(256),
                              )
