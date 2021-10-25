import FWCore.ParameterSet.Config as cms

PseudoBayesPattern = cms.PSet(pattern_filename = cms.untracked.FileInPath("L1Trigger/DTTriggerPhase2/data/PseudoBayesPatterns_uncorrelated_v0.root"),
                              debug = cms.untracked.bool(False),
                              #Minimum number of layers hit (total). Together with the two parameters under this it means 4+4, 4+3 or 3+3
                              minNLayerHits   = cms.untracked.int32(3),
                              #Minimum number of hits in the most hit superlayer
                              minSingleSLHitsMax = cms.untracked.int32(3),
                              #Minimum number of hits in the less hit superlayer
                              minSingleSLHitsMin = cms.untracked.int32(0),
                              #By default pattern width is 1, 0 can be considered (harder fits but, lower efficiency of high quality), 2 is the absolute limit unless we have extremely bent muons somehow
                              allowedVariance = cms.untracked.int32(1),
                              #If true, it will provide all candidate sets with the same hits of the same quality (with lateralities defined). If false only the leading one (with its lateralities).
                              allowDuplicates = cms.untracked.bool(False),
                              #Also provide best estimates for the lateralities
                              setLateralities = cms.untracked.bool(True),
                              #Allow for uncorrelated patterns searching 
                              allowUncorrelatedPatterns = cms.untracked.bool(True),
                              #If uncorrelated, minimum hits 
                              minUncorrelatedHits = cms.untracked.int32(3),
                              #DTPrimitives are saved in the appropriate element of the muonPath array
                              saveOnPlace = cms.untracked.bool(True),
                              #Maximum number of muonpaths created per final match
                              maxPathsPerMatch = cms.untracked.int32(256),
                              )
