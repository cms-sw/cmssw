import FWCore.ParameterSet.Config as cms

HoughGrouping = cms.PSet(debug = cms.untracked.bool(False),
                         # HOUGH TRANSFORM CONFIGURATION
                         # Tangent of the angle that puts a limit to the maximum inclination of a
                         # muon. Then, the maximum number of radians of inclination of the muon
                         # is derived as pi/2 - arctan(angletan).
                         angletan = cms.untracked.double(0.3),
                         # Width (in sexageseimal degrees) of the angle bins of the line space.
                         anglebinwidth = cms.untracked.double(1.0),
                         # Width (in centimetres) of the position bins of the line space.
                         posbinwidth = cms.untracked.double(2.1),
                         
                         # MAXIMA SEARCH CONFIGURATION
                         # Maximum distance (in sexageseimal degrees) used to derive maxima.
                         maxdeltaAngDeg = cms.untracked.double(10),
                         # Maximum distance (in centimetres) used to derive maxima.
                         maxdeltaPos = cms.untracked.double(10),
                         # Upper number of entries of a line space bin that are needed to be
                         # considered for maxima search.
                         UpperNumber = cms.untracked.int32(6),
                         # Lower number of entries of a line space bin that are needed to be
                         # considered for maxima search.
                         LowerNumber = cms.untracked.int32(4),
                         
                         # HITS ASSOCIATION CONFIGURATION
                         # Distance to the wire (in centimetres) from a candidate line below
                         # which no laterality is assumed.
                         MaxDistanceToWire = cms.untracked.double(0.03),
                         
                         # CANDIDATE QUALITY CONFIGURATION
                         # Minimum number of layers on which the candidate has a hit (maximum 8).
                         minNLayerHits = cms.untracked.int32(6),
                         # Minimum number of hits in the superlayer with more hits.
                         minSingleSLHitsMax = cms.untracked.int32(3),
                         # Minimum number of hits in the superlayer with less hits.
                         minSingleSLHitsMin = cms.untracked.int32(3),
                         # Allow the algorithm to give you back uncorrelated candidates.
                         allowUncorrelatedPatterns = cms.untracked.bool(True),
                         # Minimum number of hits that uncorrelated candidates can have.
                         minUncorrelatedHits = cms.untracked.int32(3),
                         )
