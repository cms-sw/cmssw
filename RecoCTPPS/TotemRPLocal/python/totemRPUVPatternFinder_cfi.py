import FWCore.ParameterSet.Config as cms

totemRPUVPatternFinder = cms.EDProducer("TotemRPUVPatternFinder",
    # input selection
    tagRecHit = cms.InputTag("totemRPRecHitProducer"),

    verbosity = cms.untracked.uint32(0),
    
    # if a plane has more hits than this parameter, it is considered as dirty
    maxHitsPerPlaneToSearch = cms.uint32(5),

    # minimal number of reasonable (= not empty and not dirty) planes per projeciton and per RP,
    # to start the pattern search
    minPlanesPerProjectionToSearch = cms.uint32(3),

    # (full) cluster size in slope-intercept space
    clusterSize_a = cms.double(0.02), # rad
    clusterSize_b = cms.double(0.3),  # mm

    # minimal weight of (Hough) cluster to accept it as candidate
    #   weight of cluster = sum of weights of contributing points
    #   weight of point = sigma0 / sigma_of_point
    #   most often: weight of point ~ 1, thus cluster weight is roughly number of contributing points
    threshold = cms.double(2.99),

    # minimal number of planes (in the recognised patterns) per projeciton and per RP,
    # to tag the candidate as fittable
    minPlanesPerProjectionToFit = cms.uint32(3),

    # whether to allow combination of most significant U and V pattern, in case there several of them
    # don't set it to True, unless you have reason
    allowAmbiguousCombination = cms.bool(False),

    # maximal angle (in any projection) to mark the candidate as fittable -> controls track parallelity with beam
	# huge value -> no constraint
    max_a_toFit = cms.double(10.0),

    # if a RP or projection needs adjustment of the above settings, you can use the following format
	#	exceptionalSettings = cms.VPSet(
	#	    cms.PSet(
	#	        rpId = cms.uint32(20),
	#	        minPlanesPerProjectionToFit_U = cms.uint32(2),
	#	        minPlanesPerProjectionToFit_V = cms.uint32(3),
	#	        threshold_U = cms.double(1.99),
	#	        threshold_V = cms.double(2.99)
	#	    )
	#	)
    exceptionalSettings = cms.VPSet()
)
