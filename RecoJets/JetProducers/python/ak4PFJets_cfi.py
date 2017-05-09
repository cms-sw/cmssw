import FWCore.ParameterSet.Config as cms

ak4PFJets = cms.EDProducer( "FastjetJetProducer",
		src 		= cms.InputTag('particleFlow'),     	## just needed to not change several files
		doAreaFastjet	= cms.bool(False),   			## just needed to not change several files
    		jetPtMin   	= cms.double(5.0), 			## just needed to not change several files
		jetAlgorithm 	= cms.string("AntiKt"),
		rParam       	= cms.double(0.4)
		)

