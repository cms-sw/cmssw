from RecoJets.JetProducers.hltPUIdAlgo_cff import *

MVAJetPuIdCalculator = cms.EDProducer('MVAJetPuId',
                                      produceJetIds = cms.bool(True),
                                      jetids = cms.InputTag(""),
                                      runMvas = cms.bool(True),
                                      algos = cms.VPSet(cms.VPSet(full_74x)),
				      jets     = cms.InputTag('hltAK4PFJetsCorrected'),
				      rho      = cms.InputTag('hltFixedGridRhoFastjetAll'),
				      vertexes = cms.InputTag('hltPixelVertices'),
				      jec     = cms.string("AK4PFchs"),
				      applyJec = cms.bool(False),
				      inputIsCorrected = cms.bool(True),
		)

MVAJetPuIdEvaluator = cms.EDProducer('MVAJetPuId',
		produceJetIds = cms.bool(False),
		jetids = cms.InputTag("pileupJetIdCalculator"),
		runMvas = cms.bool(True),
		algos = cms.VPSet(cms.VPSet(full_74x)),
		jets     = cms.InputTag('hltAK4PFJetsCorrected'),
		rho      = cms.InputTag('hltFixedGridRhoFastjetAll'),
		vertexes = cms.InputTag('hltPixelVertices'),
		jec     = cms.string("AK4PFchs"),
		applyJec = cms.bool(False),
		inputIsCorrected = cms.bool(True),
		)

