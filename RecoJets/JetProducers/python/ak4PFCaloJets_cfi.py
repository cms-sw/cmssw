import FWCore.ParameterSet.Config as cms

ak4PFCaloJets = cms.EDProducer( "FastjetJetProducer", 
	src = cms.InputTag('hltParticleFlow'),
	jetAlgorithm = cms.string("AntiKt"),
	rParam       = cms.double(0.4)
	)

