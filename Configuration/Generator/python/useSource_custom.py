import FWCore.ParameterSet.Config as cms

def customise(process):

	process.g4SimHits.Generator.HepMCProductLabel = cms.string('source')
	process.genParticles.src=cms.InputTag('source')

	return(process)
