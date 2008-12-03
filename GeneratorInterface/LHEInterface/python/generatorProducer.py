import FWCore.ParameterSet.Config as cms

def customise(process):
	process.RandomNumberGeneratorService.generator = process.RandomNumberGeneratorService.theSource

	process.genParticles.abortOnUnknownPDGCode = False
	process.genParticles.src = 'generator'
	process.genParticleCandidates.src = 'generator'
	process.genEventWeight.src = 'generator'
	process.genEventScale.src = 'generator'
	process.genEventPdfInfo.src = 'generator'
	process.genEventProcID.src = 'generator'

	process.VtxSmeared.src = 'generator'

	try:
		process.g4SimHits.Generator.HepMCProductLabel = 'generator'
		process.mergedtruth.HepMCDataLabels.append('generator')
	except:
		pass

	process.output.outputCommands.append('keep *_source_*_*')
	process.output.outputCommands.append('keep *_generator_*_*')

	return process
