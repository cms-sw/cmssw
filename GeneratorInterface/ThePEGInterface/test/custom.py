import FWCore.ParameterSet.Config as cms

# This statement is necessary to make sure CMSSW doesn't throw an exception
# due to its outdated particle table
def customise(process):
	process.genParticles.abortOnUnknownPDGCode = False

	return process
