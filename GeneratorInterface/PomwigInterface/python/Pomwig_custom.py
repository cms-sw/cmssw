import FWCore.ParameterSet.Config as cms

def customise(process):

	process.genParticles.abortOnUnknownPDGCode = False

	return(process)
