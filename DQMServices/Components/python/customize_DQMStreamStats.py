import FWCore.ParameterSet.Config as cms

def dumpEndOfRun(process):
	process.load("DQMServices.Components.DQMStreamStats_cfi")
	process.stream = cms.EndPath(process.dqmStreamStats)
	process.schedule.append(process.stream)
	return(process)

