import FWCore.ParameterSet.Config as cms

# prep a FED skip list
skipList = []
for i in range(9):
	skipList[len(skipList):] = [1100+i*2]
skipList[len(skipList):] = [718, 719, 720, 721, 722, 723, 724];

rawTask = cms.EDAnalyzer(
	"RawTask",
	
	#	standard parameters
	name = cms.untracked.string("RawTask"),
	debug = cms.untracked.int32(0),
	runkeyVal = cms.untracked.int32(0),
	runkeyName = cms.untracked.string("pp_run"),
	ptype = cms.untracked.int32(0),
	mtype = cms.untracked.bool(True),
	subsystem = cms.untracked.string("Hcal"),

	#	tags
	tagFEDs = cms.untracked.InputTag("rawDataCollector"),

	#	Common Parameters
	skipFEDList = cms.untracked.vint32(skipList)
)












