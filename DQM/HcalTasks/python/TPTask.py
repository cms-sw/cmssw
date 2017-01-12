import FWCore.ParameterSet.Config as cms

tpTask = cms.EDAnalyzer(
	"TPTask",
	
	#	standard parameters
	name = cms.untracked.string("TPTask"),
	debug = cms.untracked.int32(0),
	runkeyVal = cms.untracked.int32(0),
	runkeyName = cms.untracked.string("pp_run"),
	ptype = cms.untracked.int32(0),
	mtype = cms.untracked.bool(True),
	subsystem = cms.untracked.string("Hcal"),

	#	tags
	tagData = cms.untracked.InputTag("hcalDigis"),
	tagEmul = cms.untracked.InputTag("emulTPDigis"),

	#	cut to put on Et to get the occupancy
	cutEt = cms.untracked.int32(3),

	#	some speacial features
	skip1x1 = cms.untracked.bool(True),

	thresh_EtMsmRate = cms.untracked.double(0.1),
	thresh_FGMsmRate = cms.untracked.double(0.1),
	thresh_DataMsn = cms.untracked.double(0.1),
	thresh_EmulMsn = cms.untracked.double(0.1),
)












