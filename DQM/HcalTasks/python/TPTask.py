import FWCore.ParameterSet.Config as cms

fgbits = [1 for x in range(5)]
fgbits.append(0)

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
tpTask = DQMEDAnalyzer(
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
	tagDataL1Rec = cms.untracked.InputTag("caloLayer1Digis"),

	tagEmul = cms.untracked.InputTag("emulTPDigis"),
	tagEmulNoTDCCut = cms.untracked.InputTag("emulTPDigisNoTDCCut"),

	#	cut to put on Et to get the occupancy
	cutEt = cms.untracked.int32(3),

	#	some speacial features
	skip1x1 = cms.untracked.bool(True),

	thresh_EtMsmRate_high = cms.untracked.double(0.2),
	thresh_EtMsmRate_low = cms.untracked.double(0.1),
	thresh_FGMsmRate_high = cms.untracked.double(0.2),
	thresh_FGMsmRate_low = cms.untracked.double(0.1),
	thresh_DataMsn = cms.untracked.double(0.1),
	thresh_EmulMsn = cms.untracked.double(0.1),
    vFGBitsReady = cms.untracked.vint32(fgbits)
)












