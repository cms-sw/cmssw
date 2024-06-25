import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester
from DQM.HcalTasks.DigiTask_cfi import digiTask

hcalOnlineHarvesting = DQMEDHarvester(
	"HcalOnlineHarvesting",

	name = cms.untracked.string("HcalOnlineHarvesting"),
	debug = cms.untracked.int32(0),
	runkeyVal = cms.untracked.int32(0),
	runkeyName = cms.untracked.string('pp_run'),
	ptype = cms.untracked.int32(0),
	mtype = cms.untracked.bool(True),
	subsystem = cms.untracked.string("Hcal"),
	refDigiSize = digiTask.refDigiSize,
)
