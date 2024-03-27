import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester
from DQM.HcalTasks.DigiTask_cfi import digiTask

hcalOfflineHarvesting = DQMEDHarvester(
	"HcalOfflineHarvesting",

	name = cms.untracked.string("HcalOfflineHarvesting"),
	debug = cms.untracked.int32(0),
	runkeyVal = cms.untracked.int32(0),
	runkeyName = cms.untracked.string('pp_run'),
	ptype = cms.untracked.int32(1),
	mtype = cms.untracked.bool(True),
	subsystem = cms.untracked.string("Hcal"),

	thresh_EtMsmRate_high = cms.untracked.double(0.2),
	thresh_EtMsmRate_low = cms.untracked.double(0.1),
	thresh_FGMsmRate_high = cms.untracked.double(0.2),
	thresh_FGMsmRate_low = cms.untracked.double(0.1),
	thresh_unihf = cms.untracked.double(0.2),
	thresh_tcds = cms.untracked.double(1.5),
	refDigiSize = digiTask.refDigiSize,
)
