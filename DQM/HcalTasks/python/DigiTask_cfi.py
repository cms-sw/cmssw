import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
from DQM.HcalTasks.LEDCalibrationChannels import ledCalibrationChannels

digiTask = DQMEDAnalyzer(
	"DigiTask",

	# Externals
	ledCalibrationChannels = ledCalibrationChannels,

	#	standard parameters
	name = cms.untracked.string("DigiTask"),
	debug = cms.untracked.int32(0),
	runkeyVal = cms.untracked.int32(0),
	runkeyName = cms.untracked.string("pp_run"),
	ptype = cms.untracked.int32(0),
	mtype = cms.untracked.bool(True),
	subsystem = cms.untracked.string("Hcal"),

	#	tags
	tagHBHE = cms.untracked.InputTag("hcalDigis"),
	tagHO = cms.untracked.InputTag("hcalDigis"),
	tagHF = cms.untracked.InputTag("hcalDigis"),

	#	Cuts
	cutSumQ_HBHE = cms.untracked.double(20),
	cutSumQ_HO = cms.untracked.double(20),
	cutSumQ_HF = cms.untracked.double(20),

	#	ratio thresholds
	thresh_unifh = cms.untracked.double(0.2),
	thresh_led = cms.untracked.double(20),

	qie10InConditions = cms.untracked.bool(False),

	# Reference digi sizes
	refDigiSize = cms.untracked.vuint32(10, 10, 10, 4), # HB, HE, HO, HF

)

from Configuration.Eras.Modifier_run2_HF_2017_cff import run2_HF_2017
run2_HF_2017.toModify(digiTask, qie10InConditions=True)
run2_HF_2017.toModify(digiTask, refDigiSize=[10, 10, 10, 3])

from Configuration.Eras.Modifier_run2_HCAL_2018_cff import run2_HCAL_2018
run2_HCAL_2018.toModify(digiTask, qie10InConditions=True)
run2_HCAL_2018.toModify(digiTask, refDigiSize=[8, 8, 10, 3])
