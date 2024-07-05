import FWCore.ParameterSet.Config as cms

#-----------------
#	HCAL DQM Offline Source Sequence Definition for Cosmics
#	To be used for Offline DQM importing
#-----------------

#	import the tasks
from DQM.HcalTasks.DigiTask_cfi import digiTask
from DQM.HcalTasks.RawTask_cfi import rawTask
from DQM.HcalTasks.TPTask_cfi import tpTask
from DQM.HcalTasks.RecHitTask_cfi import recHitTask

#	set processing type to Offine
digiTask.ptype = 1
tpTask.ptype = 1
recHitTask.ptype = 1
rawTask.ptype = 1

#	set the run key(value and name)
digiTask.runkeyVal = 2
tpTask.runkeyVal = 2
recHitTask.runkeyVal = 2
rawTask.runkeyVal = 2

digiTask.runkeyName = "cosmic_run"
tpTask.runkeyName = "cosmic_run"
recHitTask.runkeyName = "cosmic_run"
rawTask.runkeyName = "cosmic_run"

#	set the Emulator label for TP Task
tpTask.tagEmul = "valHcalTriggerPrimitiveDigis"

hcalOfflineSourceSequence = cms.Sequence(
	digiTask
	+recHitTask
	+rawTask)



