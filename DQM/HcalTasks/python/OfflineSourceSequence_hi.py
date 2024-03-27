import FWCore.ParameterSet.Config as cms

#-----------------
#	HCAL DQM Offline Source Sequence Definition for Heavy Ions
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
digiTask.runkeyVal = 4
tpTask.runkeyVal = 4
recHitTask.runkeyVal = 4
rawTask.runkeyVal = 4

digiTask.runkeyName = "hi_run"
tpTask.runkeyName = "hi_run"
recHitTask.runkeyName = "hi_run"
rawTask.runkeyName = "hi_run"

#	Set the Emulator label for TP Task
tpTask.tagEmul = "valHcalTriggerPrimitiveDigis"

hcalOfflineSourceSequence = cms.Sequence(
	digiTask
	+tpTask
	+recHitTask
	+rawTask)



