import FWCore.ParameterSet.Config as cms

#-----------------
#	HCAL DQM Offline Source Sequence Definition for pp
#	To be used for Offline DQM importing
#-----------------

#	import the tasks
from DQM.HcalTasks.DigiTask import digiTask
from DQM.HcalTasks.RawTask import rawTask
from DQM.HcalTasks.TPTask import tpTask
from DQM.HcalTasks.RecHitTask import recHitTask

#	set processing type to Offine
digiTask.ptype = cms.untracked.int32(1)
tpTask.ptype = cms.untracked.int32(1)
recHitTask.ptype = cms.untracked.int32(1)
rawTask.ptype = cms.untracked.int32(1)

#	set the label for Emulator TP Task
tpTask.tagEmul = cms.untracked.InputTag("valHcalTriggerPrimitiveDigis")

hcalOfflineSourceSequence = cms.Sequence(
	digiTask
	+tpTask
	+recHitTask
	+rawTask)

