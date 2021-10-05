import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
qie10Task = DQMEDAnalyzer(
	"QIE10Task",

	#	standard
	name = cms.untracked.string("QIE10Task"),
	debug = cms.untracked.int32(0),
	runkeyVal = cms.untracked.int32(0),
	runkeyName = cms.untracked.string("pp_run"),

	#	tag
	tagQIE10 = cms.untracked.InputTag("hcalDigis"),

        #       folder
        subsystem  = cms.untracked.string("HcalCalib"),

	#	cuts, 
	cut = cms.untracked.double(20),
	ped = cms.untracked.int32(4)
)
