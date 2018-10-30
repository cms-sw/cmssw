import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer

# FCD channel coordinates (ieta, iphi, depth)
fcd_channels_tuple = [
	(-1,  5, -98),
	(-1,  6, -98),
	(-1,  7, -98),
	(-1,  8, -98),
	(-1,  9, -98),
	(-1, 10, -98),
	(-1, 11, -98),
	(-1, 12, -98),
	(-1, 13, -98),
	(-1, 14, -98),
	(-1, 15, -98),
	(-1, 16, -98),
	(-1, 17, -98),
	(-1, 18, -98),
	(-1, 19, -98),
	(-1, 20, -98),
	(-1, 21, -98),
	(-1, 22, -98),
	(-1, 23, -98),
	(-1, 24, -98),
	(-1, 25, -98),
	(-1, 26, -98),
	(-1, 27, -98),
	(-1, 28, -98),
]

fcd_channels_tuple = [
	(38, 11, 0, 0),
	(38, 11, 0, 1),
	(38, 11, 0, 2),
	(38, 11, 0, 3),
	(38, 11, 1, 0),
	(38, 11, 1, 1),
	(38, 11, 1, 2),
	(38, 11, 1, 3),
	(38, 11, 2, 0),
	(38, 11, 2, 1),
	(38, 11, 2, 2),
	(38, 11, 2, 3),
	(38, 11, 3, 0),
	(38, 11, 3, 1),
	(38, 11, 3, 2),
	(38, 11, 3, 3),
	(38, 11, 4, 0),
	(38, 11, 4, 1),
	(38, 11, 4, 2),
	(38, 11, 4, 3),
	(38, 11, 5, 0),
	(38, 11, 5, 1),
	(38, 11, 5, 2),
	(38, 11, 5, 3),
]

# Convert tuple to CMSSW object
fcd_channels = cms.PSet(
	crate = cms.untracked.vint32(),
	slot = cms.untracked.vint32(),
	fiber = cms.untracked.vint32(),
	fiber_channel = cms.untracked.vint32()
)
for channel in fcd_channels_tuple:
	fcd_channels.crate.append(channel[0])
	fcd_channels.slot.append(channel[1])
	fcd_channels.fiber.append(channel[2])
	fcd_channels.fiber_channel.append(channel[3])

fcdTask = DQMEDAnalyzer(
	"FCDTask",
	#	standard parameters
	name = cms.untracked.string("FDCTask"),
	debug = cms.untracked.int32(0),
	runkeyVal = cms.untracked.int32(0),
	runkeyName = cms.untracked.string("pp_run"),
	ptype = cms.untracked.int32(0),
	mtype = cms.untracked.bool(True),
	subsystem = cms.untracked.string("Hcal"),

	#	tags
	tagFCDDigis = cms.untracked.InputTag('hcalDigis', 'ZDC'),

	fcdChannels = fcd_channels,
)
