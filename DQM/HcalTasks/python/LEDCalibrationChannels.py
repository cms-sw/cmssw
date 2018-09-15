# Specify detector coordinates of LED calibration channels
import FWCore.ParameterSet.Config as cms

# Coordinates as python tuples, (ieta, iphi, depth)
led_calibration_channels_tuple = {
	"HB":{},
	"HE":{
		(-16, 48, 10),
		(-16, 112, 10),
		(-17, 48, 10),
		(-17, 112, 10),
		(-18, 48, 10),
		(-18, 112, 10),
		(-19, 48, 10),
		(-19, 112, 10),
		(-20, 48, 10),
		(-20, 112, 10),
		(-21, 48, 10),
		(-21, 112, 10),
		(-22, 48, 10),
		(-22, 112, 10),
		(-23, 48, 10),
		(-23, 112, 10),
		(-24, 48, 10),
		(-24, 112, 10),
		(-48, 48, 10),
		(-48, 112, 10),
		(-49, 48, 10),
		(-49, 112, 10),
		(-50, 48, 10),
		(-50, 112, 10),
		(-51, 48, 10),
		(-51, 112, 10),
		(-52, 48, 10),
		(-52, 112, 10),
		(-53, 48, 10),
		(-53, 112, 10),
		(-54, 48, 10),
		(-54, 112, 10),
		(-55, 48, 10),
		(-55, 112, 10),
		(-56, 48, 10),
		(-56, 112, 10),
	}, 

	"HO":{},
	"HF":{
		(-16, 16, 12),
		(-18, 48, 12),
		(-20, 80, 12),
		(-22, 112, 12),
		(-50, 48, 12),
		(-52, 80, 12),
		(-54, 112, 12),
		(-54, 120, 12),
	},
}

# Convert tuples to CMSSW objects
ledCalibrationChannels = cms.VPSet()
for subdet in ["HB", "HE", "HO", "HF"]:
	subdet_channels = cms.untracked.PSet(
		ieta = cms.untracked.vint32(),
		iphi = cms.untracked.vint32(),
		depth = cms.untracked.vint32()
	)
	for channel in led_calibration_channels_tuple[subdet]:
		subdet_channels.ieta.append(channel[0])
		subdet_channels.iphi.append(channel[1])
		subdet_channels.depth.append(channel[2])
	ledCalibrationChannels.append(subdet_channels)
