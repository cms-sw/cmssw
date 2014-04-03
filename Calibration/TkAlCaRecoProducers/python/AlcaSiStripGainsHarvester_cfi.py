import FWCore.ParameterSet.Config as cms

# FIXME: the safest option would be to import the basic cfi from a place mantained by the developer

from CalibTracker.SiStripChannelGain.computeGain_cff import SiStripCalib as alcaSiStripGainsHarvester
alcaSiStripGainsHarvester.AlgoMode = cms.untracked.string('PCL')
alcaSiStripGainsHarvester.Tracks   = cms.untracked.InputTag('ALCARECOCalibrationTracksRefit')
alcaSiStripGainsHarvester.FirstSetOfConstants = cms.untracked.bool(False)
alcaSiStripGainsHarvester.CalibrationLevel    = cms.untracked.int32(0) # 0==APV, 1==Laser, 2==module
alcaSiStripGainsHarvester.harvestingMode    = cms.untracked.bool(True)
