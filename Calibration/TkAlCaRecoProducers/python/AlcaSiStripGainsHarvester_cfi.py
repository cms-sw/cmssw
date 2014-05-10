import FWCore.ParameterSet.Config as cms


from CalibTracker.SiStripChannelGain.computeGain_cff import SiStripCalib as alcaSiStripGainsHarvester
alcaSiStripGainsHarvester.AlgoMode = cms.untracked.string('PCL')
alcaSiStripGainsHarvester.Tracks   = cms.untracked.InputTag('ALCARECOCalibrationTracksRefit')
alcaSiStripGainsHarvester.FirstSetOfConstants = cms.untracked.bool(False)
alcaSiStripGainsHarvester.CalibrationLevel    = cms.untracked.int32(0) # 0==APV, 1==Laser, 2==module
alcaSiStripGainsHarvester.harvestingMode    = cms.untracked.bool(True)
