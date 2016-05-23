import FWCore.ParameterSet.Config as cms


from CalibTracker.SiStripChannelGain.computeGain_cff import SiStripCalib
alcaSiStripGainsHarvester = SiStripCalib.clone()
alcaSiStripGainsHarvester.AlgoMode            = cms.untracked.string('PCL')
alcaSiStripGainsHarvester.FirstSetOfConstants = cms.untracked.bool(False)
alcaSiStripGainsHarvester.CalibrationLevel    = cms.untracked.int32(0) # 0==APV, 1==Laser, 2==module
alcaSiStripGainsHarvester.harvestingMode      = cms.untracked.bool(True)
