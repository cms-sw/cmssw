import FWCore.ParameterSet.Config as cms


from CalibTracker.SiStripChannelGain.computeGain_cff import SiStripCalib
alcaSiStripGainsAAGHarvester = SiStripCalib.clone()
alcaSiStripGainsAAGHarvester.AlgoMode            = cms.untracked.string('PCL')
alcaSiStripGainsAAGHarvester.FirstSetOfConstants = cms.untracked.bool(False)
alcaSiStripGainsAAGHarvester.calibrationMode     = cms.untracked.string('AagBunch')
alcaSiStripGainsAAGHarvester.DQMdir              = cms.untracked.string('AlCaReco/SiStripGainsAAG')
alcaSiStripGainsAAGHarvester.CalibrationLevel    = cms.untracked.int32(0) # 0==APV, 1==Laser, 2==module
alcaSiStripGainsAAGHarvester.harvestingMode      = cms.untracked.bool(True)
alcaSiStripGainsAAGHarvester.Record              = cms.string('SiStripApvGainRcdAAG')
