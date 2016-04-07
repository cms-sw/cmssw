import FWCore.ParameterSet.Config as cms


from CalibTracker.SiStripChannelGain.computeGain_cff import SiStripCalib
alcaSiStripGainsAfterAbortGapHarvester = SiStripCalib.clone()
alcaSiStripGainsAfterAbortGapHarvester.AlgoMode            = cms.untracked.string('PCL')
alcaSiStripGainsAfterAbortGapHarvester.FirstSetOfConstants = cms.untracked.bool(False)
alcaSiStripGainsAfterAbortGapHarvester.calibrationMode     = cms.untracked.string('AagBunch')
alcaSiStripGainsAfterAbortGapHarvester.DQMdir              = cms.untracked.string('AlCaReco/SiStripGainsAfterAbortGap')
alcaSiStripGainsAfterAbortGapHarvester.CalibrationLevel    = cms.untracked.int32(0) # 0==APV, 1==Laser, 2==module
alcaSiStripGainsAfterAbortGapHarvester.harvestingMode      = cms.untracked.bool(True)
