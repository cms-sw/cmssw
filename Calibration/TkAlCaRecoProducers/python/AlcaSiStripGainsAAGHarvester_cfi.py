import FWCore.ParameterSet.Config as cms

from CalibTracker.SiStripChannelGain.SiStripGainsPCLHarvester_cfi import SiStripGainsPCLHarvester
alcaSiStripGainsAAGHarvester = SiStripGainsPCLHarvester.clone()
alcaSiStripGainsAAGHarvester.calibrationMode     = cms.untracked.string('AagBunch')
alcaSiStripGainsAAGHarvester.DQMdir              = cms.untracked.string('AlCaReco/SiStripGainsAAG')
alcaSiStripGainsAAGHarvester.Record              = cms.untracked.string('SiStripApvGainRcdAAG')
