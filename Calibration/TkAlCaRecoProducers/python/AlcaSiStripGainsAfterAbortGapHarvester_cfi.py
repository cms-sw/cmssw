import FWCore.ParameterSet.Config as cms

from CalibTracker.SiStripChannelGain.SiStripGainsPCLHarvester_cfi import SiStripGainsPCLHarvester
alcaSiStripGainsAfterAbortGapHarvester = SiStripGainsPCLHarvester.clone()
alcaSiStripGainsAfterAbortGapHarvester.calibrationMode     = cms.untracked.string('AagBunch')
alcaSiStripGainsAfterAbortGapHarvester.DQMdir              = cms.untracked.string('AlCaReco/SiStripGainsAfterAbortGap')
alcaSiStripGainsAfterAbortGapHarvester.Record              = cms.untracked.string('SiStripApvGainRcdAfterAbortGap')
