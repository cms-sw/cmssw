import FWCore.ParameterSet.Config as cms

SiStripGainsPCLHarvester = cms.EDAnalyzer(
    "SiStripGainsPCLHarvester",
    Record              = cms.untracked.string('SiStripApvGainRcd'),
    CalibrationLevel    = cms.untracked.int32(0), # 0==APV, 1==Laser, 2==module
    DQMdir              = cms.untracked.string('AlCaReco/SiStripGains'),
    calibrationMode     = cms.untracked.string('StdBunch'),
    minNrEntries        = cms.untracked.double(25),
    GoodFracForTagProd  = cms.untracked.double(0.98),
    NClustersForTagProd = cms.untracked.double(1E8)
    )
