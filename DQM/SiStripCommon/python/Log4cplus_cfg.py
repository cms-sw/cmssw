import FWCore.ParameterSet.Config as cms

process = cms.Process("Log4cplus")
process.load("DQM.SiStripCommon.Log4cplus_cff")

process.MessageLogger.log4cplus = cms.PSet(
    threshold = cms.untracked.string('DEBUG')
)

