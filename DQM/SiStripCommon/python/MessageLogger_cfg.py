import FWCore.ParameterSet.Config as cms

process = cms.Process("MessageLogger")
process.load("DQM.SiStripCommon.MessageLogger_cfi")

process.JobReportService = cms.Service("JobReportService")

process.SiteLocalConfigService = cms.Service("SiteLocalConfigService")


