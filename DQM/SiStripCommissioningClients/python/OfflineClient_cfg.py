import FWCore.ParameterSet.Config as cms

process = cms.Process("SiStripCommissioningOfflineClient")

process.load("DQM.SiStripCommon.MessageLogger_cfi")

process.load("DQM.SiStripCommon.DaqMonitorROOTBackEnd_cfi")

process.load("IORawData.SiStripInputSources.EmptySource_cff")
process.maxEvents.input = 2

process.load("DQM.SiStripCommissioningClients.OfflineClient_cff")
process.client.FilePath       = cms.untracked.string('DATALOCATION')
process.client.RunNumber      = cms.untracked.uint32(RUNNUMBER)
process.client.UseClientFile  = cms.untracked.bool(CLIENTFLAG)
process.client.SaveClientFile = cms.untracked.bool(SAVECLIENTFILE)

process.p = cms.Path(process.client)

