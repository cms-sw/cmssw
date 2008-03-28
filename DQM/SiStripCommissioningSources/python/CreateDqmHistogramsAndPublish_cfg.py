import FWCore.ParameterSet.Config as cms

process = cms.Process("CreateDqmHistogramsAndPublish")
process.load("DQM.SiStripCommon.MessageLogger_cfi")

process.load("OnlineDB.SiStripESSources.FedCablingFromExampleXmlFile_cff")

process.load("IORawData.SiStripInputSources.ExampleRUFile_cff")

process.load("DQM.SiStripCommon.DaqMonitorROOTBackEnd_cfi")

process.load("DQM.SiStripCommon.MonitorDaemon_cfi")

process.load("EventFilter.SiStripRawToDigi.FedChannelDigis_cfi")

process.load("DQM.SiStripCommissioningSources.CommissioningHistos_cfi")

process.load("DQM.SiStripCommon.PoolOutputSafe_cfi")

process.p = cms.Path(process.FedChannelDigis*process.CommissioningHistos)
process.TBRUInputSource.fileNames = ['file:/afs/cern.ch/cms/cmt/onlinedev/data/examples/RU0030349_000.root']
process.maxEvents.input = 10

