import FWCore.ParameterSet.Config as cms

process = cms.Process("Source")

process.load("DQM.SiStripCommon.MessageLogger_cfi")

process.load("DQM.SiStripCommon.DaqMonitorROOTBackEnd_cfi")

process.load("OnlineDB.SiStripConfigDb.SiStripConfigDb_cfi")
process.SiStripConfigDb.UsingDb = True                    # should be true!
process.SiStripConfigDb.ConfDb = 'user/password@account'  # taken from $CONFDB
process.SiStripConfigDb.Partitions.PrimaryPartition.PartitionName = 'CR_14-JUL-2017_1'
process.SiStripConfigDb.Partitions.PrimaryPartition.RunNumber     = 299055

process.SiStripCondObjBuilderFromDb = cms.Service("SiStripCondObjBuilderFromDb")
process.FedCablingFromConfigDb = cms.ESSource("SiStripFedCablingBuilderFromDb",
  CablingSource = cms.untracked.string('UNDEFINED')   ## <-- this should be replaced by "DEVICES" for a connection run!
)
process.SiStripDetInfoFileReader = cms.Service("SiStripDetInfoFileReader")
process.PedestalsFromConfigDb = cms.ESSource("SiStripPedestalsBuilderFromDb")
process.NoiseFromConfigDb = cms.ESSource("SiStripNoiseBuilderFromDb")
# produce SiStripFecCabling and SiStripDetCabling out of SiStripFedCabling
process.sistripconn = cms.ESProducer("SiStripConnectivity")

process.source = cms.Source("NewEventStreamFileReader",
  fileNames = cms.untracked.vstring()
)
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.load("EventFilter.SiStripRawToDigi.FedChannelDigis_cfi")
process.FedChannelDigis.UnpackBadChannels = cms.bool(False)

process.load("DQM.SiStripCommissioningSources.CommissioningHistos_cfi")
process.CommissioningHistos.CommissioningTask = 'PEDS_FULL_NOISE'  # <-- run type taken from event data, but can be overriden
process.CommissioningHistos.PedsFullNoiseParameters.NrEvToSkipAtStart = 100
process.CommissioningHistos.PedsFullNoiseParameters.NrEvForPeds       = 1000
process.CommissioningHistos.PedsFullNoiseParameters.FillNoiseProfile  = True

process.p = cms.Path(process.FedChannelDigis*process.CommissioningHistos)

process.source.fileNames.extend(cms.untracked.vstring('file:/exports/Data/closed/USC.00299055.0001.A.storageManager.00.0000.dat'))
