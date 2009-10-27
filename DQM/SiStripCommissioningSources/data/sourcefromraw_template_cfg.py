import FWCore.ParameterSet.Config as cms

process = cms.Process("Source")

process.load("DQM.SiStripCommon.MessageLogger_cfi")

process.load("DQM.SiStripCommon.DaqMonitorROOTBackEnd_cfi")

process.load("OnlineDB.SiStripConfigDb.SiStripConfigDb_cfi")
process.SiStripConfigDb.UsingDb = True                    # should be true!
process.SiStripConfigDb.ConfDb = 'user/password@account'  # taken from $CONFDB
process.SiStripConfigDb.Partitions.PrimaryPartition.PartitionName = 'DBPART'
process.SiStripConfigDb.Partitions.PrimaryPartition.RunNumber     = RUNNUMBER
process.SiStripConfigDb.TNS_ADMIN = '/etc'  # for P5

process.SiStripCondObjBuilderFromDb = cms.Service("SiStripCondObjBuilderFromDb")
process.FedCablingFromConfigDb = cms.ESSource("SiStripFedCablingBuilderFromDb",
  CablingSource = cms.untracked.string('UNDEFINED')   ## <-- this should be replaced by "DEVICES" for a connection run!
)
process.SiStripDetInfoFileReader = cms.Service("SiStripDetInfoFileReader")
process.PedestalsFromConfigDb = cms.ESSource("SiStripPedestalsBuilderFromDb")
process.NoiseFromConfigDb = cms.ESSource("SiStripNoiseBuilderFromDb")
# produce SiStripFecCabling and SiStripDetCabling out of SiStripFedCabling
process.sistripconn = cms.ESProducer("SiStripConnectivity")

process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring()
)
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.FedChannelDigis = cms.EDProducer("SiStripRawToDigiModule",
    ProductLabel = cms.string('source'),
    ProductInstance = cms.string(''),
    CreateDigis = cms.bool(True),
    AppendedBytes = cms.int32(0),
    UseDaqRegister = cms.bool(True),
    UseFedKey = cms.bool(True),
    FedEventDumpFreq = cms.untracked.int32(0),
    FedBufferDumpFreq = cms.untracked.int32(0),
    TriggerFedId = cms.int32(-1),
)

process.load("DQM.SiStripCommissioningSources.CommissioningHistos_cfi")
process.CommissioningHistos.CommissioningTask = 'UNDEFINED'  # <-- run type taken from event data, but can be overriden

process.p = cms.Path(process.FedChannelDigis*process.CommissioningHistos)

#process.out = cms.OutputModule("PoolOutputModule",
#    outputCommands = cms.untracked.vstring('keep *'),
#    fileName = cms.untracked.string('digis.root')
#)
#process.outpath = cms.EndPath(process.out)

