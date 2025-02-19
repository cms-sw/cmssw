import FWCore.ParameterSet.Config as cms

process = cms.Process("SiStripCommissioningOfflineDbClient")

process.load("DQM.SiStripCommon.MessageLogger_cfi")

process.load("DQM.SiStripCommon.DaqMonitorROOTBackEnd_cfi")

process.load("OnlineDB.SiStripConfigDb.SiStripConfigDb_cfi")
process.SiStripConfigDb.UsingDb = True                                            # true means use database (not xml files)
process.SiStripConfigDb.ConfDb  = 'overwritten/by@confdb'                         # database connection account ( or use CONFDB env. var.)
process.SiStripConfigDb.Partitions.PrimaryPartition.PartitionName = 'DBPART'      # database partition (or use ENV_CMS_TK_PARTITION env. var.)
process.SiStripConfigDb.Partitions.PrimaryPartition.RunNumber     = RUNNUMBER     # specify run number ("0" means use major/minor versions, which are by default set to "current state")
process.SiStripConfigDb.TNS_ADMIN = '/etc'                                        # location of tnsnames.ora, needed at P5, not in TAC

process.load("IORawData.SiStripInputSources.EmptySource_cff")
process.maxEvents.input = 2

process.load("DQM.SiStripCommissioningDbClients.OfflineDbClient_cff")
process.db_client.FilePath         = cms.untracked.string('DATALOCATION')
process.db_client.RunNumber        = cms.untracked.uint32(RUNNUMBER)
process.db_client.UseClientFile    = cms.untracked.bool(CLIENTFLAG)
process.db_client.UploadHwConfig   = cms.untracked.bool(DBUPDATE)
process.db_client.UploadAnalyses   = cms.untracked.bool(ANALUPDATE)
process.db_client.DisableDevices   = cms.untracked.bool(DISABLEDEVICES)
process.db_client.SaveClientFile   = cms.untracked.bool(SAVECLIENTFILE)

process.db_client.ApvTimingParameters.SkipFecUpdate = cms.bool(False)
process.db_client.ApvTimingParameters.SkipFedUpdate = cms.bool(False)
process.db_client.ApvTimingParameters.TargetDelay = cms.int32(725)
process.db_client.OptoScanParameters.SkipGainUpdate = cms.bool(False)
process.db_client.OptoScanParameters.TargetGain = cms.double(0.8)
process.db_client.PedestalsParameters.DisableBadStrips = cms.bool(False)
process.db_client.PedestalsParameters.KeepStripsDisabled = cms.bool(False)
process.db_client.PedsFullNoiseParameters.DisableBadStrips = cms.bool(False)
process.db_client.PedsFullNoiseParameters.KeepStripsDisabled = cms.bool(False)

process.p = cms.Path(process.db_client)
