import FWCore.ParameterSet.Config as cms

# process declaration
process = cms.Process("SiStripCommissioningOfflineDbClient")


#############################################
# General setup
#############################################

# message logger
process.load('DQM.SiStripCommissioningSources.OfflineMessageLogger_cff')

# DQM service
process.load('DQM.SiStripCommissioningSources.OfflineDQM_cff')

# config db settings
process.load("OnlineDB.SiStripConfigDb.SiStripConfigDb_cfi")
process.SiStripConfigDb.UsingDb = True                                            # true means use database (not xml files)
process.SiStripConfigDb.ConfDb  = 'overwritten/by@confdb'                         # database connection account ( or use CONFDB env. var.)
process.SiStripConfigDb.Partitions.PrimaryPartition.PartitionName = 'DBPART'      # database partition (or use ENV_CMS_TK_PARTITION env. var.)
process.SiStripConfigDb.Partitions.PrimaryPartition.RunNumber     = RUNNUMBER     # specify run number ("0" means use major/minor versions, which are by default set to "current state")
#process.SiStripConfigDb.TNS_ADMIN = '/etc'                                        # location of tnsnames.ora, needed at P5, not in TAC
    
# input source
process.load("IORawData.SiStripInputSources.EmptySource_cff")
process.maxEvents.input = 2


#############################################
# extra setup for latency & fine delay
#############################################

# geometry
process.load('DQM.SiStripCommissioningSources.P5Geometry_cff')
# magnetic field (0T by default)
process.load('MagneticField.Engine.uniformMagneticField_cfi')
# fake global position
process.load('Alignment.CommonAlignmentProducer.GlobalPosition_Fake_cff')


##############################################
# modules & path for analysis
##############################################

process.load("DQM.SiStripCommissioningDbClients.OfflineDbClient_cff")
process.db_client.FilePath         = cms.untracked.string('DATALOCATION')
process.db_client.RunNumber        = cms.untracked.uint32(RUNNUMBER)
process.db_client.UseClientFile    = cms.untracked.bool(CLIENTFLAG)
process.db_client.UploadHwConfig   = cms.untracked.bool(DBUPDATE)
process.db_client.UploadAnalyses   = cms.untracked.bool(ANALUPDATE)
process.db_client.DisableDevices   = cms.untracked.bool(DISABLEDEVICES)
process.db_client.DisableBadStrips = cms.untracked.bool(DISABLEBADSTRIPS)
process.db_client.AddBadStrips		 = cms.untracked.bool(ADDBADSTRIPS)
process.db_client.SaveClientFile   = cms.untracked.bool(SAVECLIENTFILE)

process.p = cms.Path(process.db_client)
