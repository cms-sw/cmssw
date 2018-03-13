import FWCore.ParameterSet.Config as cms
process = cms.Process("SiStripCommissioningOfflineDbClient")
process.load("DQM.SiStripCommon.MessageLogger_cfi")
process.load("DQM.SiStripCommon.DaqMonitorROOTBackEnd_cfi")

process.load("OnlineDB.SiStripConfigDb.SiStripConfigDb_cfi")
process.SiStripConfigDb.UsingDb = True ### cause we don't have access to the db 
process.SiStripConfigDb.ConfDb  = 'overwritten/by@confdb'  
process.SiStripConfigDb.Partitions.PrimaryPartition.PartitionName = 'CR_14-JUL-2017_1'
process.SiStripConfigDb.Partitions.PrimaryPartition.RunNumber     = 299055
process.SiStripConfigDb.TNS_ADMIN = '/home/xdaqtk/'

process.source = cms.Source("EmptySource") 
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(2) ) 

process.load("DQM.SiStripCommissioningDbClients.OfflineDbClient_cff")
process.db_client.FilePath         = cms.untracked.string('./')
process.db_client.RunNumber        = cms.untracked.uint32(299055)
process.db_client.UseClientFile    = cms.untracked.bool(False)
process.db_client.DisableDevices   = cms.untracked.bool(False)
process.db_client.UploadHwConfig   = cms.untracked.bool(False)
process.db_client.UploadAnalyses   = cms.untracked.bool(False)
process.db_client.SaveClientFile   = cms.untracked.bool(True)
### Bad strip analysis options
process.db_client.PedsFullNoiseParameters.DisableBadStrips   = cms.bool(True)
process.db_client.PedsFullNoiseParameters.KeepStripsDisabled = cms.bool(False)
process.db_client.PedsFullNoiseParameters.UploadOnlyStripBadChannelBit = cms.bool(False)
process.db_client.PedsFullNoiseParameters.SkipEmptyStrips    =  cms.bool(True) 
process.db_client.PedsFullNoiseParameters.UploadPedsFullNoiseDBTable  =  cms.bool(False) 
   
process.db_client.OutputRootFile   = cms.untracked.string("SiStripCommissioningClient_CRACK")

process.p = cms.Path(process.db_client)


processDumpFile = open('processDump.py', 'w')
print >> processDumpFile, process.dumpPython()

