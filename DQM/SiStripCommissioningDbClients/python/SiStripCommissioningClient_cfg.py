import FWCore.ParameterSet.Config as cms
import os,sys,getopt,glob,cx_Oracle,subprocess

conn_str = os.path.expandvars("$CONFDB")
conn     = cx_Oracle.connect(conn_str)
e        = conn.cursor()
e.execute('select RUNMODE from run where runnumber = RUNNUMBER')
runmode = e.fetchall()
runtype = -1;
for result in runmode:
    runtype = int(result[0]);
conn.close()

process = cms.Process("SiStripCommissioningOfflineDbClient")

process.load("DQM.SiStripCommon.MessageLogger_cfi")

process.load("DQM.SiStripCommon.DaqMonitorROOTBackEnd_cfi")

process.load("OnlineDB.SiStripConfigDb.SiStripConfigDb_cfi")
process.SiStripConfigDb.UsingDb = True                                            # true means use database (not xml files)
process.SiStripConfigDb.ConfDb  = 'overwritten/by@confdb'                         # database connection account ( or use CONFDB env. var.)
process.SiStripConfigDb.Partitions.PrimaryPartition.PartitionName = 'DBPART'      # database partition (or use ENV_CMS_TK_PARTITION env. var.)
process.SiStripConfigDb.Partitions.PrimaryPartition.RunNumber     = RUNNUMBER     # specify run number ("0" means use major/minor versions, which are by default set to "current state")
process.SiStripConfigDb.TNS_ADMIN = '/etc'                                        # location of tnsnames.ora, needed at P5, not in TAC
#process.SiStripConfigDb.Partitions.PrimaryPartition.ForceCurrentState = cms.untracked.bool(True)

process.source = cms.Source("EmptySource") 
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(2) ) 

process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")
process.load("Geometry.TrackerNumberingBuilder.trackerTopology_cfi")
process.load("Geometry.TrackerGeometryBuilder.trackerParameters_cfi")

process.load("DQM.SiStripCommissioningDbClients.OfflineDbClient_cff")
process.db_client.FilePath         = cms.untracked.string('DATALOCATION')
process.db_client.RunNumber        = cms.untracked.uint32(RUNNUMBER)
process.db_client.UseClientFile    = cms.untracked.bool(CLIENTFLAG)
process.db_client.UploadHwConfig   = cms.untracked.bool(DBUPDATE)
process.db_client.UploadAnalyses   = cms.untracked.bool(ANALUPDATE)
process.db_client.DisableDevices   = cms.untracked.bool(DISABLEDEVICES)
process.db_client.DisableBadStrips = cms.untracked.bool(DISABLEBADSTRIPS)
process.db_client.SaveClientFile   = cms.untracked.bool(SAVECLIENTFILE)

if runtype == 15: ## only needed for spy-channel
    process.db_client.PartitionName    = cms.string("DBPART")

process.db_client.ApvTimingParameters.SkipFecUpdate = cms.bool(True)
process.db_client.ApvTimingParameters.SkipFedUpdate = cms.bool(False)
#process.db_client.ApvTimingParameters.TargetDelay = cms.int32(725)
process.db_client.ApvTimingParameters.TargetDelay = cms.int32(-1)

process.db_client.OptoScanParameters.SkipGainUpdate = cms.bool(False)

process.db_client.PedestalsParameters.KeepStripsDisabled = cms.bool(True)

process.db_client.DaqScopeModeParameters.DisableBadStrips =  cms.bool(False)
process.db_client.DaqScopeModeParameters.KeepStripsDisabled = cms.bool(True)
process.db_client.DaqScopeModeParameters.SkipPedestalUpdate = cms.bool(False)
process.db_client.DaqScopeModeParameters.SkipTickUpdate = cms.bool(False)

### Bad strip analysis options                                                                                                                                                                         
process.db_client.PedsFullNoiseParameters.DisableBadStrips   = cms.bool(True) ## if True the code loops over the dead and bad strips identified and will disable them                                  
process.db_client.PedsFullNoiseParameters.KeepStripsDisabled = cms.bool(True) ## if True, strips that have been already disabled will be kept disabled                                                 
process.db_client.PedsFullNoiseParameters.UploadOnlyStripBadChannelBit = cms.bool(True) ## if True, only the disable flag will be changed, peds and noise cloned from the previous FED version         
process.db_client.PedsFullNoiseParameters.SkipEmptyStrips    =  cms.bool(True) ## if True, empty strips (dead) are skipped --> to avoid to flag bad stuff not powered ON                               
process.db_client.PedsFullNoiseParameters.UploadPedsFullNoiseDBTable  =  cms.bool(False) ## if True, also the pedsfullnoise analysis tables is uploaded                                                

process.p = cms.Path(process.db_client)
