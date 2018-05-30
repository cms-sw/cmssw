import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing
import glob
import os
import os,sys,getopt,glob,cx_Oracle,subprocess

cmsswbase = os.path.expandvars("$CMSSW_BASE/")
inputPath = '/raid/fff'

conn_str = os.path.expandvars("$CONFDB")
conn     = cx_Oracle.connect(conn_str)
e        = conn.cursor()
e.execute('select RUNMODE from run where runnumber = RUNNUMBER')
runmode = e.fetchall()
runtype = -1;
for result in runmode:
    runtype = int(result[0]);
conn.close()

process = cms.Process("SRCEDM")

process.load("DQM.SiStripCommon.MessageLogger_cfi")
process.load("DQM.SiStripCommon.DaqMonitorROOTBackEnd_cfi")

process.load("OnlineDB.SiStripConfigDb.SiStripConfigDb_cfi")
process.SiStripConfigDb.UsingDb = True                    
process.SiStripConfigDb.ConfDb = 'user/password@account'
process.SiStripConfigDb.Partitions.PrimaryPartition.PartitionName = 'DBPART'
process.SiStripConfigDb.Partitions.PrimaryPartition.RunNumber     = RUNNUMBER
process.SiStripConfigDb.TNS_ADMIN = '/etc'

process.SiStripCondObjBuilderFromDb = cms.Service("SiStripCondObjBuilderFromDb")
process.SiStripCondObjBuilderFromDb.UseFEC = cms.untracked.bool(True)
process.SiStripCondObjBuilderFromDb.UseFED = cms.untracked.bool(True)

process.FedCablingFromConfigDb = cms.ESSource("SiStripFedCablingBuilderFromDb",
    CablingSource = cms.untracked.string('UNDEFINED')  
)

process.SiStripDetInfoFileReader = cms.Service("SiStripDetInfoFileReader")
process.PedestalsFromConfigDb = cms.ESSource("SiStripPedestalsBuilderFromDb")
process.NoiseFromConfigDb = cms.ESSource("SiStripNoiseBuilderFromDb")
process.sistripconn = cms.ESProducer("SiStripConnectivity")

process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")
process.load("Geometry.TrackerNumberingBuilder.trackerTopology_cfi")
process.load("Geometry.TrackerGeometryBuilder.trackerParameters_cfi")

process.FastMonitoringService = cms.Service("FastMonitoringService",
        sleepTime = cms.untracked.int32(1),
        microstateDefPath = cms.untracked.string( cmsswbase+'/src/EventFilter/Utilities/plugins/microstatedef.jsd'),
        fastMicrostateDefPath = cms.untracked.string( cmsswbase+'/src/EventFilter/Utilities/plugins/microstatedeffast.jsd'),
        fastName = cms.untracked.string( 'fastmoni' ),
        slowName = cms.untracked.string( 'slowmoni' )
)

process.EvFDaqDirector = cms.Service("EvFDaqDirector",
        runNumber = cms.untracked.uint32(RUNNUMBER),
        buBaseDir = cms.untracked.string(inputPath),
        directorIsBu = cms.untracked.bool(False),
        testModeNoBuilderUnit = cms.untracked.bool(False)
)

infilename = "file:"+inputPath+"/runRUNNUMBER/runRUNNUMBER.root"
process.source = cms.Source("PoolSource",
        fileNames = cms.untracked.vstring(infilename)
)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

### for run types != from DAQ SCOPE Modes                                                                                                                                                              
if runtype != 15:
    process.load("EventFilter.SiStripRawToDigi.FedChannelDigis_cfi")
    process.FedChannelDigis.UnpackBadChannels = cms.bool(True)
    process.FedChannelDigis.DoAPVEmulatorCheck = cms.bool(True)
    process.FedChannelDigis.LegacyUnpacker = cms.bool(False)
    process.FedChannelDigis.ProductLabel = cms.InputTag("rawDataCollector")
else:
    process.load('DQM.SiStripMonitorHardware.SiStripSpyUnpacker_cfi')
    process.load('DQM.SiStripMonitorHardware.SiStripSpyDigiConverter_cfi')
    process.load('DQM.SiStripMonitorHardware.SiStripSpyEventSummaryProducer_cfi')
    ## * Scope digi settings                                                                                                                                                                          
    process.SiStripSpyUnpacker.FEDIDs = cms.vuint32()                   #use a subset of FEDs or leave empty for all.                                                                                  
    process.SiStripSpyUnpacker.InputProductLabel = cms.InputTag('rawDataCollector')
    process.SiStripSpyUnpacker.AllowIncompleteEvents = True
    process.SiStripSpyUnpacker.StoreCounters = True
    process.SiStripSpyUnpacker.StoreScopeRawDigis = cms.bool(True)      # Note - needs to be True for use in other modules.                                                                            
    ## * Module digi settings                                                                                                                                                                         
    process.SiStripSpyDigiConverter.InputProductLabel = cms.InputTag('SiStripSpyUnpacker','ScopeRawDigis')
    process.SiStripSpyDigiConverter.StorePayloadDigis = True
    process.SiStripSpyDigiConverter.StoreReorderedDigis = True
    process.SiStripSpyDigiConverter.StoreModuleDigis = True
    process.SiStripSpyDigiConverter.StoreAPVAddress = True
    process.SiStripSpyDigiConverter.MinDigiRange = 100
    process.SiStripSpyDigiConverter.MaxDigiRange = 1024
    process.SiStripSpyDigiConverter.MinZeroLight = 0
    process.SiStripSpyDigiConverter.MaxZeroLight = 1024
    process.SiStripSpyDigiConverter.MinTickHeight = 0
    process.SiStripSpyDigiConverter.MaxTickHeight = 1024
    process.SiStripSpyDigiConverter.ExpectedPositionOfFirstHeaderBit = 0
    process.SiStripSpyDigiConverter.DiscardDigisWithWrongAPVAddress = False
    process.SiStripSpyEventSummary.RawDataTag = cms.InputTag('rawDataCollector')


process.load("DQM.SiStripCommissioningSources.CommissioningHistos_cfi")
process.CommissioningHistos.CommissioningTask = 'UNDEFINED'
process.CommissioningHistos.PedsFullNoiseParameters.NrEvToSkipAtStart = 100
process.CommissioningHistos.PedsFullNoiseParameters.NrEvForPeds       = 3000
process.CommissioningHistos.PedsFullNoiseParameters.FillNoiseProfile  = True

if runtype != 15:
    process.p = cms.Path(process.FedChannelDigis*process.CommissioningHistos)
else:

    process.SiStripSpyEventSummary.RunType = cms.uint32(runtype)
    process.CommissioningHistos.InputModuleLabel = 'SiStripSpyDigiConverter'  # output label from spy converter                                                                                        
    process.CommissioningHistos.InputModuleLabelAlt = cms.string('SiStripSpyUnpacker')
    process.CommissioningHistos.SummaryInputModuleLabel = 'SiStripSpyEventSummary'
    process.CommissioningHistos.isSpy = cms.bool(True)
    process.CommissioningHistos.PartitionName = cms.string('DBPART')

    process.p = cms.Path(process.SiStripSpyUnpacker*process.SiStripSpyDigiConverter*process.SiStripSpyEventSummary*process.CommissioningHistos)

