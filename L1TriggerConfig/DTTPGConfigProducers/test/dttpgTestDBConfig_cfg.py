import FWCore.ParameterSet.Config as cms

process = cms.Process("L1DTTrigProd")

process.load("L1TriggerConfig.DTTPGConfigProducers.L1DTTPGConfigFromDB_cff")

process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(3),
    firstRun = cms.untracked.uint32(147000)
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(9)
)

process.dumpConfig = cms.EDAnalyzer("DTConfigTester",
    wheel   = cms.untracked.int32(0),
    sector  = cms.untracked.int32(4),
    station = cms.untracked.int32(1),
    traco = cms.untracked.int32(2),
    bti = cms.untracked.int32(9)
)

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = "GR10_P_V7::All"

process.tpgPedestalsDB = cms.ESSource("PoolDBESSource",
     process.CondDBSetup,
     authenticationMethod = cms.untracked.uint32(0),
     toGet = cms.VPSet(cms.PSet(
         record = cms.string('DTTPGParametersRcd'),
         tag = cms.string('Pedestals_data')
     )),
     connect = cms.string('sqlite_file:dttpgPedestals.db')
)

process.tpgConfigDB = cms.ESSource("PoolDBESSource",
    loadAll = cms.bool(True),
    timetype = cms.string('runnumber'),
    toGet = cms.VPSet(
    cms.PSet(record = cms.string('DTCCBConfigRcd'),
             tag = cms.string('DT_conf_Hanytheta')
             ),
    cms.PSet(record = cms.string('DTKeyedConfigListRcd'),
             tag = cms.string('keyedConfListIOV_V01')
             ),
    cms.PSet(record = cms.string('DTKeyedConfigContainerRcd'),
             tag = cms.string('keyedConfBricks_V01')
             ) 
    ),
    connect = cms.string('sqlite_file:dttpgConf.db'),
    DBParameters = cms.PSet(authenticationPath = cms.untracked.string('.'),
                            messageLevel = cms.untracked.int32(0)
    )
)


process.p = cms.Path(process.dumpConfig)

