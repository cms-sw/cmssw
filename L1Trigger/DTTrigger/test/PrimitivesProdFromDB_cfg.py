import FWCore.ParameterSet.Config as cms

process = cms.Process("L1DTTrigProd")

### INCLUDEs 
process.load("Configuration.StandardSequences.GeometryDB_cff")

process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cff")
process.load("Geometry.DTGeometry.dtGeometry_cfi")
process.DTGeometryESModule.applyAlignment = False
process.load("L1TriggerConfig.DTTPGConfigProducers.L1DTTPGConfigFromDB_cff")
#process.load("L1TriggerConfig.DTTPGConfigProducers.L1DTTPGConfig_cff")

#from L1TriggerConfig.DTTPGConfigProducers.L1DTTPGConfigFromDB_cff import *
#L1DTConfigFromDB.UseT0 = True

### GLOBAL TAG
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')

##### MONTE CARLO RUNS - start
#process.GlobalTag.globaltag = 'MC_38Y_V9'
process.GlobalTag.globaltag = "START311_V1"
# include CCB tags - temporary untill they are not in GlobalTag
process.GlobalTag.toGet = cms.VPSet()
process.GlobalTag.toGet.append(
 cms.PSet(
        record = cms.string('DTCCBConfigRcd'),
        tag = cms.string('DTCCBConfig_NOSingleL_V03_mc'),
#        tag = cms.string('DTCCBConfig_Hanytheta_V01_mc'),
        connect = cms.untracked.string('frontier://FrontierProd/CMS_COND_31X_DT')
))
process.GlobalTag.toGet.append(
 cms.PSet(
        record = cms.string('DTKeyedConfigListRcd'),
        tag = cms.string('DTKeyedConfigList_V01_mc'),
        connect = cms.untracked.string('frontier://FrontierProd/CMS_COND_31X_DT')
))
process.GlobalTag.toGet.append(
 cms.PSet(
        record = cms.string('DTKeyedConfigContainerRcd'),
        tag = cms.string('DTKeyedConfig_NOSingleL_V03_mc'),
        connect = cms.untracked.string('frontier://FrontierProd/CMS_COND_31X_DT')
))
process.GlobalTag.toGet.append(
 cms.PSet(
        record = cms.string('DTTPGParametersRcd'),
        tag = cms.string('DTTPGParameters_V01_mc'),
        connect = cms.untracked.string('frontier://FrontierProd/CMS_COND_31X_DT')
))
##### MONTE CARLO RUNS - end

##### DATA RUNS - start
#process.GlobalTag.globaltag = "START38_V12"
## include CCB tags - temporary untill they are not in GlobalTag
#process.GlobalTag.toGet = cms.VPSet()
#process.GlobalTag.toGet.append(
# cms.PSet(
#        record = cms.string('DTCCBConfigRcd'),
#        tag = cms.string('DTCCBConfig_V05_hlt'),
#        connect = cms.untracked.string('frontier://FrontierProd/CMS_COND_31X_DT')
#))
#process.GlobalTag.toGet.append(
# cms.PSet(
#        record = cms.string('DTKeyedConfigListRcd'),
#        tag = cms.string('DTKeyedConfigList_V01_hlt'),
#        connect = cms.untracked.string('frontier://FrontierProd/CMS_COND_31X_DT')
#))
#process.GlobalTag.toGet.append(
# cms.PSet(
#        record = cms.string('DTKeyedConfigContainerRcd'),
#        tag = cms.string('DTKeyedConfig_V01_hlt'),
#        connect = cms.untracked.string('frontier://FrontierProd/CMS_COND_31X_DT')
#))
#process.GlobalTag.toGet.append(
# cms.PSet(
#        record = cms.string('DTTPGParametersRcd'),
#        tag = cms.string('DTTPGParameters_V01_hlt'),
#        connect = cms.untracked.string('frontier://FrontierProd/CMS_COND_31X_DT')
#))
##### DATA RUNS - end


### INPUT SOURCE
## 110112 SV for testing, loads an empty source and gives run number ONLY
#process.source = cms.Source("EmptySource",
#    numberEventsInRun = cms.untracked.uint32(1),
#    firstRun = cms.untracked.uint32(148300)
#    firstRun = cms.untracked.uint32(148380)
#)

# MC source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:/tmp/svanini/FCAE17F3-E9BF-DF11-9994-001731EF61B4.root')
)

# data source
#...

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10000)
)

### PRODUCER
process.dtTriggerPrimitiveDigis = cms.EDProducer("DTTrigProd",
    debug = cms.untracked.bool(False),
# mu digi tag for data 
#    digiTag = cms.InputTag("hltMuonDTDigis"),
# mu digi tag for MC 
    digiTag = cms.InputTag('simMuonDTDigis'),
    DTTFSectorNumbering = cms.bool(True),
    lutBtic = cms.untracked.int32(31),
    lutDumpFlag = cms.untracked.bool(False)
)

### configuration tester 
process.dtDTPTester = cms.EDAnalyzer("DTConfigTester",
    wheel = cms.untracked.int32(0),
    sector = cms.untracked.int32(3),
    station = cms.untracked.int32(4),
    traco = cms.untracked.int32(1),
    bti = cms.untracked.int32(10)
)  

### OUTPUT
process.out = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('drop *', 
        'keep L1MuDTChambPhContainer_*_*_*', 
        'keep L1MuDTChambThContainer_*_*_*'),
    fileName = cms.untracked.string('DTTriggerPrimitives_test.root')
)

#process.p = cms.Path(process.dtTriggerPrimitiveDigis * process.dtDTPTester)
process.p = cms.Path(process.dtTriggerPrimitiveDigis)
process.this_is_the_end = cms.EndPath(process.out)
