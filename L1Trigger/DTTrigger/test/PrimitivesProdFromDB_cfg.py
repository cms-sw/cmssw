import FWCore.ParameterSet.Config as cms

process = cms.Process("L1DTTrigProd")

process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cff")
process.load("Geometry.DTGeometry.dtGeometry_cfi")
process.DTGeometryESModule.applyAlignment = False
process.load("L1TriggerConfig.DTTPGConfigProducers.L1DTTPGConfigFromDB_cff")

process.DTTPGConfigSource = cms.ESSource("PoolDBESSource",
    loadAll = cms.bool(True),
    timetype = cms.string('runnumber'),
    toGet = cms.VPSet(
                      cms.PSet(record = cms.string('DTCCBConfigRcd'),
                               ##### CHANGED for 3XY #####
                               ###  tag = cms.string('conf_ccb_V01')
                               #### CHANGED for new DB 1005010
                               ###tag = cms.string('DT_config_V02')
			       tag = cms.string('conf_test') 
                               ),
                      cms.PSet(record = cms.string('DTKeyedConfigListRcd'),
                               tag = cms.string('keyedConfListIOV_V01')
                               ),
                      cms.PSet(record = cms.string('DTKeyedConfigContainerRcd'),
                               tag = cms.string('keyedConfBricks_V01')
                              ) 
                      ),
                      ##### CHANGED for 3XY #####
                      #connect = cms.string('sqlite_file:conf999999.db'),
                      #    connect = cms.string('oracle://cms_orcoff_prod/CMS_COND_31X_DT'),
                      #    connect = cms.string('frontier://FrontierProd/CMS_COND_31X_DT'),
                      #### CHANGED for new DB 100510
                      connect = cms.string('sqlite_file:userconf.db'), 
                      ##### CHANGED for 3XY #####
                      ###    token = cms.string('[DB=00000000-0000-0000-0000-000000000000][CNT=DTConfigList][CLID=9CB14BE8-30A2-DB11-9935-000E0C5CE283][TECH=00000B01][OID=00000004-00000000]'),
                      #### CHANGED for new DB 100510
                      #token = cms.string('[DB=00000000-0000-0000-0000-000000000000][CNT=DTConfigList][CLID=9CB14BE8-30A2-DB11-9935-000E0C5CE283][TECH=00000B01][OID=0000000B-00000000]'),
                      DBParameters = cms.PSet(
                            #### CHANGED for new DB 100510
                            #authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb'),
                            authenticationPath = cms.untracked.string('.'),
                            messageLevel = cms.untracked.int32(0)
                      )
)

process.source = cms.Source("PoolSource",
#    fileNames = cms.untracked.vstring('file:digi.root')
#    fileNames = cms.untracked.vstring('/store/relval/CMSSW_3_7_0_pre3/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V2-v1/0019/784E2ED4-3458-DF11-A4ED-001BFCDBD15E.root')
   fileNames = cms.untracked.vstring('file:/tmp/svanini/784E2ED4-3458-DF11-A4ED-001BFCDBD15E.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(200)
)


process.dtTriggerPrimitiveDigis = cms.EDProducer("DTTrigProd",
    debug = cms.untracked.bool(False),
    tTrigModeConfig = cms.PSet(
        debug = cms.untracked.bool(False),
        kFactor = cms.double(-2.0),
        vPropWire = cms.double(24.4),
        tofCorrType = cms.int32(1),
        tTrig = cms.double(500.0)
    ),
    digiTag = cms.InputTag("hltMuonDTDigis"),
    tTrigMode = cms.string('DTTTrigSyncTOFCorr'),
    DTTFSectorNumbering = cms.bool(True),
    lut_btic = cms.untracked.int32(31),
    lut_dump_flag = cms.untracked.bool(False)
)

#process.MessageLogger = cms.Service("MessageLogger",
#    debugModules = cms.untracked.vstring('*'),
#    cout = cms.untracked.PSet(
#        threshold = cms.untracked.string('INFO'),
#        WARNING = cms.untracked.PSet(
#            limit = cms.untracked.int32(0)
#        ),
#        noLineBreaks = cms.untracked.bool(True)
#    ),
#    destinations = cms.untracked.vstring('cout')
#)

process.out = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('drop *', 
        'keep L1MuDTChambPhContainer_*_*_*', 
        'keep L1MuDTChambThContainer_*_*_*'),
    fileName = cms.untracked.string('DTTriggerPrimitives.root')
)

process.p = cms.Path(process.dtTriggerPrimitiveDigis)
process.this_is_the_end = cms.EndPath(process.out)

