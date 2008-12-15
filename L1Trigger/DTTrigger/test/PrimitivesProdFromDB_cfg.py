import FWCore.ParameterSet.Config as cms

process = cms.Process("L1DTTrigProd")


process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cff")
process.load("Geometry.DTGeometry.dtGeometry_cfi")
process.DTGeometryESModule.applyAlignment = False
process.load("L1TriggerConfig.DTTPGConfigProducers.L1DTTPGConfigFromDB_cff")

process.DTTPGConfigSource = cms.ESSource("PoolDBESSource",

    toGet = cms.VPSet(cms.PSet(record = cms.string('DTCCBConfigRcd'),
                               tag = cms.string('conf_ccb_V01')
                               )
                      ),
    loadAll = cms.bool(True),
                                    
    connect = cms.string('sqlite_file:conf67838.db'),
    timetype = cms.string('runnumber'),
    token = cms.string('[DB=00000000-0000-0000-0000-000000000000][CNT=DTConfigList][CLID=9CB14BE8-30A2-DB11-9935-000E0C5CE283][TECH=00000B01][OID=00000004-00000000]'),
    DBParameters = cms.PSet(authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb'),
                            messageLevel = cms.untracked.int32(0)
    )

)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:digi.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
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
    digiTag = cms.InputTag("muonDTDigis"),
    tTrigMode = cms.string('DTTTrigSyncTOFCorr'),
    DTTFSectorNumbering = cms.bool(True)
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

