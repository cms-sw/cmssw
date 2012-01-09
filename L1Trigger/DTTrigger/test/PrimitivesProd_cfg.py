import FWCore.ParameterSet.Config as cms

process = cms.Process("L1DTTrigProd")

process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cff")
process.load("Geometry.DTGeometry.dtGeometry_cfi")
process.DTGeometryESModule.applyAlignment = False

process.load("L1Trigger.DTTrigger.dtTriggerPrimitiveDigis_cfi")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:digi.root')
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

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
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
    DTTFSectorNumbering = cms.bool(True),
    lut_btic = cms.untracked.int32(31),
    lut_dump_flag = cms.untracked.bool(False)
)

process.out = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('drop *', 
        'keep L1MuDTChambPhContainer_*_*_*', 
        'keep L1MuDTChambThContainer_*_*_*'),
    fileName = cms.untracked.string('DTTriggerPrimitives.root')
)

process.p = cms.Path(process.dtTriggerPrimitiveDigis)
process.this_is_the_end = cms.EndPath(process.out)

