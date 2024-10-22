import FWCore.ParameterSet.Config as cms

process = cms.Process("L1DTTrigProd")
process.load("Configuration.StandardSequences.GeometryDB_cff")
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cff")
process.load("Geometry.DTGeometry.dtGeometry_cfi")
process.DTGeometryESModule.applyAlignment = False

process.load("L1Trigger.DTTrigger.dtTriggerPrimitiveDigis_cfi")

process.dtTriggerPrimitiveDigis.digiTag = "hltMuonDTDigis"


process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/store/relval/CMSSW_3_9_2/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V3-v1/0068/90B5EFB8-24E8-DF11-A195-001A92971B7C.root')
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
    input = cms.untracked.int32(1000)
)

process.out = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('drop *', 
        'keep L1MuDTChambPhContainer_*_*_*', 
        'keep L1MuDTChambThContainer_*_*_*'),
    fileName = cms.untracked.string('DTTriggerPrimitives.root')
)

process.p = cms.Path(process.dtTriggerPrimitiveDigis)
process.this_is_the_end = cms.EndPath(process.out)

