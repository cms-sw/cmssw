import FWCore.ParameterSet.Config as cms

process = cms.Process("myCTPPSRawToDigi")

process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("CondFormats.CTPPSReadoutObjects.CTPPSPixelDAQMappingESSourceXML_cfi")
process.ctppsPixelDAQMappingESSourceXML.configuration = cms.VPSet(
    # example configuration block:
    cms.PSet(
        validityRange = cms.EventRange("1:min - 999999999:max"),
        mappingFileNames = cms.vstring("CondFormats/CTPPSReadoutObjects/xml/rpix_mapping_220_far_TEST.xml"),
        maskFileNames = cms.vstring("CondFormats/CTPPSReadoutObjects/xml/rpix_channel_mask_TEST.xml")
        )
    )

process.load("Configuration.StandardSequences.Services_cff")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1))

process.source = cms.Source("PoolSource",
# fileNames =  cms.untracked.vstring('file:rawdata.root')
labelRawDataLikeMC = cms.untracked.bool(False),
fileNames =  cms.untracked.vstring(
#"file:./PixelAlive_1294_153_RAW_v3.root"
"file:/afs/cern.ch/work/k/kas/public/PXtrees/PixelAlive_1294_151_RAW_v2.root"

 )
)

process.load("EventFilter.CTPPSRawToDigi.ctppsPixelRawToDigi_cfi")

process.ctppsPixelDigis.InputLabel = 'source'

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('ctppsPixelDigis'),
    destinations = cms.untracked.vstring('r2d'),
    r2d = cms.untracked.PSet( threshold = cms.untracked.string('DEBUG'))
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName =  cms.untracked.string('file:digis_PixelAlive_1294_151_RAW_v2.root'),
    outputCommands = cms.untracked.vstring("keep *")
)

process.p = cms.Path(process.ctppsPixelDigis)
process.ep = cms.EndPath(process.out)
