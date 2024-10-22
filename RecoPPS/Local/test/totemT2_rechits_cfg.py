import FWCore.ParameterSet.Config as cms

process = cms.Process('RECO')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
#process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('Geometry.ForwardCommonData.totemT22021V2XML_cfi')
process.load('Geometry.ForwardGeometry.totemGeometryESModule_cfi')
process.load('RecoPPS.Local.totemT2RecHits_cfi')

#from Configuration.AlCa.GlobalTag import GlobalTag
#process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_data', '')

process.source = cms.Source('PoolSource',
    fileNames = cms.untracked.vstring(
        '/store/group/dpg_ctpps/comm_ctpps/TotemT2/RecoTest/emulated_digi_test.root'
    ),
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.output = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string("file:output.root"),
    outputCommands = cms.untracked.vstring(
        'drop *',
        'keep *_totemT2*_*_*',
    ),
)

# execution configuration
process.p = cms.Path(
    process.totemT2RecHits
)

process.outpath = cms.EndPath(process.output)
