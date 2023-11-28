import FWCore.ParameterSet.Config as cms

process = cms.Process('RECO')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.cerr.threshold = "DEBUG"
# enable LogDebug messages only for specific modules
process.MessageLogger.debugModules = ["Totem"]

process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run3_data', '')

#dummy = cms.untracked.FileInPath('RecoPPS/Local/data/run364983_ls0001_streamA_StorageManager.dat'),

# raw data source
process.source = cms.Source("NewEventStreamFileReader",
fileNames = cms.untracked.vstring('http://cmsrep.cern.ch/cmssw/download/data/RecoPPS/Local/V1/run364983_ls0001_streamA_StorageManager.dat'
#        '/store/group/dpg_ctpps/comm_ctpps/TotemT2/RecoTest/run364983_ls0001_streamA_StorageManager.dat',
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

# raw-to-digi conversion
process.load('EventFilter.CTPPSRawToDigi.totemT2Digis_cfi')
process.totemT2Digis.rawDataTag = cms.InputTag("rawDataCollector")
process.totemT2Digis.RawUnpacking.verbosity = 1
process.totemT2Digis.RawToDigi.verbosity = 3
process.totemT2Digis.RawToDigi.useOlderT2TestFile = True
process.totemT2Digis.RawToDigi.printUnknownFrameSummary = True
process.totemT2Digis.RawToDigi.printErrorSummary = True

# rechits production
#process.load('Geometry.ForwardCommonData.totemT22021V2XML_cfi')
#process.load('Geometry.ForwardGeometry.totemGeometryESModule_cfi')
#process.load('RecoPPS.Local.totemT2RecHits_cfi')

process.output = cms.OutputModule("PoolOutputModule",
        fileName = cms.untracked.string("file:output-miniDaq2303-T2testFile-ver2.1-noRecHits--1ev.root"),
    outputCommands = cms.untracked.vstring(
        'drop *',
        'keep *_totemT2*_*_*',
    ),
)

# execution configuration
process.p = cms.Path(
    process.totemT2Digis
#    * process.totemT2RecHits
)

process.outpath = cms.EndPath(process.output)
