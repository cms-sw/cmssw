## Initial script to convert from .dat files to root output with EDMCollections - AWB 29.01.16

import FWCore.ParameterSet.Config as cms

process = cms.Process("datFile")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32(100)
process.options = cms.untracked.PSet(wantSummary = cms.untracked.bool(True))

fNames = cms.untracked.vstring('file:/afs/cern.ch/work/a/abrinke1/public/EMTF/miniDAQ/dat_dumps/2015_12_09/263608/run263608_ls0172_streamA_StorageManager.dat')
fNames = cms.untracked.vstring('file:/afs/cern.ch/work/a/abrinke1/public/EMTF/miniDAQ/dat_dumps/2015_12_09/263608/run263608_ls0160_streamA_StorageManager.dat')

process.source = cms.Source(
    "NewEventStreamFileReader",
    fileNames = fNames,
    skipEvents=cms.untracked.uint32(123)
)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10000))

# PostLS1 geometry used
process.load('Configuration.Geometry.GeometryExtended2015Reco_cff')
process.load('Configuration.Geometry.GeometryExtended2015_cff')
############################
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff')
from Configuration.AlCa.GlobalTag_condDBv2 import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')

# ## Debug / error / warning message output
# process.MessageLogger = cms.Service(
#     "MessageLogger",
#     threshold  = cms.untracked.string('DEBUG'),
#     categories = cms.untracked.vstring('L1T'),
#     debugModules = cms.untracked.vstring('*'),
#     )


# dump raw data
process.dump = cms.EDAnalyzer( 
    "DumpFEDRawDataProduct",
    label = cms.untracked.string("rawDataCollector"),
    # feds = cms.untracked.vint32(1402,813),
    # feds = cms.untracked.vint32(1402),
    dumpPayload = cms.untracked.bool ( True )
)

process.unpack = cms.EDProducer("L1TRawToDigi",
        Setup           = cms.string("stage2::EMTFSetup"),
        InputLabel      = cms.InputTag("rawDataCollector"),
        FedIds          = cms.vint32( 1385 ),
        FWId            = cms.uint32(0),
        debug = cms.untracked.bool(False), ## More debugging output
        MTF7 = cms.untracked.bool(True)
)


process.out = cms.OutputModule("PoolOutputModule", 
   outputCommands=cms.untracked.vstring(
       'keep *'
   ),
   fileName = cms.untracked.string("EMTF_DatToRoot_v0.root")
)

process.p = cms.Path(process.dump * process.unpack)
process.end = cms.EndPath(process.out)
