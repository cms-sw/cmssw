## cfg file to run the packing and unpacking steps for Phase2 OT clusters
## optionally, also run EDAnalyzer to dump the FEDRawData into a text file
## outputs an EDM file containing the original FEDRawData and the unpacked clusters

import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing
import FWCore.Utilities.FileUtils as FileUtils
import os

# -- alpaka --
# flag for using the conversion
Legacy_Format = True

process = cms.Process("PACKANDUNPACK")

# -- alpaka --
process.options.numberOfThreads = 8  
process.options.numberOfStreams = 8

process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
#process.load('FWCore.MessageService.MessageLogger_cfi')
# -- alpaka --
process.load('Configuration.StandardSequences.Accelerators_cff')
process.load('HeterogeneousCore.AlpakaCore.ProcessAcceleratorAlpaka_cfi')

'''
process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('logUnpacker','cout'),
    categories = cms.untracked.vstring('RawToClusterProducer'),
    debugModules  = cms.untracked.vstring('*'),
    cout = cms.untracked.PSet( # Writes event number also to cout.
        threshold = cms.untracked.string("INFO"),
        INFO = cms.untracked.PSet(limit = cms.untracked.int32(0)),
    ),                                     
    logUnpacker = cms.untracked.PSet( # Write output to file
        enableStatistics = cms.untracked.bool(True),     
        threshold = cms.untracked.string('DEBUG'),
        INFO =  cms.untracked.PSet(limit = cms.untracked.int32(0)),
        DEBUG = cms.untracked.PSet(limit = cms.untracked.int32(0)),
        RawToClusterProducer = cms.untracked.PSet(limit = cms.untracked.int32(-1))
    ),
)
'''
process.load('Configuration.Geometry.GeometryExtendedRun4D110Reco_cff')
process.load('Configuration.Geometry.GeometryExtendedRun4D110_cff')

process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

from Configuration.AlCa.GlobalTag import GlobalTag
#process.GlobalTag = GlobalTag(process.GlobalTag, '133X_mcRun4_realistic_v1', '')
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic', '')

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(100))

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring("/store/relval/CMSSW_15_1_0_pre5/RelValTTbar_14TeV_TuneCP5/GEN-SIM-DIGI-RAW/PU_150X_mcRun4_realistic_v1_RV269_Run4D110_PU-v2/2590000/0f0bcfd3-dafe-4dda-8d39-9765f6eae68e.root")
     #fileNames = cms.untracked.vstring("/store/relval/CMSSW_15_1_0_pre5/RelValDoubleMuFlatPt1p5To8/GEN-SIM-DIGI-RAW/150X_mcRun4_realistic_v1_RV269_Run4D110_noPU-v1/2590000/1172421f-823f-420f-8ec9-3de20dd6dda4.root")
     )

#--------------------------------------------------------
## in case of local DB file
#--------------------------------------------------------
process.load("CondCore.CondDB.CondDB_cfi")
process.CondDB.connect = 'sqlite_file:/afs/cern.ch/user/f/fiorendi/public/l1tt/unpacker/crack/OTCablingMap_newClass.db'
#process.CondDB.connect = 'frontier://FrontierProd/CMS_CONDITIONS'

process.PoolDBESSource = cms.ESSource("PoolDBESSource",
   process.CondDB,
   DumpStat = cms.untracked.bool(True),
   toGet = cms.VPSet(cms.PSet(
       record = cms.string('TrackerDetToDTCELinkCablingMapRcd'),
       tag = cms.string("DTCCablingMapProducerUserRun"),
   )),
)

process.es_prefer_local_cabling = cms.ESPrefer("PoolDBESSource", "")
#-------------------------------------------------------

# Should not need to remake clusters, unless MC made with out-of-date clusterizer
#process.load('RecoLocalTracker.SiPhase2Clusterizer.phase2TrackerClusterizer_cfi')

process.Packer = cms.EDProducer("ClusterToRawProducer",
  # Read original clusters from input dataset
  Phase2Clusters = cms.InputTag("hltSiPhase2Clusters")
  # Read clusters remade from digis in this job
  #Phase2Clusters = cms.InputTag("siPhase2Clusters", "",  "PACKANDUNPACK")
)

process.Analyzer = cms.EDAnalyzer("RawAnalyzer",
    fedDataBuffer = cms.InputTag("Packer")
)
process.Unpacker = cms.EDProducer("RawToClusterProducer",
    fedDataBuffer = cms.InputTag("Packer")
)
# -- alpaka --
process.alpakaUnpacker = cms.EDProducer("Phase2RawToClusterProducer@alpaka",
#process.alpakaUnpacker = cms.EDProducer("alpaka_serial_sync::Phase2RawToClusterProducer",
#process.alpakaUnpacker = cms.EDProducer("alpaka_cuda_async::Phase2RawToClusterProducer",
#process.alpakaUnpacker = cms.EDProducer("alpaka_rocm_async::Phase2RawToClusterProducer",
    fedRawDataCollection = cms.InputTag("Packer"),
)
process.ClusterConverter = cms.EDProducer("ClusterPropSoAToLegacyED",
    clusterSoASource = cms.InputTag("alpakaUnpacker")  
)

process.out = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880),                              
    outputCommands = cms.untracked.vstring('drop *',
      'keep RawDataBuffer_*_*_*',
      'keep Phase2TrackerCluster1D*_*_*_*',
      'keep *_remadeSiPhase2Clusters_*_*',
      'keep *_Packer_*_*',
      'keep *_Unpacker_*_*',
      'keep *_alpakaUnpacker_*_*',
      'keep *_ClusterConverter_*_*',
      'keep *_mix_Tracker_*',
      ),
    fileName = cms.untracked.string('raw2clusters.root')
)

process.Timing = cms.Service("Timing",
    summaryOnly = cms.untracked.bool(True),  # If true, only the summary is printed.
    useJobReport = cms.untracked.bool(True)  # This will also log timings in the job report.
)

#process.dtc = cms.Path(process.Packer * process.Unpacker)
#process.dtc = cms.Path(process.Packer * process.Analyzer * process.Unpacker)
#process.dtc = cms.Path(process.siPhase2Clusters * process.Packer * process.Unpacker)
# -- alpaka --
if Legacy_Format:
    # run with legacy conversion step
    process.dtc = cms.Path(
        process.Packer *
        process.Unpacker *
        process.alpakaUnpacker *
        process.ClusterConverter
    )
else:
    # run without legacy conversion,
    process.dtc = cms.Path(
        process.Packer *
        process.Unpacker *
        process.alpakaUnpacker
    )

process.output = cms.EndPath(process.out)

