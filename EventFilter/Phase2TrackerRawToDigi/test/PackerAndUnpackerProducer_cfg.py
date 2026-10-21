## cfg file to run the (packing and) unpacking step(s) for Phase2 OT clusters
## outputs an EDM file containing the original FEDRawData and the unpacked clusters

import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing
import FWCore.Utilities.FileUtils as FileUtils

process = cms.Process("PACKANDUNPACK")

# If this is False, then the clusters will be created by running the packer + unpacker chain on the original clusters.
# If it is True, then the clusters will be unpacked from (C-Rack) raw data (with corresponding geometry).
UNPACK_CRACK = False

process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
#process.load('FWCore.MessageService.MessageLogger_cfi')

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

process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load("CondCore.CondDB.CondDB_cfi")
from Configuration.AlCa.GlobalTag import GlobalTag

inputRawDataTag = 'Packer'
crackMapping = cms.VPSet()

if UNPACK_CRACK:
    ## customise for C-rack geometry
    process.load('Configuration.StandardSequences.MagneticField_0T_cff')
    process.load('Configuration.Geometry.GeometryExtendedRun4D500Reco_cff')
    process.trackerGeometry.applyAlignment = False
    process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic_0T', '')
    process.CondDB.connect = 'sqlite_file:/afs/cern.ch/user/f/fiorendi/public/l1tt/unpacker/crack/CRackDTCCablingMap_newClass.db'
 
    from EventFilter.Phase2TrackerRawToDigi.CRACKMapping_cfi import crackMapping    
    inputRawDataTag = 'rawDataCollector'
    inputFileList = ["file:/eos/project-c/cms-tracker-integration/www/results/CosmicRackData/2026/08/CRACK_VALIDATION_09_09_2026_FED.root"]

else:
    process.load('Configuration.Geometry.GeometryExtendedRun4D121Reco_cff')
    process.load('Configuration.Geometry.GeometryExtendedRun4D121_cff')
    process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic', '')
    process.CondDB.connect = 'sqlite_file:/afs/cern.ch/user/f/fiorendi/public/l1tt/unpacker/crack/OTCablingMap_newClass.db'
    inputFileList = ["/store/relval/CMSSW_20_0_0_pre1/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/PU_150X_mcRun4_realistic_v1_STD_D121_RegeneratedGS_PU_16Aug26-v3/2590000/0438e4bc-b740-48a4-9d02-ff7896522eac.root"]

process.PoolDBESSource = cms.ESSource("PoolDBESSource",
   process.CondDB,
   DumpStat = cms.untracked.bool(True),
   toGet = cms.VPSet(cms.PSet(
       record = cms.string('TrackerDetToDTCELinkCablingMapRcd'),
       tag = cms.string("DTCCablingMapProducerUserRun"),
   )),
)
process.es_prefer_local_cabling = cms.ESPrefer("PoolDBESSource", "")

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(50))
process.source = cms.Source("PoolSource",
   fileNames = cms.untracked.vstring(*inputFileList)
)



# Should not need to remake clusters, unless MC made with out-of-date clusterizer
#process.load('RecoLocalTracker.SiPhase2Clusterizer.phase2TrackerClusterizer_cfi')

process.Packer = cms.EDProducer("ClusterToRawProducer",
  # Read original clusters from input dataset
  Phase2Clusters = cms.InputTag("hltSiPhase2Clusters")
  # Read clusters remade from digis in this job
  #Phase2Clusters = cms.InputTag("siPhase2Clusters", "",  "PACKANDUNPACK")
)

process.Unpacker = cms.EDProducer("RawToClusterProducer",
    fedDataBuffer = cms.InputTag(inputRawDataTag),
    analyzeCRACK = cms.bool(UNPACK_CRACK),
    crackMapping = crackMapping
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
      'keep *_mix_Tracker_*',
      ),
    fileName = cms.untracked.string('raw2clusters.root')
)

process.Timing = cms.Service("Timing",
    summaryOnly = cms.untracked.bool(True),  # If true, only the summary is printed.
    useJobReport = cms.untracked.bool(True)  # This will also log timings in the job report.
)

if UNPACK_CRACK:
    process.dtc = cms.Path(process.Unpacker)
else:
    process.dtc = cms.Path(process.Packer * process.Unpacker)
    # process.dtc = cms.Path(process.siPhase2Clusters * process.Packer * process.Unpacker)
process.output = cms.EndPath(process.out)

