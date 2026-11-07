## Cfg file to run the Phase2TrackerDumpClusters plugin:
## will produce ntuples with cluster properties
## Runs on .root file written by SLinkProducerAndUnpacker_cfg.py,
## either analyzing original clusters or those obtained from the packer+unpacker
## sequence, depending on value of ANALYZE_PACKUNPACK variable below.

import FWCore.ParameterSet.Config as cms

process = cms.Process("Analysis")

# If this is True, then the clusters created by running the packer + unpacker on
# the original clusters will be analyzed.
# If it is False, then the original clusters will be analyzed.
ANALYZE_PACKUNPACK = False
# If this is True, the clusters from the unpacked CRack data will be analyzed.
ANALYZE_CRACK = False

# Enable summary at the end of the job
process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

# Limit the number of events to process
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

# Define the EDAnalyzer with the correct product label
process.ClusterAnalyzer = cms.EDAnalyzer('ClusterAnalyzer',
    ProductLabel = cms.InputTag("hltSiPhase2Clusters")
)

if ANALYZE_PACKUNPACK:

  print("\n === Analyzing clusters created by pack + unpack sequence ===\n")
  
  inputFileList = ["file:raw2clusters.root"]

  # Update label to match the output from the digi-raw-digi process
  process.ClusterAnalyzer.ProductLabel = cms.InputTag("Unpacker", "", "PACKANDUNPACK")

elif ANALYZE_CRACK:

  print("\n === Analyzing clusters created by CRack unpacker sequence ===\n")

  inputFileList = ["file:/home/hep/am2023/sara_crack_july_2026/CMSSW_16_0_8/src/Unpacker_CRACK_Physics_Run_September_2026.root"]
                              
  # Update label to match the output from the unpacker process
  process.ClusterAnalyzer.ProductLabel = cms.InputTag("Unpacker", "", "UNPACK")

else:
  print("\n === Analyzing original clusters ===\n")

  inputFileList = ["/store/relval/CMSSW_20_0_0_pre1/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/PU_150X_mcRun4_realistic_v1_STD_D121_RegeneratedGS_PU_16Aug26-v3/2590000/0438e4bc-b740-48a4-9d02-ff7896522eac.root"]

  # Read original clusters from input dataset
  process.ClusterAnalyzer.ProductLabel = cms.InputTag("hltSiPhase2Clusters")
  # Read clusters remade from digis by SLinkProducerAndUnpacker_cfg.py.
  #process.ClusterAnalyzer.ProductLabel = cms.InputTag("siPhase2Clusters", "", "PACKANDUNPACK")

process.source = cms.Source("PoolSource",
   fileNames = cms.untracked.vstring(*inputFileList)
)  

# Create output root file for TTree.
process.TFileService = cms.Service('TFileService', 
    fileName = cms.string(
        'ClusterAnalyzer_TTree.root'
    ), 
    closeFileFast = cms.untracked.bool(True)
)


# Load the standard sequences for conditions and global tags
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load("CondCore.CondDB.CondDB_cfi")
from Configuration.AlCa.GlobalTag import GlobalTag

if not ANALYZE_CRACK:
    process.load('Configuration.Geometry.GeometryExtendedRun4D121Reco_cff')
    # Set the GlobalTag (adjust as necessary for your geometry)
    process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic', '')
    ## the following won't be needed anymore once the cabling map generated with the updated TrackerDetToDTCELinkCablingMapRcd class is included in the GT
    process.CondDB.connect = 'sqlite_file:/afs/cern.ch/user/f/fiorendi/public/l1tt/unpacker/crack/OTCablingMap_newClass.db'


else:
    ## customise for C-rack geometry
    process.load('Configuration.Geometry.GeometryExtendedRun4D500Reco_cff')
    process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic_0T', '')
    process.CondDB.connect = 'sqlite_file:/home/hep/am2023/sara_crack_july_2026/CMSSW_16_0_8/src/crack_cabling_gIDbtFrom0.db'
    process.trackerGeometry.applyAlignment = False


## the following lines will be specific of the C-rack customisation once the updated cabling map for the OT is included in the GT
process.PoolDBESSource = cms.ESSource("PoolDBESSource",
    process.CondDB,
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('TrackerDetToDTCELinkCablingMapRcd'),
        tag = cms.string("DTCCablingMapProducerUserRun"))
    )
)
process.es_prefer_local_TrackerDetToDTCELinkCablingMapRcd = cms.ESPrefer("PoolDBESSource","")


# Define the path to run the EDAnalyzer
process.p = cms.Path(process.ClusterAnalyzer)
