import FWCore.ParameterSet.Config as cms
import sys

process = cms.Process("DumpDigi")

# Enable summary at the end of the job
process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

# Limit the number of events to process
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )

# Load necessary configurations for Phase2 Tracker Cabling
# process.load('P2TrackerCabling_cfi')

# Define the EDAnalyzer with the correct product label
process.Phase2TrackerDumpDigi = cms.EDAnalyzer(
    'Phase2TrackerDumpDigi',
    ProductLabel = cms.InputTag("siPhase2Clusters")
)
# process.source = cms.Source("PoolSource",
#     fileNames = cms.untracked.vstring("/store/relval/CMSSW_14_0_0_pre2/RelValDisplacedSingleMuFlatPt1p5To8/GEN-SIM-DIGI-RAW/133X_mcRun4_realistic_v1_STD_2026D98_noPU_RV229-v1/2580000/3ce31040-55a5-4469-8ee2-16d050bb6ade.root")
# )

### Test digis after digi-raw-digi process ###
# Set up the source for testing with the output file from digi-raw-digi
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring("file:raw2clusters.root")    
)

# Update the ProductLabel to match the output from the digi-raw-digi process
process.Phase2TrackerDumpDigi.ProductLabel = cms.InputTag("Unpacker", "", "PACKANDUNPACK")
### End test digis after digi-raw-digi process ###

## Load Geometry for the D98 configuration
process.load('Configuration.Geometry.GeometryExtended2026D98Reco_cff')

# Load the standard sequences for conditions and global tags
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag

# Set the GlobalTag (adjust as necessary for your geometry)
process.GlobalTag = GlobalTag(process.GlobalTag, '133X_mcRun4_realistic_v1', '')

# Define the path to run the EDAnalyzer
process.p = cms.EndPath(process.Phase2TrackerDumpDigi)
