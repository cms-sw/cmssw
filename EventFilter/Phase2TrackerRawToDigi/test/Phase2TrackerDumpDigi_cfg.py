import FWCore.ParameterSet.Config as cms
import sys

process = cms.Process("DumpDigi")

# Enable summary at the end of the job
process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

# Limit the number of events to process
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )

# Define the EDAnalyzer with the correct product label
process.Phase2TrackerDumpDigi = cms.EDAnalyzer(
    'Phase2TrackerDumpDigi',
    ProductLabel = cms.InputTag("siPhase2Clusters")
)

# process.TFileService = cms.Service('TFileService', 
#     fileName = cms.string(
#         'Phase2TrackerDumpDigi_original_TTBar.root'
#     ), 
#     closeFileFast = cms.untracked.bool(True)
# )
# 
# process.source = cms.Source("PoolSource",
#     fileNames = cms.untracked.vstring(
# # # # #     "/store/relval/CMSSW_14_0_0_pre2/RelValDisplacedSingleMuFlatPt1p5To8/GEN-SIM-DIGI-RAW/133X_mcRun4_realistic_v1_STD_2026D98_noPU_RV229-v1/2580000/3ce31040-55a5-4469-8ee2-16d050bb6ade.root"
#     "/store/relval/CMSSW_14_0_0_pre2/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/PU_133X_mcRun4_realistic_v1_STD_2026D98_PU200_RV229-v1/2580000/0b2b0b0b-f312-48a8-9d46-ccbadc69bbfd.root"
#     )
# )

process.TFileService = cms.Service('TFileService', 
    fileName = cms.string(
        'Phase2TrackerDumpDigi_redigi_PSstripsAndPixels_TTBar.root'
    ), 
    closeFileFast = cms.untracked.bool(True)
)

### Test digis after digi-raw-digi process ###
### Set up the source for testing with the output file from digi-raw-digi
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
    "file:/afs/cern.ch/work/f/fiorendi/private/l1tt/fetch_unpacker/CMSSW_14_2_0_pre3/src/EventFilter/Phase2TrackerRawToDigi/test/raw2clusters_PSstripsAndPixels.root"
    )    
)
# 
# ## #Update the ProductLabel to match the output from the digi-raw-digi process
process.Phase2TrackerDumpDigi.ProductLabel = cms.InputTag("Unpacker", "", "PACKANDUNPACK")
#End test digis after digi-raw-digi process ###


## Load Geometry for the D98 configuration
process.load('Configuration.Geometry.GeometryExtended2026D98Reco_cff')

# Load the standard sequences for conditions and global tags
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag

# Set the GlobalTag (adjust as necessary for your geometry)
process.GlobalTag = GlobalTag(process.GlobalTag, '133X_mcRun4_realistic_v1', '')

process.load("CondCore.CondDB.CondDB_cfi")
process.CondDB.connect = 'frontier://FrontierProd/CMS_CONDITIONS'

process.PoolDBESSource = cms.ESSource("PoolDBESSource",
    process.CondDB,
    DumpStat = cms.untracked.bool(True),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('TrackerDetToDTCELinkCablingMapRcd'),
        tag = cms.string("TrackerDetToDTCELinkCablingMap__OT800_IT711__T33__OTOnly"),
    )),
)
process.es_prefer_local_cabling = cms.ESPrefer("PoolDBESSource", "")


# Define the path to run the EDAnalyzer
process.p = cms.EndPath(process.Phase2TrackerDumpDigi)
