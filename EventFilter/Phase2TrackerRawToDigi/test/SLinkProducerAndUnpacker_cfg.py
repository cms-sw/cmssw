#!/cvmfs/cms.cern.ch/el9_amd64_gcc12/cms/cmssw/CMSSW_14_0_0_pre2/bin/el9_amd64_gcc12/cmsRun
## Stub Occupancy Script
import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing
import FWCore.Utilities.FileUtils as FileUtils
import os

process = cms.Process("PACKANDUNPACK")

def get_input_mc_line(dataset_database, line_number):
    with open(dataset_database, 'r') as file:
        lines = file.readlines()
        if line_number < 0 or line_number >= len(lines):
            raise IndexError("Line number out of range")
        return lines[line_number].strip()

options = VarParsing.VarParsing('analysis')

# Add custom command-line arguments
options.register('cluster',
                 0, # default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Cluster ID from HTCondor")

options.register('process',
                 0, # default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Process ID from HTCondor")

# Parse command-line arguments
options.parseArguments()

#GEOMETRY = "D88"
GEOMETRY = "D98"

process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.A = dict(limit = -1)

if GEOMETRY == "D88" or GEOMETRY == 'D98':
#     print("using geometry " + GEOMETRY + " (tilted)")
    process.load('Configuration.Geometry.GeometryExtended2026' + GEOMETRY + 'Reco_cff')
    process.load('Configuration.Geometry.GeometryExtended2026' + GEOMETRY +'_cff')
else:
    print("this is not a valid geometry!!!")

process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

from Configuration.AlCa.GlobalTag import GlobalTag
# process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic', '')
process.GlobalTag = GlobalTag(process.GlobalTag, '133X_mcRun4_realistic_v1', '')
# process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))

DatasetDatabase = "/home/hep/am2023/cmssw_el9_amd64_gcc12/CMSSW_14_0_0_pre2/src/Datasets/Phase2Spring23Track1GeVL1TFix.list"

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))

# try:
#     InputMC = [get_input_mc_line(DatasetDatabase, options.process)]
# except Exception as e:
#     print(f"Error: {e}")
#     InputMC = []
# 
# print(f"InputMC: {InputMC}")
# process.source = cms.Source("PoolSource", fileNames = cms.untracked.vstring(*InputMC))
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring("/store/relval/CMSSW_14_0_0_pre2/RelValDisplacedSingleMuFlatPt1p5To8/GEN-SIM-DIGI-RAW/133X_mcRun4_realistic_v1_STD_2026D98_noPU_RV229-v1/2580000/3ce31040-55a5-4469-8ee2-16d050bb6ade.root")
 )

## in case of local file
# process.load("CondCore.CondDB.CondDB_cfi")
# process.CondDB.connect = 'sqlite_file:/afs/cern.ch/work/f/fiorendi/private/l1tt/unpacker_retry/CMSSW_14_1_0_pre7/src/Phase2RawToDigi/Phase2DAQProducer/python/OTandITDTCCablingMap_T33.db'
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

process.ClustersFromPhase2TrackerDigis = cms.EDProducer("Phase2TrackerClusterizer",
    src = cms.InputTag("mix","Tracker"),
)

process.Experimental = cms.EDProducer("Phase2DAQProducer",
    Phase2Clusters = cms.InputTag("ClustersFromPhase2TrackerDigis"),
)

process.Analyzer = cms.EDAnalyzer("Phase2DAQAnalyzer",
    fedRawDataCollection = cms.InputTag("Experimental"),
)
process.Unpacker = cms.EDProducer("RawToClusterProducer",
    fedRawDataCollection = cms.InputTag("Experimental"),
)

process.out = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880),                              
    outputCommands = cms.untracked.vstring('drop *',
      'keep FEDRawDataCollection_*_*_*',
      'keep *_ClustersFromPhase2TrackerDigis_*_*',
      'keep *_Experimental_*_*',
      'keep *_Unpacker_*_*',
      'keep *_mix_Tracker_*',
      ),
    fileName = cms.untracked.string('raw2clusters.root')
)



from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2
premix_stage2.toModify(process.ClustersFromPhase2TrackerDigis, rawHits = ["mixData:Tracker"])

process.Timing = cms.Service("Timing",
    summaryOnly = cms.untracked.bool(True),  # If true, only the summary is printed.
    useJobReport = cms.untracked.bool(True)  # This will also log timings in the job report.
)

process.dtc = cms.Path(process.ClustersFromPhase2TrackerDigis * process.Experimental * process.Unpacker)
process.output = cms.EndPath(process.out)
# process.dtc = cms.Path(process.ClustersFromPhase2TrackerDigis * process.Experimental * process.Analyzer)
# process.dtc = cms.Path(process.ClustersFromPhase2TrackerDigis * process.Experimental * process.Analyzer * process.Unpacker)

