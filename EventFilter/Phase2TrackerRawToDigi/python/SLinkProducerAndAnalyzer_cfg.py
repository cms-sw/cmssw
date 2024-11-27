import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing
import FWCore.Utilities.FileUtils as FileUtils
import os

process = cms.Process("Phase2DAQProducer")

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
    print("using geometry " + GEOMETRY + " (tilted)")
    process.load('Configuration.Geometry.GeometryExtended2026' + GEOMETRY + 'Reco_cff')
    process.load('Configuration.Geometry.GeometryExtended2026' + GEOMETRY +'_cff')
else:
    print("this is not a valid geometry!!!")

process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, '133X_mcRun4_realistic_v1', '')
# process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))

DatasetDatabase = "/home/hep/am2023/cmssw_el9_amd64_gcc12/CMSSW_14_0_0_pre2/src/Datasets/Phase2Spring23Track1GeVL1TFix.list"

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))

try:
    InputMC = [get_input_mc_line(DatasetDatabase, options.process)]
except Exception as e:
    print(f"Error: {e}")
    InputMC = []

print(f"InputMC: {InputMC}")
process.source = cms.Source("PoolSource", fileNames = cms.untracked.vstring(*InputMC))

process.load("CondCore.CondDB.CondDB_cfi")
# input database (in this case the local sqlite file)
process.CondDB.connect = 'sqlite_file:OTandITDTCCablingMap_T33.db'

process.PoolDBESSource = cms.ESSource("PoolDBESSource",
    process.CondDB,
    DumpStat=cms.untracked.bool(True),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('TrackerDetToDTCELinkCablingMapRcd'),
        tag = cms.string("DTCCablingMapProducerUserRun")
    )),
)

process.es_prefer_local_cabling = cms.ESPrefer("PoolDBESSource", "")

process.ClustersFromPhase2TrackerDigis = cms.EDProducer("Phase2TrackerClusterizer",
    src = cms.InputTag("mix","Tracker"),
)

process.Experimental = cms.EDProducer("ClusterToRawProducer",
    Phase2Clusters = cms.InputTag("ClustersFromPhase2TrackerDigis"),
)

process.Analyzer = cms.EDAnalyzer("RawAnalyzer",
    fedRawDataCollection = cms.InputTag("Experimental"),
)

from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2
premix_stage2.toModify(process.ClustersFromPhase2TrackerDigis, rawHits = ["mixData:Tracker"])

process.Timing = cms.Service("Timing",
    summaryOnly = cms.untracked.bool(True),  # If true, only the summary is printed.
    useJobReport = cms.untracked.bool(True)  # This will also log timings in the job report.
)

process.dtc = cms.Path(process.ClustersFromPhase2TrackerDigis * process.Experimental * process.Analyzer)