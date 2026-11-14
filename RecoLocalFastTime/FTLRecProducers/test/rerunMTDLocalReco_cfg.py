import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Era_Phase2C26I13M9_cff import Phase2C26I13M9

process = cms.Process("MTDRERECO", Phase2C26I13M9)

process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(-1))
process.source = cms.Source(
    "PoolSource",
    fileNames=cms.untracked.vstring("file:fakeFTLClusters.root"),
    inputCommands=cms.untracked.vstring(
        "keep *",
        "drop *_mtdTrackingRecHits_*_*",
    ),
)

# Geometry and conditions must match the input sample.
process.load("Configuration.StandardSequences.Services_cff")
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("Configuration.Geometry.GeometryExtendedRun4D127Reco_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("Configuration.EventContent.EventContent_cff")
process.load("Configuration.StandardSequences.EndOfProcess_cff")

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, "auto:phase2_realistic_T35", "")

process.load("RecoLocalFastTime.FTLClusterizer.MTDCPEESProducers_cff")
process.load("RecoLocalFastTime.FTLClusterizer.mtdMergedClusters_cfi")
process.load("RecoLocalFastTime.FTLRecProducers.mtdTrackingRecHits_cfi")

process.p = cms.Path(process.mtdMergedClusters + process.mtdTrackingRecHits)
process.FEVTDEBUGHLToutput = cms.OutputModule(
    "PoolOutputModule",
    dataset=cms.untracked.PSet(
        dataTier=cms.untracked.string("GEN-SIM-RECO"),
        filterName=cms.untracked.string(""),
    ),
    fileName=cms.untracked.string("file:step3_mtdRereco.root"),
    outputCommands=process.FEVTDEBUGHLTEventContent.outputCommands,
)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.output_step = cms.EndPath(process.FEVTDEBUGHLToutput)
process.schedule = cms.Schedule(process.p, process.endjob_step, process.output_step)
process.options = cms.untracked.PSet(
    numberOfThreads=cms.untracked.uint32(1),
    numberOfStreams=cms.untracked.uint32(1),
    wantSummary=cms.untracked.bool(True),
)
