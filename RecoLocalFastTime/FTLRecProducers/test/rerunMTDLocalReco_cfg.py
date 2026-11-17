import FWCore.ParameterSet.Config as cms

import Configuration.Geometry.defaultPhase2ConditionsEra_cff as _settings
import Geometry.MTDCommonData.defaultMTDConditionsEra_cff as _mtdgeo
_mtdgeo.check_mtdgeo()
_PH2_GLOBAL_TAG, _PH2_ERA = _settings.get_era_and_conditions(_mtdgeo.MTD_DEFAULT_VERSION)
from Configuration.ProcessModifiers.dd4hep_cff import dd4hep

process = cms.Process("MTDRERECO", _PH2_ERA,dd4hep)

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
process.load("Geometry.MTDCommonData.GeometryDD4hepExtendedRun4MTDDefaultReco_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("Configuration.EventContent.EventContent_cff")
process.load("Configuration.StandardSequences.EndOfProcess_cff")

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, _PH2_GLOBAL_TAG, "")

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
