import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing

import Configuration.Geometry.defaultPhase2ConditionsEra_cff as _settings
import Geometry.MTDCommonData.defaultMTDConditionsEra_cff as _mtdgeo
_mtdgeo.check_mtdgeo()
_PH2_GLOBAL_TAG, _PH2_ERA = _settings.get_era_and_conditions(_mtdgeo.MTD_DEFAULT_VERSION)
from Configuration.ProcessModifiers.dd4hep_cff import dd4hep

options = VarParsing("analysis")
options.outputFile = "fakeFTLClusters.root"
options.parseArguments()

process = cms.Process("FAKEFTL", _PH2_ERA,dd4hep)
process.load("Configuration.StandardSequences.Services_cff")
process.load("Geometry.MTDCommonData.GeometryDD4hepExtendedRun4MTDDefaultReco_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, _PH2_GLOBAL_TAG, "")

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(1))
process.mtdClusters = cms.EDProducer("TestWriteFTLClusters")
process.p = cms.Path(process.mtdClusters)
process.out = cms.OutputModule(
    "PoolOutputModule",
    fileName=cms.untracked.string(options.outputFile),
    outputCommands=cms.untracked.vstring("drop *", "keep *_mtdClusters_*_*"),
)
process.e = cms.EndPath(process.out)
