import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Phase2C12_cff import Phase2C12
process = cms.Process("GEMCSCLUT", Phase2C12)

process.load('Configuration.StandardSequences.MagneticField_cff')
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load('Configuration.Geometry.GeometryExtended2026D74Reco_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic', '')

process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32(1)
)

process.source = cms.Source("EmptySource")

process.GEMCSCLUTAnalyzer = cms.EDAnalyzer("GEMCSCLUTAnalyzer")

process.p = cms.Path(process.GEMCSCLUTAnalyzer)
