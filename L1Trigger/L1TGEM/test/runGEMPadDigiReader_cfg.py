import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run2_2016_cff import Run3
process = cms.Process("Dump", Run3)

process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.Geometry.GeometryExtended2021Reco_cff')
process.load('Configuration.Geometry.GeometryExtended2021_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc')

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.load('L1Trigger.L1TGEM.simPadDigis_cfi')

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'file:out_l1.root'
    )
)

process.dumper = cms.EDAnalyzer("GEMPadDigiReader"
    gemDigiToken = cms.InputTag("simMuonGEMDigis"),
    gemPadToken = cms.InputTag("simMuonGEMPadDigis")
)


process.p    = cms.Path(process.simMuonGEMPadDigis * process.dumper)
