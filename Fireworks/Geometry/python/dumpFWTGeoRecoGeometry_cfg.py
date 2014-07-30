import FWCore.ParameterSet.Config as cms

process = cms.Process("DUMP")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['mc']

process.load('Configuration.Geometry.GeometryExtended2015_cff')
process.load('Configuration.Geometry.GeometryExtended2015Reco_cff')

process.add_(cms.Service("InitRootHandlers", ResetRootErrHandler = cms.untracked.bool(False)))

process.add_(cms.ESProducer("FWTGeoRecoGeometryESProducer"))

#Adding Timing service:
process.Timing = cms.Service("Timing")
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(False)
    )

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )
process.dump = cms.EDAnalyzer("DumpFWTGeoRecoGeometry",
                              level = cms.untracked.int32(1)
                              )

process.p = cms.Path(process.dump)
