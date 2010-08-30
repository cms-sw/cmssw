
import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.load("RecoHI.HiCentralityAlgos.HiTrivialCondRetriever_cfi")

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring("rfio:/castor/cern.ch/user/y/yilmaz/share/Hydjet_MinBias_Jets_runs1to20.root")
                            )

process.analyze = cms.EDAnalyzer("AnalyzerWithCentrality")

process.TFileService = cms.Service('TFileService',
                                   fileName = cms.string("histograms.root")
                                   )

process.p = cms.Path(process.analyze)

