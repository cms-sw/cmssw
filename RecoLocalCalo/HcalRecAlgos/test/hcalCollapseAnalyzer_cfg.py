import FWCore.ParameterSet.Config as cms
process = cms.Process("HcalParametersTest")

process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.Geometry.GeometryExtended2018Reco_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')

if 'MessageLogger' in process.__dict__:
    process.MessageLogger.categories.append('Collapse')

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring('file:step311.root'),
                            secondaryFileNames = cms.untracked.vstring()
                            )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
    )

process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string("Hist_step311.root")
                                   )

process.load('RecoLocalCalo.HcalRecAlgos.hcalCollapseAnalyzer_cfi')
process.hcalCollapseAnalyzer.verbosity = 1

process.p1 = cms.Path(process.hcalCollapseAnalyzer)
