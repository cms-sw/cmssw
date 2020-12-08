import FWCore.ParameterSet.Config as cms
process = cms.Process("HcalParametersTest")

process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.Geometry.GeometryExtended2018Reco_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')

if 'MessageLogger' in process.__dict__:
    process.MessageLogger.Collapse=dict()

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring('file:step311.root'),
                            secondaryFileNames = cms.untracked.vstring()
                            )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
    )

process.load('DQMOffline.Hcal.hcalCollapseAnalyzer_cfi')
process.load('Configuration/StandardSequences/EDMtoMEAtRunEnd_cff')

process.hcalCollapseAnalyzer.verbosity = 1

process.analysis_step = cms.Path(process.hcalCollapseAnalyzer)
process.dqmSaver_step = cms.Path(process.dqmSaver)

# Schedule definition
process.schedule = cms.Schedule(process.analysis_step, process.dqmSaver_step)
