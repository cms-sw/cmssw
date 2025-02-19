import FWCore.ParameterSet.Config as cms

electronStudy = cms.EDAnalyzer("ElectronStudy",
                               SourceLabel = cms.untracked.string('generator'),
                               ModuleLabel = cms.untracked.string('g4SimHits'),
                               EBCollection= cms.untracked.string('EcalHitsEB'),
                               EECollection= cms.untracked.string('EcalHitsEE'),
                               Verbosity   = cms.untracked.int32(0)
)
