import FWCore.ParameterSet.Config as cms

rivetAnalyzer = cms.EDAnalyzer('RivetAnalyzer',
  AnalysisNames = cms.vstring('CMS_2010_S8808686', 'MC_DIPHOTON', 'MC_JETS', 'MC_GENERIC'),
  HepMCCollection = cms.InputTag('generatorSmeared'),
  UseExternalWeight = cms.bool(False),
  GenEventInfoCollection = cms.InputTag('generator'),
  useLHEweights = cms.bool(False),
  LHEweightNumber = cms.int32(0),
  LHECollection = cms.InputTag('source'),
  CrossSection = cms.double(1000),
  DoFinalize = cms.bool(True),
  ProduceDQMOutput = cms.bool(False),
  OutputFile = cms.string('out.aida')
)
