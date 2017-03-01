import FWCore.ParameterSet.Config as cms

rivetHarvesting = cms.EDAnalyzer('RivetHarvesting',
  AnalysisNames = cms.vstring('CMS_2010_S8808686', 'MC_DIPHOTON', 'MC_JETS', 'MC_GENERIC'),
  CrossSection = cms.double(1000),
  HepMCCollection = cms.InputTag('generatorSmeared'),
  OutputFile = cms.string('out.yoda'),
  FilesToHarvest = cms.vstring('file1.yoda', 'file2.yoda'),
  VSumOfWeights = cms.vdouble(1.0, 1.0),
  VCrossSections = cms.vdouble(1.0, 1.0)
)
