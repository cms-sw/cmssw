import FWCore.ParameterSet.Config as cms

rivetAnalyzer = cms.EDAnalyzer('RivetAnalyzer',
  AnalysisNames = cms.vstring('CMS_2010_2PCORRELATION', 'MC_DIPHOTON', 'MC_JETS', 'MC_GENERIC'),
  HepMCCollection = cms.InputTag('generator'),
  CrossSection = cms.double(1000)
)
