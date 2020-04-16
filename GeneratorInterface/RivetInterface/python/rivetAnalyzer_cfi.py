import FWCore.ParameterSet.Config as cms

rivetAnalyzer = cms.EDAnalyzer('RivetAnalyzer',
  AnalysisNames = cms.vstring('CMS_2010_S8808686', 'MC_DIPHOTON', 'MC_JETS', 'MC_GENERIC'),
  HepMCCollection = cms.InputTag('generator:unsmeared'),
  GenEventInfoCollection = cms.InputTag('generator'),
  genLumiInfo = cms.InputTag("generator"),
  # Info: useLHEWeights will create BIG yoda files with Rivet 3.
  # Default plotting fails with too many histos, use -m/M options
  useLHEweights = cms.bool(False),
  LHECollection = cms.InputTag('externalLHEProducer'),
  CrossSection = cms.double(-1),
  DoFinalize = cms.bool(True),
  ProduceDQMOutput = cms.bool(False),
  OutputFile = cms.string('out.yoda')
)
