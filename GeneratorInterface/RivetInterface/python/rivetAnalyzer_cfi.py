import FWCore.ParameterSet.Config as cms

rivetAnalyzer = cms.EDAnalyzer('RivetAnalyzer',
  AnalysisNames = cms.vstring('CMS_2010_S8808686', 'MC_DIPHOTON', 'MC_JETS', 'MC_GENERIC'),
  HepMCCollection = cms.InputTag('generator:unsmeared'),
  GenEventInfoCollection = cms.InputTag('generator'),
  genLumiInfo = cms.InputTag("generator"),
  # Info: useLHEWeights will create BIG yoda files with Rivet 3.
  # Default plotting fails with too many histos, use -m/M options
  useLHEweights = cms.bool(False),
  weightCap = cms.double(0.),
  NLOSmearing = cms.double(0.),
  skipMultiWeights = cms.bool(False),
  setIgnoreBeams = cms.bool(False),
  selectMultiWeights = cms.string(''),
  deselectMultiWeights = cms.string(''),
  setNominalWeightName = cms.string(''),
  LHECollection = cms.InputTag('externalLHEProducer'),
  CrossSection = cms.double(-1),
  DoFinalize = cms.bool(True),
  OutputFile = cms.string('out.yoda')
)
