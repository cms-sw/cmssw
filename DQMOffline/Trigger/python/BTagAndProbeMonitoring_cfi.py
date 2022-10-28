import FWCore.ParameterSet.Config as cms

BTagAndProbeMonitoring = cms.EDProducer('BTagAndProbe',
  FolderName = cms.string('HLT/BTV'),
  requireValidHLTPaths = cms.bool(True),
  vertices = cms.InputTag('offlinePrimaryVertices'),
  muons = cms.InputTag('muons'),
  electrons = cms.InputTag('gedGsfElectrons'),
  elecID = cms.InputTag('egmGsfElectronIDsForDQM', 'cutBasedElectronID-Fall17-94X-V1-tight'),
  #jets = cms.InputTag('ak4PFJetsCHS'),
  btagAlgos = cms.VInputTag(
    'pfDeepCSVJetTags:probb',
    'pfDeepCSVJetTags:probbb'
  ),
  jetSelection = cms.string('pt > 0'),
  eleSelection = cms.string('pt > 0 && abs(eta) < 2.5'),
  muoSelection = cms.string('pt > 6 && abs(eta) < 2.4'),
  vertexSelection = cms.string('!isFake'),
  bjetSelection = cms.string('pt > 0'),
  nelectrons = cms.uint32(0),
  nmuons = cms.uint32(0),
  leptJetDeltaRmin = cms.double(0),
  bJetDeltaEtaMax = cms.double(9999),
  nbjets = cms.uint32(0),
  workingpoint = cms.double(0.4941),
  applyLeptonPVcuts = cms.bool(False),
  debug = cms.bool(False),
  #numGenericTriggerEventPSet = cms.PSet(
  #  andOr = cms.required.bool,
  #  dcsInputTag = cms.InputTag('scalersRawToDigi'),
  #  dcsPartitions = cms.vint32(),
  #  andOrDcs = cms.bool(False),
  #  errorReplyDcs = cms.bool(True),
  #  dbLabel = cms.string(''),
  #  andOrHlt = cms.bool(True),
  #  dcsRecordInputTag = cms.InputTag('onlineMetaDataDigis'),
  #  hltInputTag = cms.InputTag('TriggerResults', '', 'HLT'),
  #  hltPaths = cms.vstring(),
  #  hltDBKey = cms.string(''),
  #  errorReplyHlt = cms.bool(False),
  #  verbosityLevel = cms.uint32(1)
  #),
  genericTriggerEventPSet = cms.PSet(
    andOr = cms.required.bool,
    dcsInputTag = cms.InputTag('scalersRawToDigi'),
    dcsPartitions = cms.vint32(),
    andOrDcs = cms.bool(False),
    errorReplyDcs = cms.bool(True),
    dbLabel = cms.string(''),
    andOrHlt = cms.bool(True),
    dcsRecordInputTag = cms.InputTag('onlineMetaDataDigis'),
    hltInputTag = cms.InputTag('TriggerResults', '', 'HLT'),
    hltPaths = cms.vstring(),
    hltDBKey = cms.string(''),
    errorReplyHlt = cms.bool(False),
    verbosityLevel = cms.uint32(1)
  ),
  
  leptonPVcuts = cms.PSet(
    dxy = cms.double(9999),
    dz = cms.double(9999)
  ),
  mightGet = cms.optional.untracked.vstring
)
