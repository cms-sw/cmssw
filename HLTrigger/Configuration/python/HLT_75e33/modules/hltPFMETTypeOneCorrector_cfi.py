import FWCore.ParameterSet.Config as cms

hltPFMETTypeOneCorrector = cms.EDProducer("PFJetMETcorrInputProducer",
    jetCorrEtaMax = cms.double(9.9),
    jetCorrLabel = cms.InputTag("hltPFMETJetCorrector"),
    jetCorrLabelRes = cms.InputTag("hltPFMETJetCorrector"),
    offsetCorrLabel = cms.InputTag("hltPFMETJetCorrectorL1"),
    skipEM = cms.bool(True),
    skipEMfractionThreshold = cms.double(0.9),
    skipMuonSelection = cms.string('isGlobalMuon | isStandAloneMuon'),
    skipMuons = cms.bool(True),
    src = cms.InputTag("hltAK4PFCHSJets"),
    type1JetPtThreshold = cms.double(30.0)
)
