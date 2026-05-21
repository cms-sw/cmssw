import FWCore.ParameterSet.Config as cms

caloJetAnalyzer = cms.EDAnalyzer(
    "HiCaloJetAnalyzer",
    jetTag = cms.InputTag("slimmedCaloJets"),
    jetName = cms.untracked.string("akPu4Calo"),
    jetPtMin = cms.double(5.0),
    genjetTag = cms.InputTag("ak4HiGenJets"),
    eventInfoTag = cms.InputTag("generator"),
    isMC = cms.untracked.bool(False), 
    fillGenJets = cms.untracked.bool(False),
    rParam = cms.double(0.4),
    pfCandidateLabel = cms.untracked.InputTag('packedPFCandidates'),
    trackTag = cms.InputTag("hiTracks"),
    trackQuality  = cms.untracked.string("highPurity"),
    towersSrc = cms.InputTag("towerMaker"),
    useHepMC = cms.untracked.bool(False),
    useQuality = cms.untracked.bool(True),
    doHiJetID = cms.untracked.bool(False),
    doCaloEnergyFractions = cms.untracked.bool(False)
    )
