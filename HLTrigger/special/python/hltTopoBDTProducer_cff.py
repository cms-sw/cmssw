import FWCore.ParameterSet.Config as cms

hltTopoMuonHtPNetBXGBProducer = cms.EDProducer('HLTTopoMuonHtPNetBXGBProducer',
    # input feature configuration 
    # hardcoded: PFHT, PNetBscore, muon pt, trkiso, ecaliso, hcaliso, 2. muon etc. if nMuons>1

    # XGBoost JSON path (must be reachable via edm::FileInPath)
    modelPath = cms.string("HLTrigger/HLTfilters/data/HLT_xgb_model_HH2b2W1L_1mu_HLTHT_sorttkisoMupt-absiso_PNetB.json"),

    # PFHT (from HLTHtMhtProducer)
    PFHT = cms.InputTag("hltPFHTJet30"),

    # ParticleNet B-tag
    PNetBscore = cms.InputTag("hltParticleNetDiscriminatorsJetTags","BvsAll"),

    # selection cuts for the muon used as input to the BDT
    muonPtCut  = cms.double(10.0),
    muonEtaCut = cms.double(2.4),

    nMuons = cms.uint32(1),
    muonSortByTkIso = cms.bool(False),  # if True, pick the muon with lowest track isolation; otherwise pick the leading pt muon

    # muon candidates
    ChargedCandidates = cms.InputTag("hltIterL3MuonCandidates"),
    # isolation maps keyed to ChargedCandidates refs
    EcalPFClusterIsoMap = cms.InputTag("hltMuonEcalMFPFClusterIsoForMuons"),
    HcalPFClusterIsoMap = cms.InputTag("hltMuonHcalRegPFClusterIsoForMuons"),
    TrackIsoMap = cms.InputTag("hltMuonTkRelIsolationCut0p3Map", "combinedRelativeIsoDeposits"),

    debug = cms.bool(False),
)
