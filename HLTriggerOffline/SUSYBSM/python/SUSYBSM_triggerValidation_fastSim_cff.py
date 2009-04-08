import FWCore.ParameterSet.Config as cms

HLTSusyExoValFastSim = cms.EDAnalyzer("TriggerValidator",
    TurnOnParams = cms.PSet(
        hlt1MuonIsoList = cms.vstring('hltSingleMuIsoLevel1Seed', 
            'hltSingleMuIsoL1Filtered', 
            'hltSingleMuIsoL2PreFiltered', 
            'hltSingleMuIsoL2IsoFiltered', 
            'hltSingleMuIsoL3PreFiltered', 
            'hltSingleMuIsoL3IsoFiltered'),
        hltMuonTracks = cms.string('hltL3MuonCandidates'),
        hlt1MuonNonIsoList = cms.vstring('hltSingleMuNoIsoLevel1Seed', 
            'hltSingleMuNoIsoL1Filtered', 
            'hltSingleMuNoIsoL2PreFiltered', 
            'hltSingleMuNoIsoL3PreFiltered'),
        genMother = cms.string('b'), ## it can be W, b, WtoJ, All

        recoMuons = cms.string('muons'),
        mcParticles = cms.string('genParticles')
    ),
    statFileName = cms.untracked.string('MonElements_LM1_IDEAL_30x_v1_300pre7.stat'),
    dirname = cms.untracked.string('HLT/SusyExo'),
    L1Label = cms.InputTag("gtDigis"),
    HltLabel = cms.InputTag("TriggerResults","","HLT"),
    # if mc_flag = false the McSelection folder will contain empty histograms
    UserCutParams = cms.PSet(
        reco_ptJet2Min = cms.double(30.0),
        jets = cms.string('iterativeCone5CaloJets'),
        genMet = cms.string('genMetTrue'),
        genJets = cms.string('iterativeCone5GenJets'),
        mc_ptPhotMin = cms.double(0.0),
        reco_ptElecMin = cms.double(10.0),
        reco_ptJet1Min = cms.double(80.0),
        photonProducer = cms.string('photons'),
        reco_metMin = cms.double(100.0),
        mc_nPhot = cms.int32(0),
        mc_nElec = cms.int32(1),
        photons = cms.string(''),
        muons = cms.string('muons'),
        mc_nMuon = cms.int32(1),
        mc_ptElecMin = cms.double(10.0),
        reco_ptMuonMin = cms.double(10.0),
        mc_nJet = cms.int32(1),
        reco_ptPhotMin = cms.double(0.0),
        mcparticles = cms.string('genParticles'),
        calomet = cms.string('met'),
        mc_ptJetMin = cms.double(40.0),
        mc_ptMuonMin = cms.double(10.0),
        mc_metMin = cms.double(50.0),
        electrons = cms.string('gsfElectrons')
    ),
    mc_flag = cms.untracked.bool(True), ## put mc_flag = false if you don't want to use the mc information.

    histoFileName = cms.untracked.string('MonElements_LM1_IDEAL_30x_v1_300pre7.root'),
    ObjectList = cms.PSet(
        def_electronPtMin = cms.double(10.0),
        def_muonPtMin = cms.double(7.0),
        def_photonPtMin = cms.double(30.0),
        l1extramc = cms.string('hltL1extraParticles'),
        calomet = cms.string('met'),
        electrons = cms.string('gsfElectrons'),
        jets = cms.string('iterativeCone5CaloJets'),
        muons = cms.string('muons'),
        def_jetPtMin = cms.double(30.0),
        photons = cms.string(''),
        photonProducer = cms.string('photons')
    )
)
