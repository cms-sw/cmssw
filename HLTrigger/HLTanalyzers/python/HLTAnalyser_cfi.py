import FWCore.ParameterSet.Config as cms

hltanalysis = cms.EDAnalyzer("HLTAnalyzer",
    ### GEN objects
    mctruth                         = cms.InputTag("genParticles::HLT"),
    # genEventScale                   = cms.InputTag("genEventScale"),
    genEventInfo                    = cms.InputTag("generator::HLT"),

    ### SIM objects
    simhits                         = cms.InputTag("g4SimHits"),

    ## Input x-section weight and filter efficiency
    xSection                        = cms.untracked.double(1.),
    filterEff                       = cms.untracked.double(1.),

    ## Cut on lumi section
    firstLumi                        = cms.untracked.int32(0),
    lastLumi                         = cms.untracked.int32(-1),

    ### Trigger objects
    l1GctHFBitCounts                = cms.InputTag("hltGctDigis"),
    l1GctHFRingSums                 = cms.InputTag("hltGctDigis"),
    l1GtObjectMapRecord             = cms.InputTag("hltL1GtObjectMap::HLT"),
    l1GtReadoutRecord               = cms.InputTag("hltGtDigis::HLT"),

    l1extramc                       = cms.string('hltL1extraParticles'),
    l1extramu                       = cms.string('hltL1extraParticles'),
    hltresults                      = cms.InputTag("TriggerResults::HLT"),
    HLTProcessName                  = cms.string("HLT"),

    ### reconstructed objects
    genjets                         = cms.InputTag("ak4GenJets"),
    genmet                          = cms.InputTag("genMet"),
    hltjets                         = cms.InputTag("hltAntiKT5CaloJets"),
    hltcorjets                      = cms.InputTag("hltCaloJetCorrected"),
    hltcorL1L2L3jets                = cms.InputTag("hltCaloJetL1FastJetCorrected"),
    rho                             = cms.InputTag("hltKT6CaloJets"),
    recjets                         = cms.InputTag("ak4CaloJets"),
    reccorjets                      = cms.InputTag("ak4CaloCorJets"),
    recmet                          = cms.InputTag("hltMet"),
    pfmet                           = cms.InputTag("pfMet"),
    ht                              = cms.InputTag("hltJet40Ht"),
    recoPFJets                      = cms.InputTag("ak4PFJets"),
    calotowers                      = cms.InputTag("hltTowerMakerForAll"),
    calotowersUpperR45              = cms.InputTag("hltHcalTowerNoiseCleanerUpper"),
    calotowersLowerR45              = cms.InputTag("hltHcalTowerNoiseCleanerLower"),
    calotowersNoR45                 = cms.InputTag("hltHcalTowerNoiseCleanerNoR45"),
    muon                            = cms.InputTag("muons"),
    pfmuon                          = cms.InputTag("pfAllMuons"),
    Electron                        = cms.InputTag("gsfElectrons"),
    Photon                          = cms.InputTag("photons"),

    ### muon OpenHLT objects
    MuCandTag2                      = cms.InputTag("hltL2MuonCandidates"),
    MuCandTag3                      = cms.InputTag("hltL3MuonCandidates"),
    MuIsolTag3                      = cms.InputTag("hltL3MuonCombRelIsolations"),
    MuIsolTag2                      = cms.InputTag("hltL2MuonIsolations"),
    MuTrkIsolTag3                   = cms.InputTag("hltL3MuonTkIsolations10"),
    MuNoVtxCandTag2                 = cms.InputTag("hltL2MuonCandidatesNoVtx"),
    OniaPixelTag                    = cms.InputTag("hltMuTrackJpsiPixelTrackCands"),
    OniaTrackTag                    = cms.InputTag("hltMuTrackJpsiCtfTrackCands"),
    TrackerMuonTag                  = cms.InputTag("hltGlbTrkMuons"),
    DiMuVtx                         = cms.InputTag("hltDisplacedmumuVtxProducerLowMass"),

    ### egamma OpenHLT objects
    CandIso                         = cms.InputTag("hltL1SeededRecoEcalCandidate"),
    CandNonIso                      = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    EcalIso                         = cms.InputTag("hltL1SeededPhotonEcalIsol"),
    EcalNonIso                      = cms.InputTag("hltL1NonIsolatedPhotonEcalIsol"),
    HcalIsoPho                      = cms.InputTag("hltL1SeededPhotonHcalIsol"),
    HcalNonIsoPho                   = cms.InputTag("hltL1NonIsolatedPhotonHcalIsol"),
    IsoPhoTrackIsol                 = cms.InputTag("hltL1SeededPhotonHollowTrackIsol"),
    NonIsoPhoTrackIsol              = cms.InputTag("hltL1NonIsolatedPhotonHollowTrackIsol"),
    HcalIsoEle                      = cms.InputTag("hltL1SeededPhotonHcalIsol"),
    HcalNonIsoEle                   = cms.InputTag("hltL1NonIsolatedPhotonHcalIsol"),
    SpikeCleaningIsol               = cms.InputTag("hltL1SeededR9shape"),
    SpikeCleaningNonIsol            = cms.InputTag("hltL1NonIsoR9shape"),
    HcalForHoverEIsol               = cms.InputTag("hltL1SeededPhotonHcalForHE"),
    HcalForHoverENonIsol            = cms.InputTag("hltL1NonIsolatedPhotonHcalForHE"),
    R9IDIsol                        = cms.InputTag("hltL1SeededR9ID"),
    R9IDNonIsol                     = cms.InputTag("hltL1NonIsoR9ID"),
    HFElectrons                     = cms.InputTag("hltHFRecoEcalTightCandidate"),
    HFECALClusters                  = cms.InputTag("hltHFEMClusters"),
    ECALActivity                    = cms.InputTag("hltRecoEcalSuperClusterActivityCandidate"),
    ActivityEcalIso                 = cms.InputTag("hltActivityPhotonEcalIsol"),
    ActivityHcalIso                 = cms.InputTag("hltActivityPhotonHcalIsol"),
    ActivityTrackIso                = cms.InputTag("hltActivityPhotonHollowTrackIsolWithId"),
    ActivityR9                      = cms.InputTag("hltUnseededR9shape"), # spike cleaning
    ActivityR9ID                    = cms.InputTag("hltActivityR9ID"),
    ActivityHcalForHoverE           = cms.InputTag("hltActivityPhotonHcalForHE"),

    ### egamma - standard or startup windows
    IsoElectrons                    = cms.InputTag("hltPixelMatchElectronsL1Seeded"),
    NonIsoElectrons                 = cms.InputTag("hltPixelMatchElectronsL1NonIso"),
    PixelSeedL1Iso                  = cms.InputTag("hltL1SeededStartUpElectronPixelSeeds"),
    PixelSeedL1NonIso               = cms.InputTag("hltL1NonIsoStartUpElectronPixelSeeds"),
    IsoEleTrackIsol                 = cms.InputTag("hltL1SeededElectronTrackIsol"),
    NonIsoEleTrackIsol              = cms.InputTag("hltL1NonIsoElectronTrackIsol"),

    ### tau OpenHLT related objects
    L2Tau                           = cms.InputTag("openhltL2TauJets"),
    HLTTau                          = cms.InputTag("TauOpenHLT"),
    HLTPFTau                        = cms.InputTag("hltPFTaus"),
    HLTPFTauTightCone               = cms.InputTag("hltPFTausTightIso"),
    minPtChargedHadronsForTaus      = cms.double(1.5),
    minPtGammassForTaus             = cms.double(1.5),

    ### particle flow jets OpenHLT related objects
    HLTPFJet                        = cms.InputTag("hltAntiKT5PFJets"),

    ### reco offline particle flow tau related objects
    RecoPFTau = cms.InputTag("shrinkingConePFTauProducer"),
    RecoPFTauAgainstMuon = cms.InputTag("shrinkingConePFTauDiscriminationAgainstMuon"),
    RecoPFTauAgainstElec = cms.InputTag("shrinkingConePFTauDiscriminationAgainstElectron"),
    RecoPFTauDiscrByIso = cms.InputTag("shrinkingConePFTauDiscriminationByIsolation"),
    RecoPFTauDiscrByTanCOnePercent = cms.InputTag("shrinkingConePFTauDiscriminationByTaNCfrOnePercent"),
    RecoPFTauDiscrByTanCHalfPercent = cms.InputTag("shrinkingConePFTauDiscriminationByTaNCfrHalfPercent"),
    RecoPFTauDiscrByTanCQuarterPercent = cms.InputTag("shrinkingConePFTauDiscriminationByTaNCfrQuarterPercent"),
    RecoPFTauDiscrByTanCTenthPercent = cms.InputTag("shrinkingConePFTauDiscriminationByTaNCfrTenthPercent"),

    ### b-jet OpenHLT related objects
    CommonBJetsL2                   = cms.InputTag("hltAntiKT5CaloJets"),
    CorrectedBJetsL2                = cms.InputTag("hltCaloJetCorrected"),
    CorrectedBJetsL2L1FastJet       = cms.InputTag("hltCaloJetL1FastJetCorrected"),
    PFlowBJetsL2                    = cms.InputTag("hltAntiKT5PFJets"),
    LifetimeBJetsL25                = cms.InputTag("openHltBLifetimeL25BJetTags"),
    LifetimeBJetsL3                 = cms.InputTag("openHltBLifetimeL3BJetTags"),
    LifetimeBJetsL25L1FastJet       = cms.InputTag("openHltBLifetimeL25BJetTagsL1FastJet"),
    LifetimeBJetsL3L1FastJet        = cms.InputTag("openHltBLifetimeL3BJetTagsL1FastJet"),
    LifetimePFlowBJetsL3            = cms.InputTag("openHltBLifetimeL3BJetTagsPF"),
    LifetimeBJetsL25SingleTrack     = cms.InputTag("openHltBLifetimeL25BJetTagsSingleTrack"),
    LifetimeBJetsL3SingleTrack      = cms.InputTag("openHltBLifetimeL3BJetTagsSingleTrack"),
    LifetimeBJetsL25SingleTrackL1FastJet     = cms.InputTag("openHltBLifetimeL25BJetTagsSingleTrackL1FastJet"),
    LifetimeBJetsL3SingleTrackL1FastJet      = cms.InputTag("openHltBLifetimeL3BJetTagsSingleTrackL1FastJet"),
    PerformanceBJetsL25             = cms.InputTag("openHltBSoftmuonL25BJetTags"),
    PerformanceBJetsL3              = cms.InputTag("openHltBPerfMeasL3BJetTags"),
    PerformanceBJetsL25L1FastJet    = cms.InputTag("openHltBSoftmuonL25BJetTagsL1FastJet"),
    PerformanceBJetsL3L1FastJet     = cms.InputTag("openHltBPerfMeasL3BJetTagsL1FastJet"),

    ### AlCa OpenHLT related objects
    EERecHits                   = cms.InputTag("hltEcalRecHitAll","EcalRecHitsEE"),
    EBRecHits                   = cms.InputTag("hltEcalRecHitAll","EcalRecHitsEB"),
    pi0EBRecHits                = cms.InputTag("hltEcalRegionalPi0EtaRecHit","EcalRecHitsEB"),
    pi0EERecHits                = cms.InputTag("hltEcalRegionalPi0EtaRecHit","EcalRecHitsEE"),
    HBHERecHits                 = cms.InputTag("hltHbhereco"),
    HORecHits                   = cms.InputTag("hltHoreco"),
    HFRecHits                   = cms.InputTag("hltHfreco"),
    IsoPixelTracksL3            = cms.InputTag("hltHITIPTCorrector1E31"),
    IsoPixelTracksL2            = cms.InputTag("hltIsolPixelTrackProd1E31"),
    IsoPixelTrackVertices       = cms.InputTag("hltPixelVertices"),

    ### Track settings
    PixelTracksL3               = cms.InputTag("hltPixelCandsForMinBias"),
    PixelFEDSize                = cms.InputTag("rawDataCollector"),
    PixelClusters               = cms.InputTag("hltSiPixelClusters"),

    ### Calo tower settings
    caloTowerThreshold          = cms.double( 5.0 ),

    ### AlCa pi0 settings
    clusSeedThr                 = cms.double( 0.5 ),
    clusSeedThrEndCap           = cms.double( 1.0 ),
    clusEtaSize                 = cms.int32( 3 ),
    clusPhiSize                 = cms.int32( 3 ),
    seleXtalMinEnergy           = cms.double( 0.0 ),
    ParameterLogWeighted        = cms.bool( True ),
    ParameterX0                 = cms.double( 0.89 ),
    ParameterT0_barl            = cms.double( 7.4 ),
    ParameterT0_endc            = cms.double( 3.1 ),
    ParameterT0_endcPresh       = cms.double( 1.2 ),
    ParameterW0                 = cms.double( 4.2 ),
    RegionalMatch               = cms.untracked.bool( True ),
    ptMinEMObj                  = cms.double( 2.0 ),
    EMregionEtaMargin           = cms.double( 0.25 ),
    EMregionPhiMargin           = cms.double( 0.4 ),
    Jets                        = cms.untracked.bool( False ),

    ## reco vertices
    OfflinePrimaryVertices0     = cms.InputTag('offlinePrimaryVertices'),

    # HLT vertices
    PrimaryVertices             = cms.InputTag("hltPixelVertices"),
    PrimaryVerticesHLT          = cms.InputTag('pixelVertices'),

    ### Run parameters
    RunParameters = cms.PSet(
        HistogramFile = cms.untracked.string('openhlt.root'),
        EtaMin        = cms.untracked.double(-5.2),
        EtaMax        = cms.untracked.double( 5.2),
        CalJetMin     = cms.double(0.0),
        GenJetMin     = cms.double(0.0),
        Monte         = cms.bool(True),
        Debug         = cms.bool(False)
    ),

    JetIDParams  = cms.PSet(
         useRecHits      = cms.bool(True),
         hbheRecHitsColl = cms.InputTag("hltHbhereco"),
         hoRecHitsColl   = cms.InputTag("hltHoreco"),
         hfRecHitsColl   = cms.InputTag("hltHfreco"),
         #ebRecHitsColl   = cms.InputTag("EcalRecHitsEB"),
         #eeRecHitsColl   = cms.InputTag("EcalRecHitsEE")
         ebRecHitsColl   = cms.InputTag("hltEcalRecHitAll", "EcalRecHitsEB"),
         eeRecHitsColl   = cms.InputTag("hltEcalRecHitAll", "EcalRecHitsEE")
     )
)
