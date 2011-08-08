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
    genjets                         = cms.InputTag("iterativeCone5GenJets"),
    genmet                          = cms.InputTag("genMet"),
    recjets                         = cms.InputTag("hltAntiKT5CaloJets"),
    reccorjets                      = cms.InputTag("hltAntiKT5L2L3CorrCaloJets"),
    recmet                          = cms.InputTag("hltMet"),
    ht                              = cms.InputTag("hltHtMet"),
    recoPFJets                      = cms.InputTag("ak5PFJets"),
    calotowers                      = cms.InputTag("hltTowerMakerForAll"),
    muon                            = cms.InputTag("muons"),
    Electron                        = cms.InputTag("gsfElectrons"),
    Photon                          = cms.InputTag("photons"),
    
    ### muon OpenHLT objects                             
    MuCandTag2                      = cms.InputTag("hltL2MuonCandidates"),
    MuCandTag3                      = cms.InputTag("hltL3MuonCandidates"),
    MuIsolTag3                      = cms.InputTag("hltL3MuonIsolations"),
    MuIsolTag2                      = cms.InputTag("hltL2MuonIsolations"),
    MuNoVtxCandTag2                 = cms.InputTag("hltL2MuonCandidatesNoVtx"),
    OniaPixelTag                    = cms.InputTag("hltMuTrackJpsiPixelTrackCands"),
    OniaTrackTag                    = cms.InputTag("hltMuTrackJpsiCtfTrackCands"),

    ### egamma OpenHLT objects                             
    CandIso                         = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    CandNonIso                      = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    EcalIso                         = cms.InputTag("hltL1IsolatedPhotonEcalIsol"),
    EcalNonIso                      = cms.InputTag("hltL1NonIsolatedPhotonEcalIsol"),
    HcalIsoPho                      = cms.InputTag("hltL1IsolatedPhotonHcalIsol"),
    HcalNonIsoPho                   = cms.InputTag("hltL1NonIsolatedPhotonHcalIsol"),
    IsoPhoTrackIsol                 = cms.InputTag("hltL1IsoPhotonHollowTrackIsol"),
    NonIsoPhoTrackIsol              = cms.InputTag("hltL1NonIsoPhotonHollowTrackIsol"),
    HcalIsoEle                      = cms.InputTag("hltL1IsolatedPhotonHcalIsol"),
    HcalNonIsoEle                   = cms.InputTag("hltL1NonIsolatedPhotonHcalIsol"),
    SpikeCleaningIsol               = cms.InputTag("hltL1IsoR9shape"),
    SpikeCleaningNonIsol            = cms.InputTag("hltL1NonIsoR9shape"),            
    HcalForHoverEIsol               = cms.InputTag("hltL1IsolatedPhotonHcalForHE"),
    HcalForHoverENonIsol            = cms.InputTag("hltL1NonIsolatedPhotonHcalForHE"),
    R9IDIsol                        = cms.InputTag("hltL1IsoR9ID"),
    R9IDNonIsol                     = cms.InputTag("hltL1NonIsoR9ID"),
    HFElectrons                     = cms.InputTag("hltHFRecoEcalCandidate"),
    HFECALClusters                  = cms.InputTag("hltHFEMClusters"),
                             
    ### egamma - standard or startup windows                         
    IsoElectrons                    = cms.InputTag("hltPixelMatchElectronsL1Iso"),
    NonIsoElectrons                 = cms.InputTag("hltPixelMatchElectronsL1NonIso"),
    PixelSeedL1Iso                  = cms.InputTag("hltL1IsoStartUpElectronPixelSeeds"),
    PixelSeedL1NonIso               = cms.InputTag("hltL1NonIsoStartUpElectronPixelSeeds"),
    IsoEleTrackIsol                 = cms.InputTag("hltL1IsoElectronTrackIsol"),
    NonIsoEleTrackIsol              = cms.InputTag("hltL1NonIsoElectronTrackIsol"),
                             
    ### tau OpenHLT related objects
    HLTTau                          = cms.InputTag("TauOpenHLT"),
    HLTPFTau                        = cms.InputTag("hltPFTaus"),
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
    CorrectedBJetsL2                = cms.InputTag("hltAntiKT5L2L3CorrCaloJets"),
    LifetimeBJetsL25                = cms.InputTag("openHltBLifetimeL25BJetTags"),
    LifetimeBJetsL3                 = cms.InputTag("openHltBLifetimeL3BJetTags"),
    SoftmuonBJetsL25                = cms.InputTag("openHltBSoftmuonL25BJetTags"),
    SoftmuonBJetsL3                 = cms.InputTag("openHltBSoftmuonL3BJetTags"),
    PerformanceBJetsL25             = cms.InputTag("openHltBSoftmuonL25BJetTags"),
    PerformanceBJetsL3              = cms.InputTag("openHltBPerfMeasL3BJetTags"),

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

    ### Calo tower settings
    caloTowerThreshold          = cms.double( 2.0 ),

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
    PrimaryVertices             = cms.InputTag("hltPixelVertices"),
                             
    ### Run parameters
    RunParameters = cms.PSet(
        HistogramFile = cms.untracked.string('openhlt.root'),
        EtaMin        = cms.untracked.double(-5.2),
        EtaMax        = cms.untracked.double( 5.2),
        CalJetMin     = cms.double(0.0),
        GenJetMin     = cms.double(0.0),
        Monte         = cms.bool(True),
        Debug         = cms.bool(False)
    )
)
