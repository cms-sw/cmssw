import FWCore.ParameterSet.Config as cms

hltanalysis = cms.EDAnalyzer("HLTAnalyzer",
    ### GEN objects
    mctruth                         = cms.InputTag("genParticles"),
    genEventScale                   = cms.InputTag("genEventScale"),

    ### SIM objects
    simhits                         = cms.InputTag("g4SimHits"),

    ### Trigger objects
    l1GctHFBitCounts                 = cms.InputTag("hltGctDigis"),
    l1GctHFRingSums                  = cms.InputTag("hltGctDigis"),
    l1GtObjectMapRecord             = cms.InputTag("hltL1GtObjectMap::HLT"),
    l1GtReadoutRecord               = cms.InputTag("hltGtDigis::HLT"),

    l1extramc                       = cms.string('hltL1extraParticles'),
    l1extramu                       = cms.string('hltL1extraParticles'),
    hltresults                      = cms.InputTag("TriggerResults::HLT"),
    
    ### reconstructed objects
    genjets                         = cms.InputTag("iterativeCone5GenJets"),
    genmet                          = cms.InputTag("genMet"),
    recjets                         = cms.InputTag("hltIterativeCone5CaloJets"),
    reccorjets                      = cms.InputTag("hltMCJetCorJetIcone5"),
    recmet                          = cms.InputTag("hltMet"),
    ht                              = cms.InputTag("hltHtMet"),
    calotowers                      = cms.InputTag("towerMaker"),
    muon                            = cms.InputTag("muons"),
    Electron                        = cms.InputTag("pixelMatchGsfElectrons"),
    Photon                          = cms.InputTag("photons"),
    
    ### muon OpenHLT objects                             
    MuCandTag2                      = cms.InputTag("hltL2MuonCandidates"),
    MuCandTag3                      = cms.InputTag("hltL3MuonCandidates"),
    MuIsolTag3                      = cms.InputTag("hltL3MuonIsolations"),
    MuIsolTag2                      = cms.InputTag("hltL2MuonIsolations"),
    MuLinkTag                       = cms.InputTag("hltL3Muons"),
    ### egamma OpenHLT objects                             
    CandIso                         = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    CandNonIso                      = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    EcalIso                         = cms.InputTag("hltL1IsolatedPhotonEcalIsol"),
    EcalNonIso                      = cms.InputTag("hltL1NonIsolatedPhotonEcalIsol"),
    HcalIsoPho                      = cms.InputTag("hltL1IsolatedPhotonHcalIsol"),
    HcalNonIsoPho                   = cms.InputTag("hltL1NonIsolatedPhotonHcalIsol"),
    IsoPhoTrackIsol                 = cms.InputTag("hltL1IsoPhotonHollowTrackIsol"),
    NonIsoPhoTrackIsol              = cms.InputTag("hltL1NonIsoPhotonHollowTrackIsol"),
    HcalIsoEle                      = cms.InputTag("hltL1IsolatedElectronHcalIsol"),
    HcalNonIsoEle                   = cms.InputTag("hltL1NonIsolatedElectronHcalIsol"),
    ### egamma - standard or startup windows                         
    IsoElectrons                    = cms.InputTag("hltPixelMatchElectronsL1Iso"),
    NonIsoElectrons                 = cms.InputTag("hltPixelMatchElectronsL1NonIso"),
    PixelSeedL1Iso                  = cms.InputTag("hltL1IsoStartUpElectronPixelSeeds"),
    PixelSeedL1NonIso               = cms.InputTag("hltL1NonIsoStartUpElectronPixelSeeds"),
    IsoEleTrackIsol                 = cms.InputTag("hltL1IsoElectronTrackIsol"),
    NonIsoEleTrackIsol              = cms.InputTag("hltL1NonIsoElectronTrackIsol"),
    ### egamma - large windows
    IsoElectronsLargeWindows        = cms.InputTag("hltPixelMatchElectronsL1IsoLW"),
    NonIsoElectronsLargeWindows     = cms.InputTag("hltPixelMatchElectronsL1NonIsoLW"),
    PixelSeedL1IsoLargeWindows      = cms.InputTag("hltL1IsoLargeWindowElectronPixelSeeds"),
    PixelSeedL1NonIsoLargeWindows   = cms.InputTag("hltL1NonIsoLargeWindowElectronPixelSeeds"),
    IsoEleTrackIsolLargeWindows     = cms.InputTag("hltL1IsoLWEleTrackIsol"),
    NonIsoEleTrackIsolLargeWindows  = cms.InputTag("hltL1NonIsoLWEleTrackIsol"),

    ### tau OpenHLT related objects
    HLTTau                          = cms.InputTag("TauOpenHLT"),
    
    ### b-jet OpenHLT related objects
    CommonBJetsL2                   = cms.InputTag("hltIterativeCone5CaloJets"),
    CorrectedBJetsL2                = cms.InputTag("hltMCJetCorJetIcone5"),
    LifetimeBJetsL25                = cms.InputTag("openHltBLifetimeL25BJetTags"),
    LifetimeBJetsL3                 = cms.InputTag("openHltBLifetimeL3BJetTags"),
    LifetimeBJetsL25Relaxed         = cms.InputTag("openHltBLifetimeL25BJetTags"),
    LifetimeBJetsL3Relaxed          = cms.InputTag("openHltBLifetimeL3BJetTagsStartup"),
    SoftmuonBJetsL25                = cms.InputTag("openHltBSoftmuonL25BJetTags"),
    SoftmuonBJetsL3                 = cms.InputTag("openHltBSoftmuonL3BJetTags"),
    PerformanceBJetsL25             = cms.InputTag("openHltBSoftmuonL25BJetTags"),
    PerformanceBJetsL3              = cms.InputTag("openHltBPerfMeasL3BJetTags"),

    ### AlCa OpenHLT related objects
    EERecHits                   = cms.InputTag("hltEcalRecHit","EcalRecHitsEE"),
    EBRecHits                   = cms.InputTag("hltEcalRecHit","EcalRecHitsEB"),
    pi0EBRecHits                = cms.InputTag("hltEcalRegionalPi0EtaRecHit","EcalRecHitsEB"),
    pi0EERecHits                = cms.InputTag("hltEcalRegionalPi0EtaRecHit","EcalRecHitsEE"),
    HBHERecHits                 = cms.InputTag("hltHbhereco"),
    HORecHits                   = cms.InputTag("hltHoreco"),
    HFRecHits                   = cms.InputTag("hltHfreco"),
    IsoPixelTracksL3            = cms.InputTag("hltHITIPTCorrector1E31"),                         

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
