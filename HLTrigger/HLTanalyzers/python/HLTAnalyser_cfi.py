import FWCore.ParameterSet.Config as cms

hltanalysis = cms.EDAnalyzer("HLTAnalyzer",
    ### GEN objects

    mctruth                         = cms.InputTag("genParticles"),
    genEventInfo                    = cms.InputTag("generator"),


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
    genjets                         = cms.InputTag("iterativeCone5HiGenJets"),
    genmet                          = cms.InputTag("genMet"),
    recjets                         = cms.InputTag("hltIterativeCone5CaloJets"),
    reccorjets                      = cms.InputTag("hltMCJetCorJetIcone5"),
    recmet                          = cms.InputTag("hltMet"),
    ht                              = cms.InputTag("hltHtMet"),
    calotowers                      = cms.InputTag("hltTowerMakerForAll"),
    muon                            = cms.InputTag("muons"),
    Photon                          = cms.InputTag("photons"),                          
    Electron                        = cms.InputTag("pixelMatchGsfElectrons"),
    BarrelPhoton                    = cms.InputTag("hltIslandSuperClustersHI:islandBarrelSuperClustersHI"),
    EndcapPhoton                    = cms.InputTag("hltIslandSuperClustersHI:islandEndcapSuperClustersHI"),
    
    ### muon OpenHLT objects                             
    MuCandTag2                      = cms.InputTag("hltL2MuonCandidates"),
    MuCandTag3                      = cms.InputTag("hltL3MuonCandidates"),
    MuIsolTag3                      = cms.InputTag("hltL3MuonIsolations"),
    MuIsolTag2                      = cms.InputTag("hltL2MuonIsolations"),
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
    ### egamma - SiStrip
    IsoElectronsSiStrip             = cms.InputTag("hltPixelMatchElectronsL1IsoSS"),
    NonIsoElectronsSiStrip          = cms.InputTag("hltPixelMatchElectronsL1NonIsoSS"),
    PixelSeedL1IsoSiStrip           = cms.InputTag("hltL1IsoSiStripElectronPixelSeeds"),
    PixelSeedL1NonIsoSiStrip        = cms.InputTag("hltL1NonIsoSiStripElectronPixelSeeds"),
    IsoEleTrackIsolSiStrip          = cms.InputTag("hltL1IsoSSEleTrackIsol"),
    NonIsoEleTrackIsolSiStrip       = cms.InputTag("hltL1NonIsoSSEleTrackIsol"),

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


    ### Heavy Ion OpenHLT related objects                             
    Centrality    = cms.InputTag("hiCentrality"),
    CentralityBin    = cms.InputTag("centralityBin"),
    EvtPlane      = cms.InputTag("hiEvtPlane","recoLevel"),
    HiMC          = cms.InputTag("heavyIon"),
                             
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
    IsoPixelTracksHBL2          = cms.InputTag("hltIsolPixelTrackProdHB8E29"),
    IsoPixelTracksHBL3          = cms.InputTag("hltHITIPTCorrectorHB8E29"),
    IsoPixelTracksHEL2          = cms.InputTag("hltIsolPixelTrackProdHE8E29"),
    IsoPixelTracksHEL3          = cms.InputTag("hltHITIPTCorrectorHE8E29"),
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
    
    posCalcParameters = cms.PSet( T0_barl      = cms.double(7.4),
                                  T0_endc      = cms.double(3.1),        
                                  T0_endcPresh = cms.double(1.2),
                                  LogWeighted  = cms.bool(True),
                                  W0           = cms.double(4.2),
                                  X0           = cms.double(0.89)
                                 ),                                         
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
        Debug         = cms.bool(False),

      ### added in 2010 ###
      DoHeavyIon           = cms.untracked.bool(False),

                ### MCTruth
##                DoParticles          = cms.untracked.bool(True),
##                DoRapidity           = cms.untracked.bool(False),
##                DoVerticesByParticle = cms.untracked.bool(True),

                ### Egamma
##                DoPhotons            = cms.untracked.bool(True),
##                DoElectrons          = cms.untracked.bool(True),
##                DoSuperClusters      = cms.untracked.bool(False),

                ### Muon
                DoL1Muons            = cms.untracked.bool(True),
##                DoL2Muons            = cms.untracked.bool(True),
##                DoL3Muons            = cms.untracked.bool(True),
##                DoOfflineMuons       = cms.untracked.bool(True),
##                DoQuarkonia         = cms.untracked.bool(True)
        
    )
)
