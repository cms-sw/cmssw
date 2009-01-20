import FWCore.ParameterSet.Config as cms

hltanalysis = cms.EDAnalyzer("HLTAnalyzer",
    ### generator objects
    mctruth                         = cms.InputTag("genParticles"),
    genEventScale                   = cms.InputTag("genEventScale"),
    
    ### Trigger objects
   #l1GctCounts                     = cms.InputTag("l1GctEmulDigis"),
   #l1GtObjectMapRecord             = cms.InputTag("l1GtEmulDigis"),
   #l1GtReadoutRecord               = cms.InputTag("l1GmtEmulDigis"),
    l1GctCounts                     = cms.InputTag("hltGctDigis"),
    l1GtObjectMapRecord             = cms.InputTag("hltL1GtObjectMap"),
    l1GtReadoutRecord               = cms.InputTag("hltGtDigis"),
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
   #Photon                          = cms.InputTag("correctedPhotons"),
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
    IsoPhoTrackIsol                 = cms.InputTag("hltL1IsoPhotonTrackIsol"),
    NonIsoPhoTrackIsol              = cms.InputTag("hltL1NonIsoPhotonTrackIsol"),
    HcalIsoEle                      = cms.InputTag("hltL1IsolatedElectronHcalIsol"),
    HcalNonIsoEle                   = cms.InputTag("hltL1NonIsolatedElectronHcalIsol"),
    ### egamma - standard or startup windows                         
    IsoElectrons                    = cms.InputTag("hltPixelMatchStartUpElectronsL1Iso"),
   #IsoElectrons                    = cms.InputTag("hltPixelMatchElectronsL1Iso"),
    NonIsoElectrons                 = cms.InputTag("hltPixelMatchStartUpElectronsL1NonIso"),
   #NonIsoElectrons                 = cms.InputTag("hltPixelMatchElectronsL1NonIso"),
    PixelSeedL1Iso                  = cms.InputTag("hltL1IsoStartUpElectronPixelSeeds"),
   #PixelSeedL1Iso                  = cms.InputTag("hltL1IsoElectronPixelSeeds"),
    PixelSeedL1NonIso               = cms.InputTag("hltL1NonIsoStartUpElectronPixelSeeds"),
   #PixelSeedL1NonIso               = cms.InputTag("hltL1NonIsoElectronPixelSeeds"),
    IsoEleTrackIsol                 = cms.InputTag("hltL1IsoStartUpElectronTrackIsol"),
   #IsoEleTrackIsol                 = cms.InputTag("hltL1IsoElectronTrackIsol"),
    NonIsoEleTrackIsol              = cms.InputTag("hltL1NonIsoStartupElectronTrackIsol"),
   #NonIsoEleTrackIsol              = cms.InputTag("hltL1NonIsoElectronTrackIsol"),
    ### egamma - large windows
    IsoElectronsLargeWindows        = cms.InputTag("hltPixelMatchElectronsL1IsoLargeWindow"),
    NonIsoElectronsLargeWindows     = cms.InputTag("hltPixelMatchElectronsL1NonIsoLargeWindow"),
    PixelSeedL1IsoLargeWindows      = cms.InputTag("hltL1IsoLargeWindowElectronPixelSeeds"),
    PixelSeedL1NonIsoLargeWindows   = cms.InputTag("hltL1NonIsoLargeWindowElectronPixelSeeds"),
    IsoEleTrackIsolLargeWindows     = cms.InputTag("hltL1IsoLargeWindowElectronTrackIsol"),
    NonIsoEleTrackIsolLargeWindows  = cms.InputTag("hltL1NonIsoLargeWindowElectronTrackIsol"),

    ### tau OpenHLT related objects
    HLTTau                          = cms.InputTag("TauOpenHLT"),
    
    ### b-jet OpenHLT related objects
    CommonBJetsL2                   = cms.InputTag("hltIterativeCone5CaloJets"),
    CorrectedBJetsL2                = cms.InputTag("hltMCJetCorJetIcone5"),
    LifetimeBJetsL25                = cms.InputTag("openHltBLifetimeL25BJetTags"),
    LifetimeBJetsL3                 = cms.InputTag("openHltBLifetimeL3BJetTags"),
    LifetimeBJetsL25Relaxed         = cms.InputTag("openHltBLifetimeL25BJetTags"),
    LifetimeBJetsL3Relaxed          = cms.InputTag("openHltBLifetimeL3BJetTagsRelaxed"),
    SoftmuonBJetsL25                = cms.InputTag("openHltBSoftmuonL25BJetTags"),
    SoftmuonBJetsL3                 = cms.InputTag("openHltBSoftmuonL3BJetTags"),
    PerformanceBJetsL25             = cms.InputTag("openHltBSoftmuonL25BJetTags"),
    PerformanceBJetsL3              = cms.InputTag("openHltBPerfMeasL3BJetTags"),
    
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
