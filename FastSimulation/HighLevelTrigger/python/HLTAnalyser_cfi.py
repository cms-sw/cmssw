import FWCore.ParameterSet.Config as cms

hltanalysis = cms.EDAnalyzer("HLTAnalyzer",
### Generator objects
    #mctruth = cms.InputTag("genParticleCandidates"),
    mctruth = cms.InputTag("genParticles"),
    genEventScale = cms.InputTag("genEventScale"),

### Trigger objects
#    l1GctCounts = cms.InputTag("l1GctEmulDigis"),
    l1GctCounts = cms.InputTag("gctDigis"),
#    l1GtObjectMapRecord = cms.InputTag("l1GtEmulDigis"),
    l1GtObjectMapRecord = cms.InputTag("simGtDigis"),
#    l1GtReadoutRecord = cms.InputTag("l1GmtEmulDigis"),
    l1GtReadoutRecord = cms.InputTag("simGtDigis"),
    l1extramc = cms.string('l1extraParticles'),
    l1extramu = cms.string('l1extraParticles'),
    hltresults = cms.InputTag("TriggerResults"),

### Reconstructed objects
    genjets = cms.InputTag("ak4GenJets"),
    genmet = cms.InputTag("genMet"),
    recjets = cms.InputTag("hltIterativeCone5CaloJets"),
    recmet = cms.InputTag("hltMet"),
    ht = cms.InputTag("hltHtMet"),
    calotowers = cms.InputTag("towerMaker"),
    muon = cms.InputTag("muons"),
    Electron = cms.InputTag("pixelMatchGsfElectrons"),
#    Photon = cms.InputTag("correctedPhotons"),
    Photon = cms.InputTag("photons"),

### Muon OpenHLT objects                             
    MuCandTag2 = cms.InputTag("hltL2MuonCandidates"),
    MuCandTag3 = cms.InputTag("hltL3MuonCandidates"),
    MuIsolTag3 = cms.InputTag("hltL3MuonIsolations"),
    MuIsolTag2 = cms.InputTag("hltL2MuonIsolations"),
    MuLinkTag = cms.InputTag("hltL3Muons"),

### Egamma OpenHLT objects                             
    CandIso = cms.InputTag("hltL1IsoRecoEcalCandidate"),
    CandNonIso = cms.InputTag("hltL1NonIsoRecoEcalCandidate"),
    EcalIso =cms.InputTag("hltL1IsolatedPhotonEcalIsol"),
    EcalNonIso =cms.InputTag("hltL1NonIsolatedPhotonEcalIsol"),
    HcalIsoPho =cms.InputTag("hltL1IsolatedPhotonHcalIsol"),
    HcalNonIsoPho =cms.InputTag("hltL1NonIsolatedPhotonHcalIsol"),
    IsoPhoTrackIsol =cms.InputTag("hltL1IsoPhotonTrackIsol"),
    NonIsoPhoTrackIsol =cms.InputTag("hltL1NonIsoPhotonTrackIsol"),
    HcalIsoEle =cms.InputTag("hltL1IsolatedElectronHcalIsol"),
    HcalNonIsoEle  =cms.InputTag("hltL1NonIsolatedElectronHcalIsol"),
    #### Standard or Startup windows                         
    #IsoElectrons =cms.InputTag("hltPixelMatchElectronsL1Iso"),
    #NonIsoElectrons =cms.InputTag("hltPixelMatchElectronsL1NonIso"),
    #PixelSeedL1Iso =cms.InputTag("hltL1IsoElectronPixelSeeds"),
    #PixelSeedL1NonIso =cms.InputTag("hltL1NonIsoElectronPixelSeeds"),
    #IsoEleTrackIsol =cms.InputTag("hltL1IsoElectronTrackIsol"),
    #NonIsoEleTrackIsol =cms.InputTag("hltL1NonIsoElectronTrackIsol"),
    IsoElectrons =cms.InputTag("hltPixelMatchStartUpElectronsL1Iso"),
    NonIsoElectrons =cms.InputTag("hltPixelMatchStartUpElectronsL1NonIso"),
    PixelSeedL1Iso =cms.InputTag("hltL1IsoStartUpElectronPixelSeeds"),
    PixelSeedL1NonIso =cms.InputTag("hltL1NonIsoStartUpElectronPixelSeeds"),
    IsoEleTrackIsol =cms.InputTag("hltL1IsoStartUpElectronTrackIsol"),
    NonIsoEleTrackIsol =cms.InputTag("hltL1NonIsoStartupElectronTrackIsol"),
    ### Large windows
    IsoElectronsLargeWindows =cms.InputTag("hltPixelMatchElectronsL1IsoLargeWindow"),
    NonIsoElectronsLargeWindows =cms.InputTag("hltPixelMatchElectronsL1NonIsoLargeWindow"),
    PixelSeedL1IsoLargeWindows =cms.InputTag("hltL1IsoLargeWindowElectronPixelSeeds"),
    PixelSeedL1NonIsoLargeWindows =cms.InputTag("hltL1NonIsoLargeWindowElectronPixelSeeds"),
    IsoEleTrackIsolLargeWindows =cms.InputTag("hltL1IsoLargeWindowElectronTrackIsol"),
    NonIsoEleTrackIsolLargeWindows =cms.InputTag("hltL1NonIsoLargeWindowElectronTrackIsol"),
    HLTTau =cms.InputTag("TauOpenHLT"),

    ########
    CommonBJetsL2                   = cms.InputTag("hltIterativeCone5CaloJets"),
    CorrectedBJetsL2                = cms.InputTag("hltMCJetCorJetIcone5"),
    LifetimeBJetsL2     = cms.InputTag("hltBLifetimeL25Jets"),              # or "hltBLifetimeL25JetsRelaxed"
    LifetimeBJetsL25    = cms.InputTag("hltBLifetimeL25BJetTags"),          # or "hltBLifetimeL25BJetTagsRelaxed"
    LifetimeBJetsL3     = cms.InputTag("openHltBLifetimeL3BJetTags"),       # or "openHltBLifetimeL3BJetTagsRelaxed"
    LifetimeBJetsL25Relaxed         = cms.InputTag("openHltBLifetimeL25BJetTags"),
    LifetimeBJetsL3Relaxed          = cms.InputTag("openHltBLifetimeL3BJetTagsStartup"),
    SoftmuonBJetsL2     = cms.InputTag("hltBSoftmuonL25Jets"),
    SoftmuonBJetsL25    = cms.InputTag("hltBSoftmuonL25BJetTags"),
    SoftmuonBJetsL3     = cms.InputTag("hltBSoftmuonL3BJetTags"),
    PerformanceBJetsL2  = cms.InputTag("hltBSoftmuonL25Jets"),              # L2 and L2.5 share the same collections as soft muon
    PerformanceBJetsL25 = cms.InputTag("hltBSoftmuonL25BJetTags"),          # L2 and L2.5 share the same collections as soft muon
    PerformanceBJetsL3  = cms.InputTag("hltBSoftmuonL3BJetTagsByDR"),
    ########
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



