import FWCore.ParameterSet.Config as cms

ElectronTauPathVal = cms.EDFilter("HLTTauValidation",
    triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryRAW","","TEST"),
    refTauCollection      = cms.untracked.InputTag("TauMCProducer","Taus"),
    refLeptonCollection   = cms.untracked.InputTag("TauMCProducer","Electrons"),
    DQMFolder             = cms.untracked.string('HLT/HLTTAU/ElectronTau/Path'),
    L1SeedFilter          = cms.untracked.InputTag("hltLevel1GTSeedElectronTau","","TEST"),
    L2EcalIsolFilter      = cms.untracked.InputTag("hltFilterEcalIsolatedTauJetsElectronTau","","TEST"),
    L25PixelIsolFilter    = cms.untracked.InputTag("hltFilterIsolatedTauJetsL25ElectronTau","","TEST"),
    L3SiliconIsolFilter   = cms.untracked.InputTag("DUMMY"),
    MuonFilter            = cms.untracked.InputTag("DUMMY"),
    ElectronFilter        = cms.untracked.InputTag("hltElectronTrackIsolFilterHOneOEMinusOneOPFilterElectronTau","","TEST"),
    NTriggeredTaus        = cms.untracked.uint32(1),
    NTriggeredLeptons     = cms.untracked.uint32(1),
    DoReferenceAnalysis   = cms.untracked.bool(True),
    OutputFileName        = cms.untracked.string(''),
    LogFileName           = cms.untracked.string('ElectronTauValidation.log'),
    MatchDeltaRL1         = cms.untracked.double(0.5),
    MatchDeltaRHLT        = cms.untracked.double(0.3)
)


ElectronTauL2Val = cms.EDFilter("HLTTauCaloDQMOfflineSource",
    DQMFolder              = cms.string('HLT/HLTTAU/ElectronTau/L2'),
    L2InfoAssociationInput = cms.InputTag("hltL2ElectronTauIsolationProducer","L2TauIsolationInfoAssociator"),
    refCollection          = cms.InputTag("TauMCProducer","Taus"),
    MET                    = cms.InputTag("hltMet"),
    doReference            = cms.bool(True),
    MatchDeltaR            = cms.double(0.3),
    OutputFileName         = cms.string(''),
    L2IsolatedJets         = cms.InputTag("hltL2ElectronTauIsolationSelector","Isolated"),
    EtMin                  = cms.double(0.),
    EtMax                  = cms.double(60.),
    NBins                  = cms.int32(30)                            
)


ElectronTauL25Val = cms.EDFilter("HLTTauTrkDQMOfflineSource",
    DQMFolder              = cms.string('HLT/HLTTAU/ElectronTau/L25'),
    ConeIsolation          = cms.InputTag("hltConeIsolationL25ElectronTau"),
    InputJets              = cms.InputTag("hltL2ElectronTauIsolationSelector","Isolated"),                             
    IsolatedJets           = cms.InputTag("hltIsolatedTauJetsSelectorL25ElectronTau"),                             
    refCollection          = cms.InputTag("TauMCProducer","Taus"),
    Type                   = cms.string('L25'),                           
    doReference            = cms.bool(True),
    MatchDeltaR            = cms.double(0.3),
    OutputFileName         = cms.string(''),
    EtMin                  = cms.double(0.),
    EtMax                  = cms.double(60.),
    NBins                  = cms.int32(30)                            
)



ElectronTauElVal = cms.EDFilter("HLTTauElDQMOfflineSource",
    genEtaAcc = cms.double(2.5),
    outputFile = cms.string(''),

    filters = cms.VPSet(cms.PSet(
        PlotBounds = cms.vdouble(0.0, 0.0),
        HLTCollectionLabels = cms.InputTag("hltLevel1GTSeedElectronTau","","TEST"),
        IsoCollections = cms.VInputTag(cms.InputTag("none")),
        theHLTOutputTypes = cms.uint32(83)
    ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltEgammaL1MatchFilterRegionalElectronTau","","TEST"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(100)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltEgammaEtFilterElectronTau","","TEST"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(100)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 10.0),
            HLTCollectionLabels = cms.InputTag("hltEgammaHcalIsolFilterElectronTau","","TEST"),
            IsoCollections = cms.VInputTag(cms.InputTag("l1IsolatedElectronHcalIsol","","TEST")),
            theHLTOutputTypes = cms.uint32(100)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltElectronPixelMatchFilterElectronTau","","TEST"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(100)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltElectronOneOEMinusOneOPFilterElectronTau","","TEST"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(92)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 10.0),
            HLTCollectionLabels = cms.InputTag("hltElectronTrackIsolFilterHOneOEMinusOneOPFilterElectronTau","","TEST"),
            IsoCollections = cms.VInputTag(cms.InputTag("l1IsoElectronTrackIsol","","TEST")),
            theHLTOutputTypes = cms.uint32(92)
        )),
    genEtAcc = cms.double(10.0),
    reqNum = cms.uint32(1),
    triggerName = cms.string('HLT/HLTTAU/ElectronTau/Electron'),
    PtMax = cms.untracked.double(200.0),
    pdgGen = cms.int32(11),
    refCollection = cms.untracked.InputTag("TauMCProducer","Electrons")
)

ElectronTauValidation = cms.Sequence(ElectronTauPathVal + ElectronTauL2Val + ElectronTauL25Val + ElectronTauElVal)



