import FWCore.ParameterSet.Config as cms

ElectronTauPathVal = cms.EDFilter("HLTTauValidation",
    triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryRAW","","HLT"),
    refTauCollection      = cms.untracked.InputTag("TauMCProducer","HadronicTauOneAndThreeProng"),
    refLeptonCollection   = cms.untracked.InputTag("TauMCProducer","LeptonicTauElectrons"),
    DQMFolder             = cms.untracked.string('HLT/HLTTAU/ElectronTau/Path'),
    L1SeedFilter          = cms.untracked.InputTag("hltL1sElectronTau","","HLT"),
    L2EcalIsolFilter      = cms.untracked.InputTag("hltFilterEcalIsolatedTauJetsElectronTau","","HLT"),
    L25PixelIsolFilter    = cms.untracked.InputTag("hltFilterIsolatedTauJetsL25ElectronTau","","HLT"),
    L3SiliconIsolFilter   = cms.untracked.InputTag("DUMMY"),
    MuonFilter            = cms.untracked.InputTag("DUMMY"),
    ElectronFilter        = cms.untracked.InputTag("hltElectronTrackIsolFilterHOneOEMinusOneOPFilterElectronTau","","HLT"),
    NTriggeredTaus        = cms.untracked.uint32(1),
    NTriggeredLeptons     = cms.untracked.uint32(1),
    DoReferenceAnalysis   = cms.untracked.bool(True),
    OutputFileName        = cms.untracked.string(''),
    LogFileName           = cms.untracked.string(''),
    MatchDeltaRL1         = cms.untracked.double(0.5),
    MatchDeltaRHLT        = cms.untracked.double(0.3)
)


ElectronTauL2Val = cms.EDFilter("HLTTauCaloDQMOfflineSource",
    DQMFolder              = cms.string('HLT/HLTTAU/ElectronTau/L2'),
    L2InfoAssociationInput = cms.InputTag("hltL2ElectronTauIsolationProducer","L2TauIsolationInfoAssociator"),
    refCollection          = cms.InputTag("TauMCProducer","HadronicTauOneAndThreeProng"),
    MET                    = cms.InputTag("hltMet"),
    doReference            = cms.bool(True),
    MatchDeltaR            = cms.double(0.3),
    OutputFileName         = cms.string(''),
    L2IsolatedJets         = cms.InputTag("hltL2ElectronTauIsolationSelector","Isolated"),
    EtMin                  = cms.double(0.),
    EtMax                  = cms.double(100.),
    NBins                  = cms.int32(50)                            
)


ElectronTauL25Val = cms.EDFilter("HLTTauTrkDQMOfflineSource",
    DQMFolder              = cms.string('HLT/HLTTAU/ElectronTau/L25'),
    ConeIsolation          = cms.InputTag("hltConeIsolationL25ElectronTau"),
    InputJets              = cms.InputTag("hltL2ElectronTauIsolationSelector","Isolated"),                             
    IsolatedJets           = cms.InputTag("hltIsolatedTauJetsSelectorL25ElectronTau"),                             
    refCollection          = cms.InputTag("TauMCProducer","HadronicTauOneAndThreeProng"),
    Type                   = cms.string('L25'),                           
    doReference            = cms.bool(True),
    MatchDeltaR            = cms.double(0.3),
    OutputFileName         = cms.string(''),
    EtMin                  = cms.double(0.),
    EtMax                  = cms.double(100.),
    NBins                  = cms.int32(50)                            
)



ElectronTauElVal = cms.EDFilter("HLTTauElDQMOfflineSource",
    genEtaAcc = cms.double(2.5),
    outputFile = cms.string(''),

    filters = cms.VPSet(cms.PSet(
        PlotBounds = cms.vdouble(0.0, 0.0),
        HLTCollectionLabels = cms.InputTag("hltLevel1GTSeedElectronTau","","HLT"),
        IsoCollections = cms.VInputTag(cms.InputTag("none")),
        theHLTOutputTypes = cms.uint32(83)
    ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltEgammaL1MatchFilterRegionalElectronTau","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(100)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltEgammaEtFilterElectronTau","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(100)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 10.0),
            HLTCollectionLabels = cms.InputTag("hltEgammaHcalIsolFilterElectronTau","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("l1IsolatedElectronHcalIsol","","HLT")),
            theHLTOutputTypes = cms.uint32(100)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltElectronPixelMatchFilterElectronTau","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(100)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 0.0),
            HLTCollectionLabels = cms.InputTag("hltElectronOneOEMinusOneOPFilterElectronTau","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("none")),
            theHLTOutputTypes = cms.uint32(92)
        ), 
        cms.PSet(
            PlotBounds = cms.vdouble(0.0, 10.0),
            HLTCollectionLabels = cms.InputTag("hltElectronTrackIsolFilterHOneOEMinusOneOPFilterElectronTau","","HLT"),
            IsoCollections = cms.VInputTag(cms.InputTag("l1IsoElectronTrackIsol","","HLT")),
            theHLTOutputTypes = cms.uint32(92)
        )),
    genEtAcc = cms.double(10.0),
    reqNum = cms.uint32(1),
    triggerName = cms.string('HLT/HLTTAU/ElectronTau/Electron'),
    PtMax = cms.untracked.double(200.0),
    pdgGen = cms.int32(11),
    refCollection = cms.untracked.InputTag("TauMCProducer","LeptonicTauElectrons")
)

ElectronTauValidation = cms.Sequence(ElectronTauPathVal + ElectronTauL2Val + ElectronTauL25Val + ElectronTauElVal)



