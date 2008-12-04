import FWCore.ParameterSet.Config as cms


ElectronTauPathVal = cms.EDFilter("HLTTauValidation",
    triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryRAW","","HLT"),
    refTauCollection      = cms.untracked.InputTag("TauMCProducer","HadronicTauOneAndThreeProng"),
    refLeptonCollection   = cms.untracked.InputTag("TauMCProducer","LeptonicTauElectrons"),
    DQMFolder             = cms.untracked.string('HLT/HLTTAU/ElectronTau/Path'),
    Filter                = cms.untracked.VInputTag(
                                   cms.InputTag("hltL1sElectronTau","","HLT"),
                                   cms.InputTag("hltEgammaL1MatchFilterRegionalElectronTau","","HLT"),
                                   cms.InputTag("hltEgammaEtFilterElectronTau","","HLT"),
                                   cms.InputTag("hltEgammaHcalIsolFilterElectronTau","","HLT"),
                                   cms.InputTag("hltElectronPixelMatchFilterElectronTau","","HLT"),
                                   cms.InputTag("hltElectronOneOEMinusOneOPFilterElectronTau","","HLT"),
                                   cms.InputTag("hltElectronTrackIsolFilterHOneOEMinusOneOPFilterElectronTau","","HLT"),
                                   cms.InputTag("hltFilterEcalIsolatedTauJetsElectronTau","","HLT"),
                                   cms.InputTag("hltFilterIsolatedTauJetsL25ElectronTauPtLeadTk","","HLT"),
                                   cms.InputTag("hltFilterIsolatedTauJetsL25ElectronTau","","HLT")
                           ),                           
    MatchDeltaR           = cms.untracked.vdouble(0.5,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3),    #One per filter
    NTriggeredTaus        = cms.untracked.vuint32(1,1,0,0,0,0,0,0,1,1,1), #The first one is for the ref events
    NTriggeredLeptons     = cms.untracked.vuint32(1,1,1,1,1,1,1,1,0,0,0), #the first one is for the ref events
    TauType               = cms.untracked.vint32(86,0,0,0,0,0,0,94,94,94),
    LeptonType            = cms.untracked.vint32(84,92,92,92,92,92,92,0,0,0),                            
    DoReferenceAnalysis   = cms.untracked.bool(True)
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



