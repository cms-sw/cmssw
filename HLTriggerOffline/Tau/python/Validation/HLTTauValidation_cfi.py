import FWCore.ParameterSet.Config as cms

DoubleTauPathVal = cms.EDFilter("HLTTauValidation",
    triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryRAW","","HLT"),
    refTauCollection      = cms.untracked.InputTag("TauMCProducer","HadronicTauOneAndThreeProng"),
    refLeptonCollection   = cms.untracked.InputTag("NOTHING"),
    DQMFolder             = cms.untracked.string('HLT/HLTTAU/DoubleTau'),
    Filter                = cms.untracked.VInputTag(
                                     cms.InputTag("hltL1sDoubleTau40","","HLT"), 
                                     cms.InputTag("hltFilterL2EcalIsolationDoubleTau","","HLT"),
                                     cms.InputTag("hltFilterL25PixelTracksLeadingTrackPtCutDoubleTau","","HLT"),
                                     cms.InputTag("hltFilterL25PixelTracksIsolationDoubleTau","","HLT")
                                     ),
    MatchDeltaR           = cms.untracked.vdouble(0.5,0.3,0.3,0.3),    #One per filter
    NTriggeredTaus        = cms.untracked.vuint32(2,2,2,2,2), #The first one is for the ref events
    NTriggeredLeptons     = cms.untracked.vuint32(0,0,0,0,0), #the first one is for the ref events
    TauType               = cms.untracked.vint32(86,94,94,94),
    LeptonType            = cms.untracked.vint32(0,0,0,0),                            
    DoReferenceAnalysis   = cms.untracked.bool(True),
)

SingleTauPathVal = cms.EDFilter("HLTTauValidation",
    triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryRAW","","HLT"),
    refTauCollection      = cms.untracked.InputTag("TauMCProducer","HadronicTauOneAndThreeProng"),
    refLeptonCollection   = cms.untracked.InputTag("NOTHING"),
    DQMFolder             = cms.untracked.string('HLT/HLTTAU/SingleTau'),
    MatchDeltaR           = cms.untracked.vdouble(0.5,0.0,0.3,0.3,0.3,0.3),    #One per filter
    Filter                = cms.untracked.VInputTag(
                                     cms.InputTag("hltL1sSingleTau80","","HLT"),
                                     cms.InputTag("hlt1METSingleTau","","HLT"),
                                     cms.InputTag("hltFilterL2EcalIsolationSingleTau","","HLT"),
                                     cms.InputTag("hltFilterL25PixelTracksLeadingTrackPtCutSingleTau","","HLT"),
                                     cms.InputTag("hltFilterL25PixelTracksIsolationSingleTau","","HLT"),
                                     cms.InputTag("hltFilterL3LeadingTrackPtCutSingleTau","","HLT")
                                     ),
    NTriggeredTaus        = cms.untracked.vuint32(1,1,0,1,1,1,1), #The first one is for the ref events
    NTriggeredLeptons     = cms.untracked.vuint32(0,0,0,0,0,0,0), #the first one is for the ref events
    TauType               = cms.untracked.vint32(86,0,94,94,94,94),
    LeptonType            = cms.untracked.vint32(0,0,0,0,0),                            
    DoReferenceAnalysis   = cms.untracked.bool(True)
)


SingleTauMETPathVal = cms.EDFilter("HLTTauValidation",
    triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryRAW","","HLT"),
    refTauCollection      = cms.untracked.InputTag("TauMCProducer","HadronicTauOneAndThreeProng"),
    refLeptonCollection   = cms.untracked.InputTag("NOTHING"),
    DQMFolder             = cms.untracked.string('HLT/HLTTAU/SingleTau'),
    MatchDeltaR           = cms.untracked.vdouble(0.5,0.3,0.0,0.3,0.3,0.3),    #One per filter
    Filter                = cms.untracked.VInputTag(
                                     cms.InputTag("hltL1sTau30ETM30","","HLT"),
                                     cms.InputTag("hltFilterL2EcalIsolationSingleTauMET","","HLT"),
                                     cms.InputTag("hlt1METSingleTauMET","","HLT"),
                                     cms.InputTag("hltFilterL25PixelTracksLeadingTrackPtCutSingleTauMET","","HLT"),
                                     cms.InputTag("hltFilterL25PixelTracksIsolationSingleTauMET","","HLT"),
                                     cms.InputTag("hltFilterL3LeadingTrackPtCutSingleTauMET","","HLT")
                                     ),
    NTriggeredTaus        = cms.untracked.vuint32(1,1,1,0,1,1,1), #The first one is for the ref events
    NTriggeredLeptons     = cms.untracked.vuint32(0,0,0,0,0,0,0), #the first one is for the ref events
    TauType               = cms.untracked.vint32(86,94,0,94,94,94),
    LeptonType            = cms.untracked.vint32(0,0,0,0,0),                            
    DoReferenceAnalysis   = cms.untracked.bool(True)
)


ElectronTauPathVal = cms.EDFilter("HLTTauValidation",
    triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryRAW","","HLT"),
    refTauCollection      = cms.untracked.InputTag("TauMCProducer","HadronicTauOneAndThreeProng"),
    refLeptonCollection   = cms.untracked.InputTag("TauMCProducer","LeptonicTauElectrons"),
    DQMFolder             = cms.untracked.string('HLT/HLTTAU/ElectronTau'),
    Filter                = cms.untracked.VInputTag(
                                   cms.InputTag("hltL1sIsoEG10Tau20","","HLT"),
                                   cms.InputTag("hltEgammaL1MatchFilterRegionalElectronTau","","HLT"),
                                   cms.InputTag("hltEgammaEtFilterElectronTau","","HLT"),
                                   cms.InputTag("hltEgammaHcalIsolFilterElectronTau","","HLT"),
                                   cms.InputTag("hltElectronPixelMatchFilterElectronTau","","HLT"),
                                   cms.InputTag("hltElectronOneOEMinusOneOPFilterElectronTau","","HLT"),
                                   cms.InputTag("hltElectronTrackIsolFilterHOneOEMinusOneOPFilterElectronTau","","HLT"),
                                   cms.InputTag("hltFilterL2EcalIsolationElectronTau","","HLT"),
                                   cms.InputTag("hltFilterL25PixelTracksLeadingTrackPtCutElectronTau","","HLT"),
                                   cms.InputTag("hltFilterL25PixelTracksIsolationElectronTau","","HLT")
                           ),                           
    MatchDeltaR           = cms.untracked.vdouble(0.5,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3),    #One per filter
    NTriggeredTaus        = cms.untracked.vuint32(1,1,0,0,0,0,0,0,1,1,1), #The first one is for the ref events
    NTriggeredLeptons     = cms.untracked.vuint32(1,1,1,1,1,1,1,1,0,0,0), #the first one is for the ref events
    TauType               = cms.untracked.vint32(86,0,0,0,0,0,0,94,94,94),
    LeptonType            = cms.untracked.vint32(84,92,92,92,92,92,92,0,0,0),                            
    DoReferenceAnalysis   = cms.untracked.bool(True)
)


MuonTauPathVal = cms.EDFilter("HLTTauValidation",
    triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryRAW","","HLT"),
    refTauCollection      = cms.untracked.InputTag("TauMCProducer","HadronicTauOneAndThreeProng"),
    refLeptonCollection   = cms.untracked.InputTag("TauMCProducer","LeptonicTauMuons"),
    DQMFolder             = cms.untracked.string('HLT/HLTTAU/MuonTau'),
    Filter                = cms.untracked.VInputTag(
                                   cms.InputTag("hltL1sMuon5Tau20","","HLT"),
                                   cms.InputTag("hltMuonTauL1Filtered","","HLT"),
                                   cms.InputTag("hltMuonTauIsoL2PreFiltered","","HLT"),
                                   cms.InputTag("hltMuonTauIsoL2IsoFiltered","","HLT"),
                                   cms.InputTag("hltMuonTauIsoL3PreFiltered","","HLT"),
                                   cms.InputTag("hltMuonTauIsoL3IsoFiltered","","HLT"),
                                   cms.InputTag("hltFilterL2EcalIsolationMuonTau","","HLT"),
                                   cms.InputTag("hltFilterL25PixelTracksLeadingTrackPtCutMuonTau","","HLT"),
                                   cms.InputTag("hltFilterL25PixelTracksIsolationMuonTau","","HLT")
                           ),                           
    MatchDeltaR           = cms.untracked.vdouble(0.5,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3),    #One per filter
    NTriggeredTaus        = cms.untracked.vuint32(1,1,0,0,0,0,0,1,1,1), #The first one is for the ref events
    NTriggeredLeptons     = cms.untracked.vuint32(1,1,1,1,1,1,1,0,0,0), #the first one is for the ref events
    TauType               = cms.untracked.vint32(86,0,0,0,0,0,94,94,94),
    LeptonType            = cms.untracked.vint32(81,81,93,93,93,93,0,0,0),                            
    DoReferenceAnalysis   = cms.untracked.bool(True),

)


L2Val = cms.EDFilter("HLTTauCaloDQMOfflineSource",
    DQMFolder              = cms.string('HLT/HLTTAU/L2'),
    L2InfoAssociationInput = cms.InputTag("hltL2TauIsolationProducer","L2TauIsolationInfoAssociator"),
    refCollection          = cms.InputTag("TauMCProducer","HadronicTauOneAndThreeProng"),
    MET                    = cms.InputTag("hltMet"),
    doReference            = cms.bool(True),
    MatchDeltaR            = cms.double(0.3),
    OutputFileName         = cms.string(''),
    L2IsolatedJets         = cms.InputTag("hltL2TauIsolationSelector","Isolated"),
    EtMin                  = cms.double(0.),
    EtMax                  = cms.double(100.),
    NBins                  = cms.int32(20)                            
)


L25Val = cms.EDFilter("HLTTauTrkDQMOfflineSource",
    DQMFolder              = cms.string('HLT/HLTTAU/L25'),
    ConeIsolation          = cms.InputTag("hltL25TauPixelTracksConeIsolation"),
    InputJets              = cms.InputTag("hltL2TauIsolationSelector","Isolated"),                             
    IsolatedJets           = cms.InputTag("hltL25TauPixelTracksIsolationSelector"),                             
    refCollection          = cms.InputTag("TauMCProducer","HadronicTauOneAndThreeProng"),
    Type                   = cms.string('L25'),                           
    doReference            = cms.bool(True),
    MatchDeltaR            = cms.double(0.3),
    OutputFileName         = cms.string(''),
    EtMin                  = cms.double(0.),
    EtMax                  = cms.double(100.),
    NBins                  = cms.int32(20)                            
)

L3Val = cms.EDFilter("HLTTauTrkDQMOfflineSource",
    DQMFolder              = cms.string('HLT/HLTTAU/L3'),
    ConeIsolation          = cms.InputTag("hltL3TauConeIsolation"),
    InputJets              = cms.InputTag("hltL25TauPixelTracksIsolationSelector"),                             
    IsolatedJets           = cms.InputTag("hltL3TauIsolationSelector"),                             
    refCollection          = cms.InputTag("TauMCProducer","HadronicTauOneAndThreeProng"),
    Type                   = cms.string('L3'),                           
    doReference            = cms.bool(True),
    MatchDeltaR            = cms.double(0.3),
    OutputFileName         = cms.string(''),
    EtMin                  = cms.double(0.),
    EtMax                  = cms.double(100.),
    NBins                  = cms.int32(20)                            
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



HLTTauValidationSequence = cms.Sequence(DoubleTauPathVal + SingleTauPathVal + SingleTauMETPathVal+ ElectronTauPathVal+ MuonTauPathVal+L2Val + L25Val + L3Val+ElectronTauElVal)

