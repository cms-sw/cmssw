import FWCore.ParameterSet.Config as cms

hltTauDQMProcess = "HLT"

hltTauMonitor = cms.EDAnalyzer("HLTTauDQMSource",
    MonitorSetup = cms.VPSet(
        cms.PSet( #HLT_DoubleIsoPFTau45_Trk5_eta2p1_v3
            ConfigType            = cms.untracked.string("Path"),
            triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryRAW","",hltTauDQMProcess),
            DQMFolder             = cms.untracked.string('HLT/TauOnline/Inclusive/DoubleTau'),
            Filter                = cms.untracked.VInputTag(
                                        cms.InputTag("hltL1sDoubleTauJet44Eta2p17orDoubleJet64Central","",hltTauDQMProcess), 
                                        cms.InputTag("hltFilterL2EtCutDoublePFIsoTau45Trk5","",hltTauDQMProcess),
                                        cms.InputTag("hltDoublePFTauTightIso45Track","",hltTauDQMProcess),
                                        cms.InputTag("hltDoublePFTauTightIso45Track5","",hltTauDQMProcess),
                                        cms.InputTag("hltFilterDoubleIsoPFTau45Trk5LeadTrack5IsolationL1HLTMatched","",hltTauDQMProcess)
                                    ),
            MatchDeltaR           = cms.untracked.vdouble(0.5,0.2,0.2,0.2,0.2),    #One per filter
            NTriggeredTaus        = cms.untracked.vuint32(2,2,2,2,2,2), #The first one is for the ref events
            NTriggeredLeptons     = cms.untracked.vuint32(0,0,0,0,0,0), #the first one is for the ref events
            TauType               = cms.untracked.vint32(-86,84,84,84,84),
            LeptonType            = cms.untracked.vint32(0,0,0,0,0)                            
        ),
        cms.PSet( #HLT_Ele20_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_MediumIsoPFTau20_v1
            ConfigType            = cms.untracked.string("Path"),
            triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryRAW","",hltTauDQMProcess),
            DQMFolder             = cms.untracked.string('HLT/TauOnline/Inclusive/EleTau'),
            Filter                = cms.untracked.VInputTag(
                                        cms.InputTag("hltL1sL1SingleEG18orL1SingleEG20","",hltTauDQMProcess), 
                                        cms.InputTag("hltEle20CaloIdVTCaloIsoTTrkIdTTrkIsoTTrackIsoFilterL1SingleEG18orL1SingleEG20","",hltTauDQMProcess),
                                        cms.InputTag("hltPFTauMediumIso20","",hltTauDQMProcess),
                                        cms.InputTag("hltPFTauMediumIso20Track","",hltTauDQMProcess),
                                        cms.InputTag("hltPFTauMediumIso20TrackMediumIso","",hltTauDQMProcess),
                                        cms.InputTag("hltOverlapFilterIsoEle20MediumIsoPFTau20","",hltTauDQMProcess)
                                    ),
            MatchDeltaR           = cms.untracked.vdouble(0.5,0.2,0.2,0.2,0.2,0.2),    #One per filter
            NTriggeredTaus        = cms.untracked.vuint32(1,0,0,1,1,1,1), #The first one is for the ref events
            NTriggeredLeptons     = cms.untracked.vuint32(1,1,1,0,0,0,1), #the first one is for the ref events
            TauType               = cms.untracked.vint32(0,0,84,84,84,84),
            LeptonType            = cms.untracked.vint32(-83,82,0,0,0,82)                            
        ),
        cms.PSet( #HLT_IsoMu15_eta2p1_LooseIsoPFTau20_v1
            ConfigType            = cms.untracked.string("Path"),
            triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryRAW","",hltTauDQMProcess),
            DQMFolder             = cms.untracked.string('HLT/TauOnline/Inclusive/MuTau'),
            Filter                = cms.untracked.VInputTag(
                                        cms.InputTag("hltL1sL1SingleMu14Eta2p1","",hltTauDQMProcess), 
                                        cms.InputTag("hltSingleMuIsoL1s14L3IsoFiltered15eta2p1","",hltTauDQMProcess),
                                        cms.InputTag("hltPFTau20Track","",hltTauDQMProcess),
                                        cms.InputTag("hltPFTau20TrackLooseIso","",hltTauDQMProcess),
                                        cms.InputTag("hltOverlapFilterIsoMu15IsoPFTau20","",hltTauDQMProcess)
                                    ),
            MatchDeltaR           = cms.untracked.vdouble(0.5,0.2,0.2,0.2,0.2),    #One per filter
            NTriggeredTaus        = cms.untracked.vuint32(1,0,0,1,1,1), #The first one is for the ref events
            NTriggeredLeptons     = cms.untracked.vuint32(1,1,1,0,0,1), #the first one is for the ref events
            TauType               = cms.untracked.vint32(0,0,84,84,84),
            LeptonType            = cms.untracked.vint32(-81,83,0,0,83)                            
        ),
        cms.PSet(
            ConfigType            = cms.untracked.string("LitePath"),
            triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryAOD","",hltTauDQMProcess),
            DQMFolder             = cms.untracked.string('HLT/TauOnline/Inclusive/Summary'),
            Filter                = cms.untracked.VInputTag(
                                        cms.InputTag("hltFilterDoubleIsoPFTau45Trk5LeadTrack5IsolationL1HLTMatched","",hltTauDQMProcess),
                                        cms.InputTag("hltOverlapFilterIsoEle20MediumIsoPFTau20","",hltTauDQMProcess),
                                        cms.InputTag("hltOverlapFilterIsoMu15IsoPFTau20","",hltTauDQMProcess),
                                        cms.InputTag("hltFilterSingleIsoPFTau35Trk20MET70LeadTrack20IsolationL1HLTMatched","",hltTauDQMProcess)
                                    ),
            PathName              = cms.untracked.vstring(
                                        'DoubleIsoTau',
                                        'EleTau',
                                        'MuTau',
                                        'SingleIsoTau'
                                    ),
            NTriggeredTaus        = cms.untracked.vuint32(2,1,1,1), 
            NTriggeredLeptons     = cms.untracked.vuint32(0,1,1,0), 
            TauType               = cms.untracked.vint32(15,15,15,15),
            LeptonType            = cms.untracked.vint32(0,11,13,0)                            
        ),
        cms.PSet( #HLT_MediumIsoPFTau35_Trk20_MET70_v1
            ConfigType            = cms.untracked.string("Path"),
            triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryRAW","",hltTauDQMProcess),
            DQMFolder             = cms.untracked.string('HLT/TauOnline/Inclusive/SingleTau'),
            Filter                = cms.untracked.VInputTag(
                                        cms.InputTag("hltL1sL1Jet52ETM30","",hltTauDQMProcess), 
                                        cms.InputTag("hltFilterL2EtCutSingleIsoPFTau35Trk20MET70","",hltTauDQMProcess),
                                        cms.InputTag("hltPFTauMediumIso35","",hltTauDQMProcess),
                                        cms.InputTag("hltPFTauMediumIso35Track","",hltTauDQMProcess),
                                        cms.InputTag("hltFilterSingleIsoPFTau35Trk20LeadTrackPt20","",hltTauDQMProcess),
                                        cms.InputTag("hltFilterSingleIsoPFTau35Trk20MET70LeadTrack20IsolationL1HLTMatched","",hltTauDQMProcess)
                                    ),
            MatchDeltaR           = cms.untracked.vdouble(0.5,0.2,0.2,0.2,0.2,0.2),    #One per filter
            NTriggeredTaus        = cms.untracked.vuint32(1,1,1,1,1,1,1), #The first one is for the ref events
            NTriggeredLeptons     = cms.untracked.vuint32(0,0,0,0,0,0,0), #the first one is for the ref events
            TauType               = cms.untracked.vint32(-86,84,84,84,84,84),
            LeptonType            = cms.untracked.vint32(0,0,0,0,0,0)                         
        ),
        cms.PSet(
            ConfigType            = cms.untracked.string("L1"),
            DQMFolder             = cms.string('HLT/TauOnline/Inclusive/L1'),
            L1Taus                = cms.InputTag("hltL1extraParticles","Tau"),
            L1Jets                = cms.InputTag("hltL1extraParticles","Central"),
            L1Electrons           = cms.InputTag("hltL1extraParticles","Isolated"),
            L1Muons               = cms.InputTag("hltL1extraParticles"),
        ),
    ),
    
    doMatching = cms.bool(False),
    matchFilter = cms.untracked.VInputTag(cms.InputTag("","",hltTauDQMProcess) ),
    matchObjectID = cms.untracked.vint32(0),
    matchObjectMinPt = cms.untracked.vdouble(15),
    TriggerEvent = cms.InputTag("hltTriggerSummaryAOD","",hltTauDQMProcess )                          
)




hltTauElectronMonitor = cms.EDAnalyzer("HLTTauDQMSource",
    MonitorSetup = cms.VPSet(
        cms.PSet( #HLT_DoubleIsoPFTau45_Trk5_eta2p1_v3
            ConfigType            = cms.untracked.string("Path"),
            triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryRAW","",hltTauDQMProcess),
            DQMFolder             = cms.untracked.string('HLT/TauOnline/Electrons/DoubleTau'),
            Filter                = cms.untracked.VInputTag(
                                        cms.InputTag("hltL1sDoubleTauJet44Eta2p17orDoubleJet64Central","",hltTauDQMProcess), 
                                        cms.InputTag("hltFilterL2EtCutDoublePFIsoTau45Trk5","",hltTauDQMProcess),
                                        cms.InputTag("hltDoublePFTauTightIso45Track","",hltTauDQMProcess),
                                        cms.InputTag("hltDoublePFTauTightIso45Track5","",hltTauDQMProcess),
                                        cms.InputTag("hltFilterDoubleIsoPFTau45Trk5LeadTrack5IsolationL1HLTMatched","",hltTauDQMProcess)
                                    ),
            MatchDeltaR           = cms.untracked.vdouble(0.5,0.2,0.2,0.2,0.2),    #One per filter
            NTriggeredTaus        = cms.untracked.vuint32(2,2,2,2,2,2), #The first one is for the ref events
            NTriggeredLeptons     = cms.untracked.vuint32(0,0,0,0,0,0), #the first one is for the ref events
            TauType               = cms.untracked.vint32(-86,84,84,84,84),
            LeptonType            = cms.untracked.vint32(0,0,0,0,0)                            
        ),
        cms.PSet( #HLT_MediumIsoPFTau35_Trk20_MET70_v1
            ConfigType            = cms.untracked.string("Path"),
            triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryRAW","",hltTauDQMProcess),
            DQMFolder             = cms.untracked.string('HLT/TauOnline/Electrons/SingleTau'),
            Filter                = cms.untracked.VInputTag(
                                        cms.InputTag("hltL1sL1Jet52ETM30","",hltTauDQMProcess), 
                                        cms.InputTag("hltFilterL2EtCutSingleIsoPFTau35Trk20MET70","",hltTauDQMProcess),
                                        cms.InputTag("hltPFTauMediumIso35","",hltTauDQMProcess),
                                        cms.InputTag("hltPFTauMediumIso35Track","",hltTauDQMProcess),
                                        cms.InputTag("hltFilterSingleIsoPFTau35Trk20LeadTrackPt20","",hltTauDQMProcess),
                                        cms.InputTag("hltFilterSingleIsoPFTau35Trk20MET70LeadTrack20IsolationL1HLTMatched","",hltTauDQMProcess)
                                    ),
            MatchDeltaR           = cms.untracked.vdouble(0.5,0.2,0.2,0.2,0.2,0.2),    #One per filter
            NTriggeredTaus        = cms.untracked.vuint32(1,1,1,1,1,1,1), #The first one is for the ref events
            NTriggeredLeptons     = cms.untracked.vuint32(0,0,0,0,0,0,0), #the first one is for the ref events
            TauType               = cms.untracked.vint32(-86,84,84,84,84,84),
            LeptonType            = cms.untracked.vint32(0,0,0,0,0,0)                         
        ),
        cms.PSet(
            ConfigType            = cms.untracked.string("Path"),
            triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryRAW","",hltTauDQMProcess),
            DQMFolder             = cms.untracked.string('HLT/TauOnline/Electrons/LoosePFTau'),
            Filter                = cms.untracked.VInputTag(
                                        cms.InputTag("hltPFTau20","",hltTauDQMProcess),
                                        cms.InputTag("hltPFTau20Track","",hltTauDQMProcess),
                                        cms.InputTag("hltPFTau20TrackLooseIso","",hltTauDQMProcess)
                                	),
            MatchDeltaR           = cms.untracked.vdouble(0.2,0.2,0.2),    #One per filter
            NTriggeredTaus        = cms.untracked.vuint32(1,1,1,1), #The first one is for the ref events
            NTriggeredLeptons     = cms.untracked.vuint32(0,0,0,0), #the first one is for the ref events
            TauType               = cms.untracked.vint32(84,84,84),
            LeptonType            = cms.untracked.vint32(0,0,0)                         
        ),
        cms.PSet(
            ConfigType            = cms.untracked.string("Path"),
            triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryRAW","",hltTauDQMProcess),
            DQMFolder             = cms.untracked.string('HLT/TauOnline/Electrons/MediumPFTau'),
            Filter                = cms.untracked.VInputTag(
                                        cms.InputTag("hltPFTauMediumIso20","",hltTauDQMProcess),
                                        cms.InputTag("hltPFTauMediumIso20Track","",hltTauDQMProcess),
                                        cms.InputTag("hltPFTauMediumIso20TrackMediumIso","",hltTauDQMProcess)
                                	),
            MatchDeltaR           = cms.untracked.vdouble(0.2,0.2,0.2),    #One per filter
            NTriggeredTaus        = cms.untracked.vuint32(1,1,1,1), #The first one is for the ref events
            NTriggeredLeptons     = cms.untracked.vuint32(0,0,0,0), #the first one is for the ref events
            TauType               = cms.untracked.vint32(84,84,84),
            LeptonType            = cms.untracked.vint32(0,0,0)                         
        ),

        cms.PSet(
            ConfigType            = cms.untracked.string("Path"),
            triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryRAW","",hltTauDQMProcess),
            DQMFolder             = cms.untracked.string('HLT/TauOnline/Electrons/TightPFTau'),
            Filter                = cms.untracked.VInputTag(
                                        cms.InputTag("hltPFTauTightIso20","",hltTauDQMProcess),
                                        cms.InputTag("hltPFTauTightIso20Track","",hltTauDQMProcess),
                                        cms.InputTag("hltPFTauTightIso20TrackTightIso","",hltTauDQMProcess)
                                	),
            MatchDeltaR           = cms.untracked.vdouble(0.2,0.2,0.2),    #One per filter
            NTriggeredTaus        = cms.untracked.vuint32(1,1,1,1), #The first one is for the ref events
            NTriggeredLeptons     = cms.untracked.vuint32(0,0,0,0), #the first one is for the ref events
            TauType               = cms.untracked.vint32(84,84,84),
            LeptonType            = cms.untracked.vint32(0,0,0)                         
        ),        
        cms.PSet(
            ConfigType            = cms.untracked.string("LitePath"),
            triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryAOD","",hltTauDQMProcess),
            DQMFolder             = cms.untracked.string('HLT/TauOnline/Electrons/Summary'),
            Filter                = cms.untracked.VInputTag(
                                        cms.InputTag("hltFilterDoubleIsoPFTau45Trk5LeadTrack5IsolationL1HLTMatched","",hltTauDQMProcess),
                                        cms.InputTag("hltFilterSingleIsoPFTau35Trk20MET70LeadTrack20IsolationL1HLTMatched","",hltTauDQMProcess)
                                    ),
            PathName              = cms.untracked.vstring(
                                        'DoubleIsoTau',
                                        'SingleTau'
                                    ),
            NTriggeredTaus        = cms.untracked.vuint32(2,1), 
            NTriggeredLeptons     = cms.untracked.vuint32(0,0), 
            TauType               = cms.untracked.vint32(15,15),
            LeptonType            = cms.untracked.vint32(0,0)
        ),
        cms.PSet(
            ConfigType             = cms.untracked.string("L1"),
            DQMFolder              = cms.string('HLT/TauOnline/Electrons/L1'),
            L1Taus                 = cms.InputTag("hltL1extraParticles","Tau"),
            L1Jets                 = cms.InputTag("hltL1extraParticles","Central"),
            L1Electrons            = cms.InputTag("hltL1extraParticles","Isolated"),
            L1Muons                = cms.InputTag("hltL1extraParticles"),
        ),
        cms.PSet(
            ConfigType              = cms.untracked.string("Summary"),
            L1Dirs                  = cms.vstring("HLT/TauOnline/Electrons/L1"),
            caloDirs                = cms.vstring(""),
            trackDirs               = cms.vstring(""),
            pathDirs                = cms.vstring("HLT/TauOnline/Electrons/DoubleTau","HLT/TauOnline/Electrons/SingleTau","HLT/TauOnline/Electrons/LoosePFTau","HLT/TauOnline/Electrons/MediumPFTau","HLT/TauOnline/Electrons/TightPFTau"),
            pathSummaryDirs         = cms.vstring("HLT/TauOnline/Electrons/Summary")
        ),
    
    ),
    
    doMatching = cms.bool(True),
    matchFilter = cms.untracked.VInputTag(cms.InputTag("hltMu8Ele17CaloIdTCaloIsoVLPixelMatchFilter","",hltTauDQMProcess) ),
    matchObjectID = cms.untracked.vint32(11),
    matchObjectMinPt = cms.untracked.vdouble(10),
    TriggerEvent = cms.InputTag("hltTriggerSummaryAOD","",hltTauDQMProcess )                          
)

hltMonTauReco =cms.Sequence(hltTauMonitor+hltTauElectronMonitor)
