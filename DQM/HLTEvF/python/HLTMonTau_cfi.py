import FWCore.ParameterSet.Config as cms

hltTauDQMProcess = "HLT"

hltTauMonitor = cms.EDAnalyzer("HLTTauDQMSource",

    MonitorSetup = cms.VPSet(
        cms.PSet(
            triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryRAW","",hltTauDQMProcess),
            DQMFolder             = cms.untracked.string('HLT/TauOnline/Inclusive/DoubleTau'),
            Filter                = cms.untracked.VInputTag(
                                        cms.InputTag("hltL1sDoubleIsoTau20Trk5","",hltTauDQMProcess), 
                                        cms.InputTag("hltFilterL2EtCutDoublePFIsoTau20Trk5","",hltTauDQMProcess),
                                        cms.InputTag("hltDoublePFTauTightIso20Track","",hltTauDQMProcess),
                                        cms.InputTag("hltDoublePFTauTightIso20Track5","",hltTauDQMProcess),
                                        cms.InputTag("hltFilterDoubleIsoPFTau20Trk5LeadTrack5IsolationL1HLTMatched","",hltTauDQMProcess)
                                        ),
            MatchDeltaR           = cms.untracked.vdouble(0.5,0.2,0.2,0.2,0.2),    #One per filter
            NTriggeredTaus        = cms.untracked.vuint32(2,2,2,2,2,2), #The first one is for the ref events
            NTriggeredLeptons     = cms.untracked.vuint32(0,0,0,0,0,0), #the first one is for the ref events
            TauType               = cms.untracked.vint32(-86,84,84,84,84),
            LeptonType            = cms.untracked.vint32(0,0,0,0,0)                            
        ),
        
        cms.PSet(
            triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryRAW","",hltTauDQMProcess),
            DQMFolder             = cms.untracked.string('HLT/TauOnline/Inclusive/EleTau'),
            Filter                = cms.untracked.VInputTag(
                                        cms.InputTag("hltL1sL1SingleEG12","",hltTauDQMProcess), 
                                        cms.InputTag("hltEle15CaloIdVTTrkIdTCaloIsoTTrkIsoTTrackIsolFilter","",hltTauDQMProcess),
                                        cms.InputTag("hltPFTau15","",hltTauDQMProcess),
                                        cms.InputTag("hltPFTau15Track","",hltTauDQMProcess),
                                        cms.InputTag("hltPFTau15TrackLooseIso","",hltTauDQMProcess),
                                        cms.InputTag("hltOverlapFilterIsoEle15IsoPFTau15","",hltTauDQMProcess)
                                        ),
            MatchDeltaR           = cms.untracked.vdouble(0.5,0.2,0.2,0.2,0.2,0.2),    #One per filter
            NTriggeredTaus        = cms.untracked.vuint32(1,0,0,1,1,1,1), #The first one is for the ref events
            NTriggeredLeptons     = cms.untracked.vuint32(1,1,1,0,0,0,1), #the first one is for the ref events
            TauType               = cms.untracked.vint32(0,0,84,84,84,84),
            LeptonType            = cms.untracked.vint32(-83,82,0,0,0,82)                            
        ),
        
        cms.PSet(
            triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryRAW","",hltTauDQMProcess),
            DQMFolder             = cms.untracked.string('HLT/TauOnline/Inclusive/MuTau'),
            Filter                = cms.untracked.VInputTag(
                                        cms.InputTag("hltL1sL1SingleMu10","",hltTauDQMProcess), 
                                        cms.InputTag("hltL3Muon15","",hltTauDQMProcess),
                                        cms.InputTag("hltPFTau20Track","",hltTauDQMProcess),
                                        cms.InputTag("hltPFTau20TrackLooseIso","",hltTauDQMProcess),
                                        cms.InputTag("hltOverlapFilterMu15IsoPFTau20","",hltTauDQMProcess)
                                        ),
            MatchDeltaR           = cms.untracked.vdouble(0.5,0.2,0.2,0.2,0.2),    #One per filter
            NTriggeredTaus        = cms.untracked.vuint32(1,0,0,1,1,1), #The first one is for the ref events
            NTriggeredLeptons     = cms.untracked.vuint32(1,1,1,0,0,1), #the first one is for the ref events
            TauType               = cms.untracked.vint32(0,0,84,84,84),
            LeptonType            = cms.untracked.vint32(-81,83,0,0,83)                            
        ),
        
        cms.PSet(
            triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryAOD","",hltTauDQMProcess),
            DQMFolder             = cms.untracked.string('HLT/TauOnline/Inclusive/Summary'),
            Filter                = cms.untracked.VInputTag(
                                        cms.InputTag("hltFilterDoubleIsoPFTau20Trk5LeadTrack5IsolationL1HLTMatched","",hltTauDQMProcess),
                                        cms.InputTag("hltOverlapFilterIsoEle15IsoPFTau15","",hltTauDQMProcess),
                                        cms.InputTag("hltOverlapFilterMu15IsoPFTau20","",hltTauDQMProcess),
                                        cms.InputTag("hltFilterSingleIsoPFTau35Trk20LeadTrackPt20","",hltTauDQMProcess)
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
        
        cms.PSet(
            triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryRAW","",hltTauDQMProcess),
            DQMFolder             = cms.untracked.string('HLT/TauOnline/Inclusive/SingleTau'),
            Filter                = cms.untracked.VInputTag(
                                        cms.InputTag("hltL1sSingleIsoTau35Trk20MET45","",hltTauDQMProcess), 
                                        cms.InputTag("hltFilterL2EtCutSingleIsoPFTau35Trk20MET45","",hltTauDQMProcess),
                                        cms.InputTag("hltPFTauTightIso35","",hltTauDQMProcess),
                                        cms.InputTag("hltPFTauTightIso35Track","",hltTauDQMProcess),
                                        cms.InputTag("hltFilterSingleIsoPFTau35Trk20LeadTrackPt20","",hltTauDQMProcess),
                                        cms.InputTag("hltFilterSingleIsoPFTau35Trk20MET45LeadTrack20MET45IsolationL1HLTMatched","",hltTauDQMProcess)
                                        ),
            MatchDeltaR           = cms.untracked.vdouble(0.5,0.2,0.2,0.2,0.2,0.2),    #One per filter
            NTriggeredTaus        = cms.untracked.vuint32(1,1,1,1,1,1,1), #The first one is for the ref events
            NTriggeredLeptons     = cms.untracked.vuint32(0,0,0,0,0,0,0), #the first one is for the ref events
            TauType               = cms.untracked.vint32(-86,84,84,84,84,84),
            LeptonType            = cms.untracked.vint32(0,0,0,0,0,0)                         
        ),
        
        cms.PSet(
            DQMFolder              = cms.string('HLT/TauOnline/Inclusive/L1'),
            L1Taus                 = cms.InputTag("hltL1extraParticles","Tau"),
            L1Jets                 = cms.InputTag("hltL1extraParticles","Central"),
            L1Electrons            = cms.InputTag("hltL1extraParticles","Isolated"),
            L1Muons                = cms.InputTag("hltL1extraParticles"),
        ),

    
    ),


    ConfigType = cms.vstring(
        "Path",
        "Path",
        "Path",
        "LitePath",
        "Path",
        "L1"

    ),
    
   doMatching = cms.bool(False),

   matchFilter         = cms.untracked.VInputTag(cms.InputTag("","",hltTauDQMProcess) ),
   matchObjectID       = cms.untracked.vint32(0),
   matchObjectMinPt    = cms.untracked.vdouble(15),
   TriggerEvent = cms.InputTag("hltTriggerSummaryAOD","",hltTauDQMProcess )                          
)




hltTauElectronMonitor = cms.EDAnalyzer("HLTTauDQMSource",
    MonitorSetup = cms.VPSet(
        cms.PSet(
            triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryRAW","",hltTauDQMProcess),
            DQMFolder             = cms.untracked.string('HLT/TauOnline/Electrons/DoubleTau'),
            Filter                = cms.untracked.VInputTag(
                                        cms.InputTag("hltL1sDoubleIsoTau20Trk5","",hltTauDQMProcess), 
                                        cms.InputTag("hltFilterL2EtCutDoublePFIsoTau20Trk5","",hltTauDQMProcess),
                                        cms.InputTag("hltDoublePFTauTightIso20Track","",hltTauDQMProcess),
                                        cms.InputTag("hltDoublePFTauTightIso20Track5","",hltTauDQMProcess),
                                        cms.InputTag("hltFilterDoubleIsoPFTau20Trk5LeadTrack5IsolationL1HLTMatched","",hltTauDQMProcess)
                                        ),
            MatchDeltaR           = cms.untracked.vdouble(0.5,0.2,0.2,0.2,0.2),    #One per filter
            NTriggeredTaus        = cms.untracked.vuint32(2,2,2,2,2,2), #The first one is for the ref events
            NTriggeredLeptons     = cms.untracked.vuint32(0,0,0,0,0,0), #the first one is for the ref events
            TauType               = cms.untracked.vint32(-86,84,84,84,84),
            LeptonType            = cms.untracked.vint32(0,0,0,0,0)                            
        ),
        
        cms.PSet(
            triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryRAW","",hltTauDQMProcess),
            DQMFolder             = cms.untracked.string('HLT/TauOnline/Electrons/SingleTau'),
            Filter                = cms.untracked.VInputTag(
                                        cms.InputTag("hltL1sSingleIsoTau35Trk20MET45","",hltTauDQMProcess), 
                                        cms.InputTag("hltFilterL2EtCutSingleIsoPFTau35Trk20MET45","",hltTauDQMProcess),
                                        cms.InputTag("hltPFTauTightIso35","",hltTauDQMProcess),
                                        cms.InputTag("hltPFTauTightIso35Track","",hltTauDQMProcess),
                                        cms.InputTag("hltFilterSingleIsoPFTau35Trk20LeadTrackPt20","",hltTauDQMProcess),
                                        cms.InputTag("hltFilterSingleIsoPFTau35Trk20MET45LeadTrack20MET45IsolationL1HLTMatched","",hltTauDQMProcess)
                                        ),
            MatchDeltaR           = cms.untracked.vdouble(0.5,0.2,0.2,0.2,0.2,0.2),    #One per filter
            NTriggeredTaus        = cms.untracked.vuint32(1,1,1,1,1,1,1), #The first one is for the ref events
            NTriggeredLeptons     = cms.untracked.vuint32(0,0,0,0,0,0,0), #the first one is for the ref events
            TauType               = cms.untracked.vint32(-86,84,84,84,84,84),
            LeptonType            = cms.untracked.vint32(0,0,0,0,0,0)                         
        ),
        
        cms.PSet(
            triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryAOD","",hltTauDQMProcess),
            DQMFolder             = cms.untracked.string('HLT/TauOnline/Electrons/Summary'),
            Filter                = cms.untracked.VInputTag(
                                        cms.InputTag("hltFilterDoubleIsoPFTau20Trk5LeadTrack5IsolationL1HLTMatched","",hltTauDQMProcess),
                                        cms.InputTag("hltFilterSingleIsoPFTau35Trk20MET45LeadTrack20MET45IsolationL1HLTMatched","",hltTauDQMProcess)
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
            DQMFolder              = cms.string('HLT/TauOnline/Electrons/L1'),
            L1Taus                 = cms.InputTag("hltL1extraParticles","Tau"),
            L1Jets                 = cms.InputTag("hltL1extraParticles","Central"),
            L1Electrons            = cms.InputTag("hltL1extraParticles","Isolated"),
            L1Muons                = cms.InputTag("hltL1extraParticles"),
        ),
        cms.PSet(
            L1Dirs                  = cms.vstring("HLT/TauOnline/Electrons/L1"),
            caloDirs                = cms.vstring(""),
            trackDirs               = cms.vstring(""),
            pathDirs                = cms.vstring("HLT/TauOnline/Electrons/DoubleTau","HLT/TauOnline/Electrons/SingleTau"),
            pathSummaryDirs         = cms.vstring("HLT/TauOnline/Electrons/Summary")
        ),
    
    ),
                                     

    ConfigType = cms.vstring(
        "Path",
        "Path",
        "LitePath",
        "L1",
        "Summary"
    ),
    
   doMatching = cms.bool(True),
   matchFilter       = cms.untracked.VInputTag(cms.InputTag("hltEle17CaloIdVTCaloIsoVTTrkIdTTrkIsoVTSC8TrackIsolFilter","",hltTauDQMProcess) ),
   matchObjectID       = cms.untracked.vint32(11),
   matchObjectMinPt    = cms.untracked.vdouble(10),
   TriggerEvent = cms.InputTag("hltTriggerSummaryAOD","",hltTauDQMProcess )                          
)

hltMonTauReco =cms.Sequence(hltTauMonitor+hltTauElectronMonitor)




