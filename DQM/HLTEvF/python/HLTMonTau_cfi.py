import FWCore.ParameterSet.Config as cms

hltTauDQMProcess = "HLT"

hltTauMonitor = cms.EDAnalyzer("HLTTauDQMSource",
    MonitorSetup = cms.VPSet(
        cms.PSet(
            triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryRAW","",hltTauDQMProcess),
            DQMFolder             = cms.untracked.string('HLT/TauOnline/Inclusive/DoubleTau'),
            Filter                = cms.untracked.VInputTag(
                                        cms.InputTag("hltL1sDoubleIsoTau15Trk5","",hltTauDQMProcess), 
                                        cms.InputTag("hltFilterL2EtCutDoubleIsoTau15Trk5","",hltTauDQMProcess),
                                        cms.InputTag("hltFilterL2EcalIsolationDoubleIsoTau15Trk5","",hltTauDQMProcess),
                                        cms.InputTag("hltFilterL25LeadingTrackPtCutDoubleIsoTau15Trk5","",hltTauDQMProcess),
                                        cms.InputTag("hltFilterL3TrackIsolationDoubleIsoTau15Trk5","",hltTauDQMProcess),
                                        ),
            MatchDeltaR           = cms.untracked.vdouble(0.5,0.2,0.2,0.2,0.2),    #One per filter
            NTriggeredTaus        = cms.untracked.vuint32(2,2,2,2,2,2,2), #The first one is for the ref events
            NTriggeredLeptons     = cms.untracked.vuint32(0,0,0,0,0,0,0), #the first one is for the ref events
            TauType               = cms.untracked.vint32(-86,84,84,84,84,84),
            LeptonType            = cms.untracked.vint32(0,0,0,0,0,0)                            
        ),

        cms.PSet(
            triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryAOD","",hltTauDQMProcess),
            DQMFolder             = cms.untracked.string('HLT/TauOnline/Inclusive/Summary'),
            Filter                = cms.untracked.VInputTag(
                                        cms.InputTag("hltFilterL3TrackIsolationDoubleIsoTau15Trk5","",hltTauDQMProcess),
                                        cms.InputTag("hltFilterL3TrackIsolationSingleIsoTau30Trk5MET20","",hltTauDQMProcess)
                                        ),
            PathName              = cms.untracked.vstring(
                                        'DoubleLooseIsoTau',
                                        'SingleLooseIsoTau'
                                        ),
            NTriggeredTaus        = cms.untracked.vuint32(2,1), 
            NTriggeredLeptons     = cms.untracked.vuint32(0,0), 
            TauType               = cms.untracked.vint32(15,15),
            LeptonType            = cms.untracked.vint32(0,0)                            
        ),
        
        cms.PSet(
            triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryRAW","",hltTauDQMProcess),
            DQMFolder             = cms.untracked.string('HLT/TauOnline/Inclusive/SingleTau'),
            Filter                = cms.untracked.VInputTag(
                                        cms.InputTag("hltL1sSingleIsoTau20Trk15MET25","",hltTauDQMProcess), 
                                        cms.InputTag("hltFilterL2EtCutSingleIsoTau20Trk15MET25","",hltTauDQMProcess),
                                        cms.InputTag("hltFilterL2EcalIsolationSingleIsoTau20Trk15MET25","",hltTauDQMProcess),
                                        cms.InputTag("hltFilterL25LeadingTrackPtCutSingleIsoTau20Trk15MET25","",hltTauDQMProcess),
                                        cms.InputTag("hltFilterL3TrackIsolationSingleIsoTau20Trk15MET25","",hltTauDQMProcess)
                                        ),
            MatchDeltaR           = cms.untracked.vdouble(0.5,0.2,0.2,0.2,0.2),    #One per filter
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

        cms.PSet(
            DQMFolder              = cms.string('HLT/TauOnline/Inclusive/L2'),
            L2RegionalJets         = cms.VInputTag(
                                                   cms.InputTag("hltIconeTau1Regional"),
                                                   cms.InputTag("hltIconeTau2Regional"),
                                                   cms.InputTag("hltIconeTau3Regional"),
                                                   cms.InputTag("hltIconeTau4Regional"),
                                                   cms.InputTag("hltIconeCentral1Regional"),
                                                   cms.InputTag("hltIconeCentral2Regional"),
                                                   cms.InputTag("hltIconeCentral3Regional"),
                                                   cms.InputTag("hltIconeCentral4Regional")),
            L2InfoAssociationInput = cms.InputTag("hltL2TauNarrowConeIsolationProducer"),
            L2IsolatedJets         = cms.InputTag("hltL2TauRelaxingIsolationSelector","Isolated")
        ),
        
        cms.PSet(
            DQMFolder              = cms.string('HLT/TauOnline/Inclusive/L25'),
            ConeIsolation          = cms.InputTag("hltL25TauConeIsolation"),
            IsolatedJets           = cms.InputTag("hltL25TauLeadingTrackPtCutSelector"),
            Type                   = cms.string('L25')
        ),
        cms.PSet(
            DQMFolder              = cms.string('HLT/TauOnline/Inclusive/L3'),
            ConeIsolation          = cms.InputTag("hltL3TauConeIsolation"),
            IsolatedJets           = cms.InputTag("hltL3TauIsolationSelector"),
            Type                   = cms.string('L3')
        ) ,           

        cms.PSet(
            L1Dirs                  = cms.vstring("HLT/TauOnline/Inclusive/L1"),
            caloDirs                = cms.vstring("HLT/TauOnline/Inclusive/L2"),
            trackDirs               = cms.vstring("HLT/TauOnline/Inclusive/L25","HLT/TauOnline/Inclusive/L3"),
            pathDirs                = cms.vstring(""),
            pathSummaryDirs         = cms.vstring("")
        ),
    
    ),


    ConfigType = cms.vstring(
        "Path",
        "LitePath",
        "Path",
        "L1",
        "Calo",
        "Track",
        "Track",
        "Summary"
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
                                        cms.InputTag("hltL1sDoubleIsoTau15Trk5","",hltTauDQMProcess), 
                                        cms.InputTag("hltFilterL2EtCutDoubleIsoTau15Trk5","",hltTauDQMProcess),
                                        cms.InputTag("hltFilterL2EcalIsolationDoubleIsoTau15Trk5","",hltTauDQMProcess),
                                        cms.InputTag("hltFilterL25LeadingTrackPtCutDoubleIsoTau15Trk5","",hltTauDQMProcess),
                                        cms.InputTag("hltFilterL3TrackIsolationDoubleIsoTau15Trk5","",hltTauDQMProcess),
                                        ),
            MatchDeltaR           = cms.untracked.vdouble(0.5,0.2,0.2,0.2,0.2),    #One per filter
            NTriggeredTaus        = cms.untracked.vuint32(2,2,2,2,2,2,2), #The first one is for the ref events
            NTriggeredLeptons     = cms.untracked.vuint32(0,0,0,0,0,0,0), #the first one is for the ref events
            TauType               = cms.untracked.vint32(-86,84,84,84,84,84),
            LeptonType            = cms.untracked.vint32(0,0,0,0,0,0)                            
        ),

        cms.PSet(
            triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryAOD","",hltTauDQMProcess),
            DQMFolder             = cms.untracked.string('HLT/TauOnline/Electrons/Summary'),
            Filter                = cms.untracked.VInputTag(
                                        cms.InputTag("hltFilterL3TrackIsolationDoubleIsoTau15Trk5","",hltTauDQMProcess),
                                        cms.InputTag("hltFilterL3TrackIsolationSingleIsoTau30Trk5MET20","",hltTauDQMProcess)
                                       ),
            PathName              = cms.untracked.vstring(
                                        'DoubleLooseIsoTau',
                                        'SingleLooseIsoTau'
                                        ),
            NTriggeredTaus        = cms.untracked.vuint32(2,1), 
            NTriggeredLeptons     = cms.untracked.vuint32(0,0), 
            TauType               = cms.untracked.vint32(15,15),
            LeptonType            = cms.untracked.vint32(0,0)                            
        ),
        
        cms.PSet(
            triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryRAW","",hltTauDQMProcess),
            DQMFolder             = cms.untracked.string('HLT/TauOnline/Electrons/SingleTau'),
            Filter                = cms.untracked.VInputTag(
                                        cms.InputTag("hltL1sSingleIsoTau20Trk15MET25","",hltTauDQMProcess), 
                                        cms.InputTag("hltFilterL2EtCutSingleIsoTau20Trk15MET25","",hltTauDQMProcess),
                                        cms.InputTag("hltFilterL2EcalIsolationSingleIsoTau20Trk15MET25","",hltTauDQMProcess),
                                        cms.InputTag("hltFilterL25LeadingTrackPtCutSingleIsoTau20Trk15MET25","",hltTauDQMProcess),
                                        cms.InputTag("hltFilterL3TrackIsolationSingleIsoTau20Trk15MET25","",hltTauDQMProcess)
                                        ),
            MatchDeltaR           = cms.untracked.vdouble(0.5,0.2,0.2,0.2,0.2),    #One per filter
            NTriggeredTaus        = cms.untracked.vuint32(1,1,1,1,1,1,1), #The first one is for the ref events
            NTriggeredLeptons     = cms.untracked.vuint32(0,0,0,0,0,0,0), #the first one is for the ref events
            TauType               = cms.untracked.vint32(-86,84,84,84,84,84),
            LeptonType            = cms.untracked.vint32(0,0,0,0,0,0)                            

        ),
        cms.PSet(
            DQMFolder              = cms.string('HLT/TauOnline/Electrons/L1'),
            L1Taus                 = cms.InputTag("hltL1extraParticles","Tau"),
            L1Jets                 = cms.InputTag("hltL1extraParticles","Central"),
            L1Electrons            = cms.InputTag("hltL1extraParticles","Isolated"),
            L1Muons                = cms.InputTag("hltL1extraParticles"),
        ),

        cms.PSet(
            DQMFolder              = cms.string('HLT/TauOnline/Electrons/L2'),
            L2InfoAssociationInput = cms.InputTag("hltL2TauNarrowConeIsolationProducer"),
            L2RegionalJets         = cms.VInputTag(
                                                   cms.InputTag("hltIconeTau1Regional"),
                                                   cms.InputTag("hltIconeTau2Regional"),
                                                   cms.InputTag("hltIconeTau3Regional"),
                                                   cms.InputTag("hltIconeTau4Regional"),
                                                   cms.InputTag("hltIconeCentral1Regional"),
                                                   cms.InputTag("hltIconeCentral2Regional"),
                                                   cms.InputTag("hltIconeCentral3Regional"),
                                                   cms.InputTag("hltIconeCentral4Regional")),
            L2IsolatedJets         = cms.InputTag("hltL2TauRelaxingIsolationSelector","Isolated")
        ),
        cms.PSet(
            DQMFolder              = cms.string('HLT/TauOnline/Electrons/L25'),
            ConeIsolation          = cms.InputTag("hltL25TauConeIsolation"),
            IsolatedJets           = cms.InputTag("hltL25TauLeadingTrackPtCutSelector"),
            Type                   = cms.string('L25')
        ),
        cms.PSet(
            DQMFolder              = cms.string('HLT/TauOnline/Electrons/L3'),
            ConeIsolation          = cms.InputTag("hltL3TauConeIsolation"),
            IsolatedJets           = cms.InputTag("hltL3TauIsolationSelector"),
            Type                   = cms.string('L3')
        ) ,           
        cms.PSet(
            L1Dirs                  = cms.vstring("HLT/TauOnline/Electrons/L1"),
            caloDirs                = cms.vstring("HLT/TauOnline/Electrons/L2"),
            trackDirs               = cms.vstring("HLT/TauOnline/Electrons/L25","HLT/TauOnline/Electrons/L3"),
            pathDirs                = cms.vstring("HLT/TauOnline/Electrons/DoubleTau","HLT/TauOnline/Electrons/SingleTau"),
            pathSummaryDirs         = cms.vstring("HLT/TauOnline/Electrons/Summary")
        ),
    
    ),
                                     

    ConfigType = cms.vstring(
        "Path",
        "LitePath",
        "Path",
        "L1",
        "Calo",
        "Track",
        "Track",        
        "Summary"
    ),
    
   doMatching = cms.bool(True),
   matchFilter       = cms.untracked.VInputTag(cms.InputTag("hltL1NonIsoHLTNonIsoSingleElectronEt22TighterCaloIdIsolTrackIsolFilter","",hltTauDQMProcess) ),
   matchObjectID       = cms.untracked.vint32(11),
   matchObjectMinPt    = cms.untracked.vdouble(10),
   TriggerEvent = cms.InputTag("hltTriggerSummaryAOD","",hltTauDQMProcess )                          
)

hltMonTauReco =cms.Sequence(hltTauMonitor+hltTauElectronMonitor)




