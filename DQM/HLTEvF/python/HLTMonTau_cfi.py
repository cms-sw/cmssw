import FWCore.ParameterSet.Config as cms

hltTauDQMProcess = "HLT"

hltTauMonitor = cms.EDAnalyzer("HLTTauDQMSource",
    MonitorSetup = cms.VPSet(
        cms.PSet(
            triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryRAW","",hltTauDQMProcess),
            DQMFolder             = cms.untracked.string('HLT/TauOnline/Inclusive/DoubleTau'),
            Filter                = cms.untracked.VInputTag(
                                        cms.InputTag("hltL1sDoubleLooseIsoTau15","",hltTauDQMProcess), 
                                        cms.InputTag("hltFilterL2EtCutDoubleLooseIsoTau15","",hltTauDQMProcess),
                                        cms.InputTag("hltFilterL2EcalIsolationDoubleLooseIsoTau15","",hltTauDQMProcess),
                                        ),
            MatchDeltaR           = cms.untracked.vdouble(0.5,0.2,0.2),    #One per filter
            NTriggeredTaus        = cms.untracked.vuint32(2,2,2,2,2), #The first one is for the ref events
            NTriggeredLeptons     = cms.untracked.vuint32(0,0,0,0,0), #the first one is for the ref events
            TauType               = cms.untracked.vint32(-86,84,84,84),
            LeptonType            = cms.untracked.vint32(0,0,0,0)                            
        ),

        cms.PSet(
            triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryAOD","",hltTauDQMProcess),
            DQMFolder             = cms.untracked.string('HLT/TauOnline/Inclusive/Summary'),
            Filter                = cms.untracked.VInputTag(
                                        cms.InputTag("hltFilterL2EcalIsolationDoubleLooseIsoTau15","",hltTauDQMProcess),
                                        cms.InputTag("hltFilterL2EcalIsolationSingleLooseIsoTau20","",hltTauDQMProcess)
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
                                        cms.InputTag("hltL1sSingleLooseIsoTau20","",hltTauDQMProcess), 
                                        cms.InputTag("hltFilterL2EtCutSingleLooseIsoTau20","",hltTauDQMProcess),
                                        cms.InputTag("hltFilterL2EcalIsolationSingleLooseIsoTau20","",hltTauDQMProcess)
                                        ),
            MatchDeltaR           = cms.untracked.vdouble(0.5,0.2,0.2,0.2),    #One per filter
            NTriggeredTaus        = cms.untracked.vuint32(1,1,1,1,1), #The first one is for the ref events
            NTriggeredLeptons     = cms.untracked.vuint32(0,0,0,0,0), #the first one is for the ref events
            TauType               = cms.untracked.vint32(-86,84,84,84),
            LeptonType            = cms.untracked.vint32(0,0,0,0)                            

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
            L1Dirs                  = cms.vstring("HLT/TauOnline/Inclusive/L1"),
            caloDirs                = cms.vstring("HLT/TauOnline/Inclusive/L2"),
            trackDirs               = cms.vstring(""),
            pathDirs                = cms.vstring(""),
            pathSummaryDirs         = cms.vstring("")
        )
        
    ),


    ConfigType = cms.vstring(
        "Path",
        "LitePath",
        "Path",
        "L1",
        "Calo",
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
            DQMFolder             = cms.untracked.string('HLT/TauOnline/Photons/DoubleTau'),
            Filter                = cms.untracked.VInputTag(
                                        cms.InputTag("hltL1sDoubleLooseIsoTau15","",hltTauDQMProcess), 
                                        cms.InputTag("hltFilterL2EtCutDoubleLooseIsoTau15","",hltTauDQMProcess),
                                        cms.InputTag("hltFilterL2EcalIsolationDoubleLooseIsoTau15","",hltTauDQMProcess)
                                        ),
            MatchDeltaR           = cms.untracked.vdouble(0.5,0.2,0.2,0.2),    #One per filter
            NTriggeredTaus        = cms.untracked.vuint32(2,2,2,2,2), #The first one is for the ref events
            NTriggeredLeptons     = cms.untracked.vuint32(0,0,0,0,0), #the first one is for the ref events
            TauType               = cms.untracked.vint32(-86,84,84,84),
            LeptonType            = cms.untracked.vint32(0,0,0,0)                            
        ),

        cms.PSet(
            triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryAOD","",hltTauDQMProcess),
            DQMFolder             = cms.untracked.string('HLT/TauOnline/Photons/Summary'),
            Filter                = cms.untracked.VInputTag(
                                        cms.InputTag("hltFilterL2EcalIsolationDoubleLooseIsoTau15","",hltTauDQMProcess),
                                        cms.InputTag("hltFilterL2EcalIsolationSingleLooseIsoTau20","",hltTauDQMProcess)
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
            DQMFolder             = cms.untracked.string('HLT/TauOnline/Photons/SingleTau'),
            Filter                = cms.untracked.VInputTag(
                                        cms.InputTag("hltL1sSingleLooseIsoTau20","",hltTauDQMProcess), 
                                        cms.InputTag("hltFilterL2EtCutSingleLooseIsoTau20","",hltTauDQMProcess),
                                        cms.InputTag("hltFilterL2EcalIsolationSingleLooseIsoTau20","",hltTauDQMProcess)
                                        ),
            MatchDeltaR           = cms.untracked.vdouble(0.5,0.2,0.2,0.2),    #One per filter
            NTriggeredTaus        = cms.untracked.vuint32(1,1,1,1,1), #The first one is for the ref events
            NTriggeredLeptons     = cms.untracked.vuint32(0,0,0,0,0), #the first one is for the ref events
            TauType               = cms.untracked.vint32(-86,84,84,84),
            LeptonType            = cms.untracked.vint32(0,0,0,0)                            

        ),
        cms.PSet(
            DQMFolder              = cms.string('HLT/TauOnline/Photons/L1'),
            L1Taus                 = cms.InputTag("hltL1extraParticles","Tau"),
            L1Jets                 = cms.InputTag("hltL1extraParticles","Central"),
            L1Electrons            = cms.InputTag("hltL1extraParticles","Isolated"),
            L1Muons                = cms.InputTag("hltL1extraParticles"),
        ),

        cms.PSet(
            DQMFolder              = cms.string('HLT/TauOnline/Photons/L2'),
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
            L1Dirs                  = cms.vstring("HLT/TauOnline/Photons/L1"),
            caloDirs                = cms.vstring("HLT/TauOnline/Photons/L2"),
            trackDirs               = cms.vstring(),
            pathDirs                = cms.vstring("HLT/TauOnline/Photons/DoubleTau","HLT/TauOnline/Photons/SingleTau"),
            pathSummaryDirs         = cms.vstring("HLT/TauOnline/Photons/Summary")
        )
    ),
                                     

    ConfigType = cms.vstring(
        "Path",
        "LitePath",
        "Path",
        "L1",
        "Calo",
        "Summary"
    ),
    
   doMatching = cms.bool(True),
   matchFilter       = cms.untracked.VInputTag(cms.InputTag("hltL1NonIsoHLTNonIsoDoublePhotonEt10HcalIsolFilter","",hltTauDQMProcess) ),
   matchObjectID       = cms.untracked.vint32(0),
   matchObjectMinPt    = cms.untracked.vdouble(10),
   TriggerEvent = cms.InputTag("hltTriggerSummaryAOD","",hltTauDQMProcess )                          
)

hltMonTauReco =cms.Sequence(hltTauMonitor+hltTauElectronMonitor)




