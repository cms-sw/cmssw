import FWCore.ParameterSet.Config as cms

hltTauDQMProcess = "HLT"


hltTauMonitor = cms.EDFilter("HLTTauDQMSource",
    MonitorSetup = cms.VPSet(
        cms.PSet(
            triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryRAW","",hltTauDQMProcess),
            DQMFolder             = cms.untracked.string('HLT/TauOnline/Inclusive/DoubleTau'),
            Filter                = cms.untracked.VInputTag(
                                        cms.InputTag("hltL1sDoubleLooseIsoTau15","",hltTauDQMProcess), 
                                        cms.InputTag("hltFilterL2EtCutDoubleLooseIsoTau15","",hltTauDQMProcess),
                                        cms.InputTag("hltFilterL2EcalIsolationDoubleLooseIsoTau15","",hltTauDQMProcess),
                                        ),
            MatchDeltaR           = cms.untracked.vdouble(0.5,0.3,0.3),    #One per filter
            NTriggeredTaus        = cms.untracked.vuint32(2,2,2,2), #The first one is for the ref events
            NTriggeredLeptons     = cms.untracked.vuint32(0,0,0,0,0), #the first one is for the ref events
            TauType               = cms.untracked.vint32(86,94,94),
            LeptonType            = cms.untracked.vint32(0,0,0)                            
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
            MatchDeltaR           = cms.untracked.vdouble(0.5,0.3,0.3),    #One per filter
            NTriggeredTaus        = cms.untracked.vuint32(1,1,1,1), #The first one is for the ref events
            NTriggeredLeptons     = cms.untracked.vuint32(0,0,0,0), #the first one is for the ref events
            TauType               = cms.untracked.vint32(86,94,94),
            LeptonType            = cms.untracked.vint32(0,0,0)                            

        ),
        cms.PSet(
            DQMFolder              = cms.string('HLT/TauOnline/Inclusive/L1'),
            L1Taus                 = cms.InputTag("hltL1extraParticles","Tau"),
            L1Jets                 = cms.InputTag("hltL1extraParticles","Central"),
            L1Leptons              = cms.InputTag("none"),
            LeptonType             = cms.int32(0),
            NTriggeredTaus         = cms.uint32(2),
            NTriggeredLeptons      = cms.uint32(0)
        ),

        cms.PSet(
            DQMFolder              = cms.string('HLT/TauOnline/Inclusive/L2'),
            L2InfoAssociationInput = cms.InputTag("hltL2TauNarrowConeIsolationProducer"),
            L2IsolatedJets         = cms.InputTag("hltL2TauRelaxingIsolationSelector","Isolated")
        ),


#        cms.PSet(
#           DQMFolder              = cms.string('HLT/TauOnline/Inclusive/L25'),
#            ConeIsolation          = cms.InputTag("hltL25TauConeIsolation"),
#            IsolatedJets           = cms.InputTag("hltL25TauLeadingTrackPtCutSelector"),                             
#            Type                   = cms.string('L25')                           
#        ),
#        cms.PSet(
#            DQMFolder              = cms.string('HLT/TauOnline/Inclusive/L3'),
#            ConeIsolation          = cms.InputTag("hltL3TauConeIsolation"),
#            IsolatedJets           = cms.InputTag("hltL3TauIsolationSelector"),                             
#            Type                   = cms.string('L3')                           
#        ),


     cms.PSet(
      L1Dirs                  = cms.vstring(""),
      caloDirs                = cms.vstring(
                            "HLT/TauOnline/Inclusive/L2"
                            ),
      trackDirs               = cms.vstring(
#                            "HLT/TauOnline/Inclusive/L25",
#                            "HLT/TauOnline/Inclusive/L3"
                            ),
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
#        "Track",
#        "Track",
        "Summary"
    ),
    
   doMatching = cms.bool(False),

   matchFilter         = cms.untracked.VInputTag(cms.InputTag("","",hltTauDQMProcess) ),
   matchObjectID       = cms.untracked.vint32(92),
   matchObjectMinPt    = cms.untracked.vdouble(15),
   TriggerEvent = cms.InputTag("hltTriggerSummaryRAW","",hltTauDQMProcess )                          
)




hltTauElectronMonitor = cms.EDFilter("HLTTauDQMSource",
    MonitorSetup = cms.VPSet(
        cms.PSet(
            triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryRAW","",hltTauDQMProcess),
            DQMFolder             = cms.untracked.string('HLT/TauOnline/Electrons/DoubleTau'),
            Filter                = cms.untracked.VInputTag(
                                        cms.InputTag("hltL1sDoubleLooseIsoTau15","",hltTauDQMProcess), 
                                        cms.InputTag("hltFilterL2EtCutDoubleLooseIsoTau15","",hltTauDQMProcess),
                                        cms.InputTag("hltFilterL2EcalIsolationDoubleLooseIsoTau15","",hltTauDQMProcess)
                                        ),
            MatchDeltaR           = cms.untracked.vdouble(0.5,0.3,0.3),    #One per filter
            NTriggeredTaus        = cms.untracked.vuint32(2,2,2,2), #The first one is for the ref events
            NTriggeredLeptons     = cms.untracked.vuint32(0,0,0,0), #the first one is for the ref events
            TauType               = cms.untracked.vint32(86,94,94),
            LeptonType            = cms.untracked.vint32(0,0,0)                            
        ),

        cms.PSet(
            triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryAOD","",hltTauDQMProcess),
            DQMFolder             = cms.untracked.string('HLT/TauOnline/Electrons/Summary'),
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
            DQMFolder             = cms.untracked.string('HLT/TauOnline/Electrons/SingleTau'),
            Filter                = cms.untracked.VInputTag(
                                        cms.InputTag("hltL1sSingleLooseIsoTau20","",hltTauDQMProcess), 
                                        cms.InputTag("hltFilterL2EtCutSingleLooseIsoTau20","",hltTauDQMProcess),
                                        cms.InputTag("hltFilterL2EcalIsolationSingleLooseIsoTau20","",hltTauDQMProcess)
                                        ),
            MatchDeltaR           = cms.untracked.vdouble(0.5,0.3,0.3),    #One per filter
            NTriggeredTaus        = cms.untracked.vuint32(1,1,1,1), #The first one is for the ref events
            NTriggeredLeptons     = cms.untracked.vuint32(0,0,0,0), #the first one is for the ref events
            TauType               = cms.untracked.vint32(86,94,94),
            LeptonType            = cms.untracked.vint32(0,0,0)                            

        ),
        cms.PSet(
            DQMFolder              = cms.string('HLT/TauOnline/Electrons/L1'),
            L1Taus                 = cms.InputTag("hltL1extraParticles","Tau"),
            L1Jets                 = cms.InputTag("hltL1extraParticles","Central"),
            L1Leptons              = cms.InputTag("none"),
            LeptonType             = cms.int32(0),
            NTriggeredTaus         = cms.uint32(2),
            NTriggeredLeptons      = cms.uint32(0)
        ),

        cms.PSet(
            DQMFolder              = cms.string('HLT/TauOnline/Electrons/L2'),
            L2InfoAssociationInput = cms.InputTag("hltL2TauNarrowConeIsolationProducer"),
            L2IsolatedJets         = cms.InputTag("hltL2TauRelaxingIsolationSelector","Isolated")
        ),


#        cms.PSet(
#            DQMFolder              = cms.string('HLT/TauOnline/Electrons/L25'),
#            ConeIsolation          = cms.InputTag("hltL25TauConeIsolation"),
#            IsolatedJets           = cms.InputTag("hltL25TauLeadingTrackPtCutSelector"),                             
#            Type                   = cms.string('L25')                           
#        ),
#        cms.PSet(
#            DQMFolder              = cms.string('HLT/TauOnline/Electrons/L3'),
#            ConeIsolation          = cms.InputTag("hltL3TauConeIsolation"),
#            IsolatedJets           = cms.InputTag("hltL3TauIsolationSelector"),                             
#            Type                   = cms.string('L3')                           
#        ),
      cms.PSet(
      L1Dirs                  = cms.vstring(
                           "HLT/TauOnline/Electrons/L1"
                           ),
      caloDirs                = cms.vstring(
                            "HLT/TauOnline/Electrons/L2"
                            ),
      trackDirs               = cms.vstring(
 #                           "HLT/TauOnline/Electrons/L25",
 #                           "HLT/TauOnline/Electrons/L3"
                            ),
      pathDirs                = cms.vstring(
                            "HLT/TauOnline/Electrons/DoubleTau",
                            "HLT/TauOnline/Electrons/SingleTau"
                             ),
      pathSummaryDirs         = cms.vstring(
                            'HLT/TauOnline/Electrons/Summary'
                            )
      )



    ),


    ConfigType = cms.vstring(
        "Path",
        "LitePath",
        "Path",
        "L1",
        "Calo",
#        "Track",
#        "Track",
        "Summary"
    ),
    
   doMatching = cms.bool(True),

   matchFilter         = cms.untracked.VInputTag(cms.InputTag("hltL1NonIsoHLTNonIsoSingleElectronLWEt15ESDoubleSC10","",hltTauDQMProcess) ),
   matchObjectID       = cms.untracked.vint32(92),
   matchObjectMinPt    = cms.untracked.vdouble(10),
   TriggerEvent = cms.InputTag("hltTriggerSummaryRAW","",hltTauDQMProcess )                          
)

hltMonTauReco =cms.Sequence(hltTauMonitor+hltTauElectronMonitor)



