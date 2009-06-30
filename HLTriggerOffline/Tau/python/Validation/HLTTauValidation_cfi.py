import FWCore.ParameterSet.Config as cms

hltTauValidationProcess = "HLT"

hltTauValDefMonitor = cms.EDFilter("HLTTauDQMOfflineSource",
    MonitorSetup = cms.VPSet(
        cms.PSet(
            triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryRAW","",hltTauValidationProcess),
            DQMFolder             = cms.untracked.string('HLT/TauRelVal/MC_Default/DoubleTau'),
            Filter                = cms.untracked.VInputTag(
                                        cms.InputTag("hltL1sDoubleLooseIsoTau15","",hltTauValidationProcess), 
                                        cms.InputTag("hltFilterL2EtCutDoubleLooseIsoTau15","",hltTauValidationProcess),
                                        cms.InputTag("hltFilterL2EcalIsolationDoubleLooseIsoTau15","",hltTauValidationProcess)
                                        ),
            MatchDeltaR           = cms.untracked.vdouble(0.5,0.3,0.3,0.3),    #One per filter
            NTriggeredTaus        = cms.untracked.vuint32(2,2,2,2,2), #The first one is for the ref events
            NTriggeredLeptons     = cms.untracked.vuint32(0,0,0,0,0), #the first one is for the ref events
            TauType               = cms.untracked.vint32(-86,84,84,84),
            LeptonType            = cms.untracked.vint32(0,0,0,0)                            
        ),

        cms.PSet(
            triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryAOD","",hltTauValidationProcess),
            DQMFolder             = cms.untracked.string('HLT/TauRelVal/MC_Default/Summary'),
            Filter                = cms.untracked.VInputTag(
                                        cms.InputTag("hltFilterL2EcalIsolationDoubleLooseIsoTau15","",hltTauValidationProcess),
                                        cms.InputTag("hltFilterL2EcalIsolationSingleIsoTau20","",hltTauValidationProcess)
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
            triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryRAW","",hltTauValidationProcess),
            DQMFolder             = cms.untracked.string('HLT/TauRelVal/MC_Default/SingleTau'),
            Filter                = cms.untracked.VInputTag(
                                        cms.InputTag("hltL1sSingleLooseIsoTau20","",hltTauValidationProcess), 
                                        cms.InputTag("hltFilterL2EtCutSingleLooseIsoTau20","",hltTauValidationProcess),
                                        cms.InputTag("hltFilterL2EcalIsolationSingleLooseIsoTau20","",hltTauValidationProcess)
                                        ),
            MatchDeltaR           = cms.untracked.vdouble(0.5,0.3,0.3,0.3,0,0.3),    #One per filter
            NTriggeredTaus        = cms.untracked.vuint32(1,1,1,1,1,1), #The first one is for the ref events
            NTriggeredLeptons     = cms.untracked.vuint32(0,0,0,0,0,0), #the first one is for the ref events
            TauType               = cms.untracked.vint32(-86,84,84,84,84),
            LeptonType            = cms.untracked.vint32(0,0,0,0,0)                            
        ),
        cms.PSet(
            DQMFolder              = cms.string('HLT/TauRelVal/MC_Default/L1'),
            L1Taus                 = cms.InputTag("hltL1extraParticles","Tau"),
            L1Jets                 = cms.InputTag("hltL1extraParticles","Central"),
            L1Electrons            = cms.InputTag("hltL1extraParticles","Isolated"),
            L1Muons                = cms.InputTag("hltL1extraParticles")
        ),

        cms.PSet(
            DQMFolder              = cms.string('HLT/TauRelVal/MC_Default/L2'),
            L2InfoAssociationInput = cms.InputTag("hltL2TauNarrowConeIsolationProducer"),
            L2IsolatedJets         = cms.InputTag("hltL2TauRelaxingIsolationSelector","Isolated")
        ),

   ),
    ConfigType = cms.vstring(
        "Path",
        "LitePath",
        "Path",
        "L1",
        "Calo"
    ),
    
   doMatching = cms.bool(True),
      refObjects = cms.untracked.VInputTag(
      cms.InputTag("TauMCProducer","HadronicTauOneAndThreeProng")
     )
)


hltTauValDefault = cms.Sequence(hltTauValDefMonitor)

