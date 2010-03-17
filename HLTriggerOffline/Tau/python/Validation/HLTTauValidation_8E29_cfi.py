import FWCore.ParameterSet.Config as cms

hltTauValidationProcess_STARTUP = "HLT"

hltTauValidationFolder_MC =  'HLT/TauRelVal/MC_8E29/'

hltTauValStartupMonitorMC = cms.EDAnalyzer("HLTTauDQMOfflineSource",
    MonitorSetup = cms.VPSet(
        cms.PSet(
            triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryRAW","",hltTauValidationProcess_STARTUP),
            DQMFolder             = cms.untracked.string(hltTauValidationFolder_MC+'DoubleTau'),
            Filter                = cms.untracked.VInputTag(
                                        cms.InputTag("hltL1sDoubleLooseIsoTau15","",hltTauValidationProcess_STARTUP), 
                                        cms.InputTag("hltFilterL2EtCutDoubleLooseIsoTau15","",hltTauValidationProcess_STARTUP),
                                        cms.InputTag("hltFilterL2EcalIsolationDoubleLooseIsoTau15","",hltTauValidationProcess_STARTUP)
                                        ),
            MatchDeltaR           = cms.untracked.vdouble(0.5,0.2,0.2),    #One per filter
            NTriggeredTaus        = cms.untracked.vuint32(2,2,2,2), #The first one is for the ref events
            NTriggeredLeptons     = cms.untracked.vuint32(0,0,0,0,0), #the first one is for the ref events
            TauType               = cms.untracked.vint32(-86,84,84,84),
            LeptonType            = cms.untracked.vint32(0,0,0,0)                            
        ),

        cms.PSet(
            triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryAOD","",hltTauValidationProcess_STARTUP),
            DQMFolder             = cms.untracked.string(hltTauValidationFolder_MC+'Summary'),
            Filter                = cms.untracked.VInputTag(
                                        cms.InputTag("hltFilterL2EcalIsolationDoubleLooseIsoTau15","",hltTauValidationProcess_STARTUP),
                                        cms.InputTag("hltFilterL2EcalIsolationSingleIsoTau20","",hltTauValidationProcess_STARTUP)
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
            triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryRAW","",hltTauValidationProcess_STARTUP),
            DQMFolder             = cms.untracked.string(hltTauValidationFolder_MC+'SingleTau'),
            Filter                = cms.untracked.VInputTag(
                                        cms.InputTag("hltL1sSingleLooseIsoTau20","",hltTauValidationProcess_STARTUP), 
                                        cms.InputTag("hltFilterL2EtCutSingleLooseIsoTau20","",hltTauValidationProcess_STARTUP),
                                        cms.InputTag("hltFilterL2EcalIsolationSingleLooseIsoTau20","",hltTauValidationProcess_STARTUP)
                                        ),
            MatchDeltaR           = cms.untracked.vdouble(0.5,0.2,0.2),    #One per filter
            NTriggeredTaus        = cms.untracked.vuint32(1,1,1,1), #The first one is for the ref events
            NTriggeredLeptons     = cms.untracked.vuint32(0,0,0,0), #the first one is for the ref events
            TauType               = cms.untracked.vint32(-86,84,84),
            LeptonType            = cms.untracked.vint32(0,0,0)                            
        ),
        cms.PSet(
            DQMFolder              = cms.string(hltTauValidationFolder_MC+'L1'),
            L1Taus                 = cms.InputTag("hltL1extraParticles","Tau"),
            L1Jets                 = cms.InputTag("hltL1extraParticles","Central"),
            L1Electrons            = cms.InputTag("hltL1extraParticles","Isolated"),
            L1Muons                = cms.InputTag("hltL1extraParticles")
        ),


        cms.PSet(
            DQMFolder              = cms.string(hltTauValidationFolder_MC+'L2'),
            L2InfoAssociationInput = cms.InputTag("hltL2TauNarrowConeIsolationProducer"),
            L2IsolatedJets         = cms.InputTag("hltL2TauRelaxingIsolationSelector","Isolated"),
            L2RegionalJets         = cms.VInputTag(
                                                   cms.InputTag("hltIconeTau1Regional"),
                                                   cms.InputTag("hltIconeTau2Regional"),
                                                   cms.InputTag("hltIconeTau3Regional"),
                                                   cms.InputTag("hltIconeTau4Regional"),
                                                   cms.InputTag("hltIconeCentral1Regional"),
                                                   cms.InputTag("hltIconeCentral2Regional"),
                                                   cms.InputTag("hltIconeCentral3Regional"),
                                                   cms.InputTag("hltIconeCentral4Regional"))

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
          cms.InputTag("TauMCProducer","HadronicTauOneAndThreeProng"),
          cms.InputTag("TauMCProducer","LeptonicTauElectrons"),
          cms.InputTag("TauMCProducer","LeptonicTauMuons")
     )
)



hltTauValidationFolder_PF =  'HLT/TauRelVal/PF_8E29/'

hltTauValStartupMonitorPF = cms.EDAnalyzer("HLTTauDQMOfflineSource",
    MonitorSetup = cms.VPSet(
        cms.PSet(
            triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryRAW","",hltTauValidationProcess_STARTUP),
            DQMFolder             = cms.untracked.string(hltTauValidationFolder_PF+'DoubleTau'),
            Filter                = cms.untracked.VInputTag(
                                        cms.InputTag("hltL1sDoubleLooseIsoTau15","",hltTauValidationProcess_STARTUP), 
                                        cms.InputTag("hltFilterL2EtCutDoubleLooseIsoTau15","",hltTauValidationProcess_STARTUP),
                                        cms.InputTag("hltFilterL2EcalIsolationDoubleLooseIsoTau15","",hltTauValidationProcess_STARTUP)
                                        ),
            MatchDeltaR           = cms.untracked.vdouble(0.5,0.2,0.2),    #One per filter
            NTriggeredTaus        = cms.untracked.vuint32(2,2,2,2), #The first one is for the ref events
            NTriggeredLeptons     = cms.untracked.vuint32(0,0,0,0,0), #the first one is for the ref events
            TauType               = cms.untracked.vint32(-86,84,84,84),
            LeptonType            = cms.untracked.vint32(0,0,0,0)                            
        ),

        cms.PSet(
            triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryAOD","",hltTauValidationProcess_STARTUP),
            DQMFolder             = cms.untracked.string(hltTauValidationFolder_PF+'Summary'),
            Filter                = cms.untracked.VInputTag(
                                        cms.InputTag("hltFilterL2EcalIsolationDoubleLooseIsoTau15","",hltTauValidationProcess_STARTUP),
                                        cms.InputTag("hltFilterL2EcalIsolationSingleIsoTau20","",hltTauValidationProcess_STARTUP)
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
            triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryRAW","",hltTauValidationProcess_STARTUP),
            DQMFolder             = cms.untracked.string(hltTauValidationFolder_PF+'SingleTau'),
            Filter                = cms.untracked.VInputTag(
                                        cms.InputTag("hltL1sSingleLooseIsoTau20","",hltTauValidationProcess_STARTUP), 
                                        cms.InputTag("hltFilterL2EtCutSingleLooseIsoTau20","",hltTauValidationProcess_STARTUP),
                                        cms.InputTag("hltFilterL2EcalIsolationSingleLooseIsoTau20","",hltTauValidationProcess_STARTUP)
                                        ),
            MatchDeltaR           = cms.untracked.vdouble(0.5,0.2,0.2),    #One per filter
            NTriggeredTaus        = cms.untracked.vuint32(1,1,1,1), #The first one is for the ref events
            NTriggeredLeptons     = cms.untracked.vuint32(0,0,0,0), #the first one is for the ref events
            TauType               = cms.untracked.vint32(-86,84,84),
            LeptonType            = cms.untracked.vint32(0,0,0)                            
        ),
        cms.PSet(
            DQMFolder              = cms.string(hltTauValidationFolder_PF+'L1'),
            L1Taus                 = cms.InputTag("hltL1extraParticles","Tau"),
            L1Jets                 = cms.InputTag("hltL1extraParticles","Central"),
            L1Electrons            = cms.InputTag("hltL1extraParticles","Isolated"),
            L1Muons                = cms.InputTag("hltL1extraParticles")
        ),


        cms.PSet(
            DQMFolder              = cms.string(hltTauValidationFolder_PF+'L2'),
            L2InfoAssociationInput = cms.InputTag("hltL2TauNarrowConeIsolationProducer"),
            L2IsolatedJets         = cms.InputTag("hltL2TauRelaxingIsolationSelector","Isolated"),
            L2RegionalJets         = cms.VInputTag(
                                                   cms.InputTag("hltIconeTau1Regional"),
                                                   cms.InputTag("hltIconeTau2Regional"),
                                                   cms.InputTag("hltIconeTau3Regional"),
                                                   cms.InputTag("hltIconeTau4Regional"),
                                                   cms.InputTag("hltIconeCentral1Regional"),
                                                   cms.InputTag("hltIconeCentral2Regional"),
                                                   cms.InputTag("hltIconeCentral3Regional"),
                                                   cms.InputTag("hltIconeCentral4Regional"))

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
          cms.InputTag("TauRefCombiner",""),
          cms.InputTag("TauMCProducer","LeptonicTauElectrons"),
          cms.InputTag("TauMCProducer","LeptonicTauMuons")
     )
)




hltTauValStartup = cms.Sequence(hltTauValStartupMonitorMC+hltTauValStartupMonitorPF)

