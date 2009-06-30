import FWCore.ParameterSet.Config as cms

hltTauValidationProcess_IDEAL = "HLT"

hltTauValIdealMonitor = cms.EDFilter("HLTTauDQMOfflineSource",
    MonitorSetup = cms.VPSet(
        cms.PSet(
            triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryRAW","",hltTauValidationProcess_IDEAL),
            DQMFolder             = cms.untracked.string('HLT/TauRelVal/MC_1E31/DoubleTau'),
            Filter                = cms.untracked.VInputTag(
                                        cms.InputTag("hltL1sDoubleLooseIsoTau15Trk5","",hltTauValidationProcess_IDEAL), 
                                        cms.InputTag("hltFilterL2EtCutDoubleLooseIsoTau15Trk5","",hltTauValidationProcess_IDEAL),
                                        cms.InputTag("hltFilterL2EcalIsolationDoubleLooseIsoTau15Trk5","",hltTauValidationProcess_IDEAL),
                                        cms.InputTag("hltFilterL25LeadingTrackPtCutDoubleLooseIsoTau15Trk5","",hltTauValidationProcess_IDEAL)
                                        ),
            MatchDeltaR           = cms.untracked.vdouble(0.5,0.2,0.2,0.2),    #One per filter
            NTriggeredTaus        = cms.untracked.vuint32(2,2,2,2,2), #The first one is for the ref events
            NTriggeredLeptons     = cms.untracked.vuint32(0,0,0,0,0), #the first one is for the ref events
            TauType               = cms.untracked.vint32(-86,84,84,84),
            LeptonType            = cms.untracked.vint32(0,0,0,0)                            
        ),

        cms.PSet(
            triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryAOD","",hltTauValidationProcess_IDEAL),
            DQMFolder             = cms.untracked.string('HLT/TauRelVal/MC_1E31/Summary'),
            Filter                = cms.untracked.VInputTag(
                                        cms.InputTag("hltFilterL25LeadingTrackPtCutDoubleLooseIsoTau15Trk5","",hltTauValidationProcess_IDEAL),
                                        cms.InputTag("hltFilterL3IsolationCutSingleIsoTau30Trk5","",hltTauValidationProcess_IDEAL)
                                        ),
            PathName              = cms.untracked.vstring(
                                        'DoubleLooseIsoTau',
                                        'SingleLooseIsoTau'
                                        ),
            NTriggeredTaus        = cms.untracked.vuint32(2,1), 
            NTriggeredLeptons     = cms.untracked.vuint32(0,0), 
            TauType               = cms.untracked.vint32(0,0),
            LeptonType            = cms.untracked.vint32(0,0)                            
        ),
        
        cms.PSet(
            triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryRAW","",hltTauValidationProcess_IDEAL),
            DQMFolder             = cms.untracked.string('HLT/TauRelVal/MC_1E31/SingleTau'),
            Filter                = cms.untracked.VInputTag(
                                        cms.InputTag("hltL1sSingleIsoTau30Trk5","",hltTauValidationProcess_IDEAL), 
                                        cms.InputTag("hltFilterL2EtCutSingleIsoTau30Trk5","",hltTauValidationProcess_IDEAL),
                                        cms.InputTag("hltFilterL2EcalIsolationSingleIsoTau30Trk5","",hltTauValidationProcess_IDEAL),
                                        cms.InputTag("hltFilterL25LeadingTrackPtCutSingleIsoTau30Trk5","",hltTauValidationProcess_IDEAL),
                                        cms.InputTag("hltFilterL3IsolationCutSingleIsoTau30Trk5","",hltTauValidationProcess_IDEAL)
                                        ),
            MatchDeltaR           = cms.untracked.vdouble(0.5,0.2,0.2,0.2,0.2),    #One per filter
            NTriggeredTaus        = cms.untracked.vuint32(1,1,1,1,1,1), #The first one is for the ref events
            NTriggeredLeptons     = cms.untracked.vuint32(0,0,0,0,0,0), #the first one is for the ref events
            TauType               = cms.untracked.vint32(-86,84,84,84,84),
            LeptonType            = cms.untracked.vint32(0,0,0,0,0)                            
        ),
        cms.PSet(
            DQMFolder              = cms.string('HLT/TauRelVal/MC_1E31/L1'),
            L1Taus                 = cms.InputTag("hltL1extraParticles","Tau"),
            L1Jets                 = cms.InputTag("hltL1extraParticles","Central"),
            L1Electrons            = cms.InputTag("hltL1extraParticles","Isolated"),
            L1Muons                = cms.InputTag("hltL1extraParticles")
        ),
        cms.PSet(
            DQMFolder              = cms.string('HLT/TauRelVal/MC_1E31/L2'),
            L2InfoAssociationInput = cms.InputTag("hltL2TauNarrowConeIsolationProducer"),
            L2IsolatedJets         = cms.InputTag("hltL2TauRelaxingIsolationSelector","Isolated")
        ),
        cms.PSet(
            DQMFolder              = cms.string('HLT/TauRelVal/MC_1E31/L25'),
            ConeIsolation          = cms.InputTag("hltL25TauConeIsolation"),
            IsolatedJets           = cms.InputTag("hltL25TauLeadingTrackPtCutSelector"),
            Type                   = cms.string('L25')
        ),
        cms.PSet(
            DQMFolder              = cms.string('HLT/TauRelVal/MC_1E31/L3'),
            ConeIsolation          = cms.InputTag("hltL3TauConeIsolation"),
            IsolatedJets           = cms.InputTag("hltL3TauIsolationSelector"),
            Type                   = cms.string('L3')
      )
        
   ),
    ConfigType = cms.vstring(
        "Path",
        "LitePath",
        "Path",
        "L1",
        "Calo",
        "Track",
        "Track"
    ),
    
   doMatching = cms.bool(True),
      refObjects = cms.untracked.VInputTag(
      cms.InputTag("TauMCProducer","HadronicTauOneAndThreeProng")
     )
)


hltTauValIdeal = cms.Sequence(hltTauValIdealMonitor)

