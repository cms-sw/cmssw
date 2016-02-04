import FWCore.ParameterSet.Config as cms

hltTauValidationProcess_IDEAL = "HLT"
hltTauValidationFolder_MC_IDEAL =  'HLT/TauRelVal/MC_2E32/'
hltTauValidationFolder_PF_IDEAL =  'HLT/TauRelVal/PF_2E32/'

hltTauValIdealMonitorMC2E32 = cms.EDAnalyzer("HLTTauDQMOfflineSource",
    MonitorSetup = cms.VPSet(
        cms.PSet(
            triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryRAW","",hltTauValidationProcess_IDEAL),
            DQMFolder             = cms.untracked.string(hltTauValidationFolder_MC_IDEAL+'DoubleTau'),
            Filter                = cms.untracked.VInputTag(
                                        cms.InputTag("hltL1sDoubleIsoTau15Trk5","",hltTauValidationProcess_IDEAL), 
                                        cms.InputTag("hltFilterL2EtCutDoubleIsoTau15Trk5","",hltTauValidationProcess_IDEAL),
                                        cms.InputTag("hltFilterL2EcalIsolationDoubleIsoTau15Trk5","",hltTauValidationProcess_IDEAL),
                                        cms.InputTag("hltFilterL25LeadingTrackPtCutDoubleIsoTau15Trk5","",hltTauValidationProcess_IDEAL),
                                        cms.InputTag("hltFilterL3TrackIsolationDoubleIsoTau15Trk5","",hltTauValidationProcess_IDEAL)
                                        ),
            MatchDeltaR           = cms.untracked.vdouble(0.5,0.2,0.2,0.2,0.2),    #One per filter
            NTriggeredTaus        = cms.untracked.vuint32(2,2,2,2,2,2), #The first one is for the ref events
            NTriggeredLeptons     = cms.untracked.vuint32(0,0,0,0,0,0), #the first one is for the ref events
            TauType               = cms.untracked.vint32(-86,84,84,84,84),
            LeptonType            = cms.untracked.vint32(0,0,0,0,0)                            
        ),

        cms.PSet(
            triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryAOD","",hltTauValidationProcess_IDEAL),
            DQMFolder             = cms.untracked.string(hltTauValidationFolder_MC_IDEAL+'Summary'),
            Filter                = cms.untracked.VInputTag(
                                        cms.InputTag("hltFilterL3TrackIsolationDoubleIsoTau15Trk5","",hltTauValidationProcess_IDEAL),
                                        cms.InputTag("hltFilterL3TrackIsolationSingleIsoTau30Trk5MET20","",hltTauValidationProcess_IDEAL)
                                        ),
            PathName              = cms.untracked.vstring(
                                        'DoubleIsoTau',
                                        'SingleIsoTau'
                                        ),
            NTriggeredTaus        = cms.untracked.vuint32(2,1), 
            NTriggeredLeptons     = cms.untracked.vuint32(0,0), 
            TauType               = cms.untracked.vint32(15,15),
            LeptonType            = cms.untracked.vint32(0,0)                            
        ),
        
        cms.PSet(
            triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryRAW","",hltTauValidationProcess_IDEAL),
            DQMFolder             = cms.untracked.string(hltTauValidationFolder_MC_IDEAL+'SingleTau'),
            Filter                = cms.untracked.VInputTag(
                                        cms.InputTag("hltL1sSingleIsoTau20Trk15MET25","",hltTauValidationProcess_IDEAL), 
                                        cms.InputTag("hltFilterL2EtCutSingleIsoTau20Trk15MET25","",hltTauValidationProcess_IDEAL),
                                        cms.InputTag("hltFilterL2EcalIsolationSingleIsoTau20Trk15MET25","",hltTauValidationProcess_IDEAL),
                                        cms.InputTag("hltFilterL25LeadingTrackPtCutSingleIsoTau20Trk15MET25","",hltTauValidationProcess_IDEAL),
                                        cms.InputTag("hltFilterL3TrackIsolationSingleIsoTau20Trk15MET25","",hltTauValidationProcess_IDEAL)
                                        ),
            MatchDeltaR           = cms.untracked.vdouble(0.5,0.2,0.2,0.2,0.2),    #One per filter
            NTriggeredTaus        = cms.untracked.vuint32(1,1,1,1,1,1), #The first one is for the ref events
            NTriggeredLeptons     = cms.untracked.vuint32(0,0,0,0,0,0), #the first one is for the ref events
            TauType               = cms.untracked.vint32(-86,84,84,84,84),
            LeptonType            = cms.untracked.vint32(0,0,0,0,0)                            
        ),
        cms.PSet(
            DQMFolder              = cms.string(hltTauValidationFolder_MC_IDEAL+'L1'),
            L1Taus                 = cms.InputTag("hltL1extraParticles","Tau"),
            L1Jets                 = cms.InputTag("hltL1extraParticles","Central"),
            L1Electrons            = cms.InputTag("hltL1extraParticles","Isolated"),
            L1Muons                = cms.InputTag("hltL1extraParticles")
        ),
        cms.PSet(
            DQMFolder              = cms.string(hltTauValidationFolder_MC_IDEAL+'L2'),
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
        cms.PSet(
            DQMFolder              = cms.string(hltTauValidationFolder_MC_IDEAL+'L25'),
            ConeIsolation          = cms.InputTag("hltL25TauConeIsolation"),
            IsolatedJets           = cms.InputTag("hltL25TauLeadingTrackPtCutSelector"),
            Type                   = cms.string('L25')
        ),
        cms.PSet(
            DQMFolder              = cms.string(hltTauValidationFolder_MC_IDEAL+'L3'),
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
          cms.InputTag("TauMCProducer","HadronicTauOneAndThreeProng"),
          cms.InputTag("TauMCProducer","LeptonicTauElectrons"),
          cms.InputTag("TauMCProducer","LeptonicTauMuons")
     )
)


hltTauValIdealMonitorPF2E32 = cms.EDAnalyzer("HLTTauDQMOfflineSource",
    MonitorSetup = cms.VPSet(
        cms.PSet(
            triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryRAW","",hltTauValidationProcess_IDEAL),
            DQMFolder             = cms.untracked.string(hltTauValidationFolder_PF_IDEAL+'DoubleTau'),
            Filter                = cms.untracked.VInputTag(
                                        cms.InputTag("hltL1sSingleIsoTau20Trk15MET25","",hltTauValidationProcess_IDEAL), 
                                        cms.InputTag("hltFilterL2EtCutSingleIsoTau20Trk15MET25","",hltTauValidationProcess_IDEAL),
                                        cms.InputTag("hltFilterL2EcalIsolationSingleIsoTau20Trk15MET25","",hltTauValidationProcess_IDEAL),
                                        cms.InputTag("hltFilterL25LeadingTrackPtCutSingleIsoTau20Trk15MET25","",hltTauValidationProcess_IDEAL),
                                        cms.InputTag("hltFilterL3TrackIsolationSingleIsoTau20Trk15MET25","",hltTauValidationProcess_IDEAL)
                                        ),
            MatchDeltaR           = cms.untracked.vdouble(0.5,0.2,0.2,0.2,0.2),    #One per filter
            NTriggeredTaus        = cms.untracked.vuint32(2,2,2,2,2,2), #The first one is for the ref events
            NTriggeredLeptons     = cms.untracked.vuint32(0,0,0,0,0,0), #the first one is for the ref events
            TauType               = cms.untracked.vint32(-86,84,84,84,84),
            LeptonType            = cms.untracked.vint32(0,0,0,0,0)                            
        ),

        cms.PSet(
            triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryAOD","",hltTauValidationProcess_IDEAL),
            DQMFolder             = cms.untracked.string(hltTauValidationFolder_PF_IDEAL+'Summary'),
            Filter                = cms.untracked.VInputTag(
                                        cms.InputTag("hltFilterL3TrackIsolationDoubleIsoTau15Trk5","",hltTauValidationProcess_IDEAL),
                                        cms.InputTag("hltFilterL3TrackIsolationSingleIsoTau30Trk5MET20","",hltTauValidationProcess_IDEAL)
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
            triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryRAW","",hltTauValidationProcess_IDEAL),
            DQMFolder             = cms.untracked.string(hltTauValidationFolder_PF_IDEAL+'SingleTau'),
            Filter                = cms.untracked.VInputTag(
                                        cms.InputTag("hltL1sSingleIsoTau20Trk5MET20","",hltTauValidationProcess_IDEAL), 
                                        cms.InputTag("hltFilterL2EtCutSingleIsoTau30Trk5MET20","",hltTauValidationProcess_IDEAL),
                                        cms.InputTag("hltFilterL2EcalIsolationSingleIsoTau30Trk5MET20","",hltTauValidationProcess_IDEAL),
                                        cms.InputTag("hltFilterL25LeadingTrackPtCutSingleIsoTau30Trk5MET20","",hltTauValidationProcess_IDEAL),
                                        cms.InputTag("hltFilterL3TrackIsolationSingleIsoTau30Trk5MET20","",hltTauValidationProcess_IDEAL)
                                        ),
            MatchDeltaR           = cms.untracked.vdouble(0.5,0.2,0.2,0.2,0.2),    #One per filter
            NTriggeredTaus        = cms.untracked.vuint32(1,1,1,1,1,1), #The first one is for the ref events
            NTriggeredLeptons     = cms.untracked.vuint32(0,0,0,0,0,0), #the first one is for the ref events
            TauType               = cms.untracked.vint32(-86,84,84,84,84),
            LeptonType            = cms.untracked.vint32(0,0,0,0,0)                            
        ),
        cms.PSet(
            DQMFolder              = cms.string(hltTauValidationFolder_PF_IDEAL+'L1'),
            L1Taus                 = cms.InputTag("hltL1extraParticles","Tau"),
            L1Jets                 = cms.InputTag("hltL1extraParticles","Central"),
            L1Electrons            = cms.InputTag("hltL1extraParticles","Isolated"),
            L1Muons                = cms.InputTag("hltL1extraParticles")
        ),
        cms.PSet(
            DQMFolder              = cms.string(hltTauValidationFolder_PF_IDEAL+'L2'),
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
        cms.PSet(
            DQMFolder              = cms.string(hltTauValidationFolder_PF_IDEAL+'L25'),
            ConeIsolation          = cms.InputTag("hltL25TauConeIsolation"),
            IsolatedJets           = cms.InputTag("hltL25TauLeadingTrackPtCutSelector"),
            Type                   = cms.string('L25')
        ),
        cms.PSet(
            DQMFolder              = cms.string(hltTauValidationFolder_PF_IDEAL+'L3'),
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
          cms.InputTag("TauRefCombiner",""),
          cms.InputTag("TauMCProducer","LeptonicTauElectrons"),
          cms.InputTag("TauMCProducer","LeptonicTauMuons")
     )
)


hltTauValIdeal2E32 = cms.Sequence(hltTauValIdealMonitorMC2E32+hltTauValIdealMonitorPF2E32)

