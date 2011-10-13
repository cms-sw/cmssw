import FWCore.ParameterSet.Config as cms

hltTauValidationProcess_IDEAL = "HLT"
hltTauValidationFolder_MC_IDEAL =  'HLT/TauRelVal/MC_1E33/'
hltTauValidationFolder_PF_IDEAL =  'HLT/TauRelVal/PF_1E33/'
hltTauValIdealMonitorMC1E33 = cms.EDAnalyzer("HLTTauDQMOfflineSource",
    MonitorSetup = cms.VPSet(
        cms.PSet(
            ConfigType            = cms.untracked.string("Path"),
            triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryRAW","",hltTauValidationProcess_IDEAL),
            DQMFolder             = cms.untracked.string(hltTauValidationFolder_MC_IDEAL+'DoubleTau'),
            Filter                = cms.untracked.VInputTag(
                                        cms.InputTag("hltL1sDoubleIsoTau40Trk5","",hltTauValidationProcess_IDEAL), 
                                        cms.InputTag("hltFilterL2EtCutDoublePFIsoTau40Trk5","",hltTauValidationProcess_IDEAL),
                                        cms.InputTag("hltDoublePFTauTightIso40Track","",hltTauValidationProcess_IDEAL),
                                        cms.InputTag("hltDoublePFTauTightIso40Track5","",hltTauValidationProcess_IDEAL),
                                        cms.InputTag("hltFilterDoubleIsoPFTau40Trk5LeadTrack5IsolationL1HLTMatched","",hltTauValidationProcess_IDEAL)
                                        ),
            MatchDeltaR           = cms.untracked.vdouble(0.5,0.2,0.2,0.2,0.2),    #One per filter
            NTriggeredTaus        = cms.untracked.vuint32(2,2,2,2,2,2), #The first one is for the ref events
            NTriggeredLeptons     = cms.untracked.vuint32(0,0,0,0,0,0), #the first one is for the ref events
            TauType               = cms.untracked.vint32(-86,84,84,84,84),
            LeptonType            = cms.untracked.vint32(0,0,0,0,0)                            
        ),
        cms.PSet(
            ConfigType            = cms.untracked.string("Path"),
            triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryRAW","",hltTauValidationProcess_IDEAL),
            DQMFolder             = cms.untracked.string(hltTauValidationFolder_MC_IDEAL+'EleTau'),
            Filter                = cms.untracked.VInputTag(
                                        cms.InputTag("hltL1sL1SingleEG15","",hltTauValidationProcess_IDEAL), 
                                        cms.InputTag("hltEle18CaloIdVTCaloIsoTTrkIdTTrkIsoTTrackIsoFilter","",hltTauValidationProcess_IDEAL),
                                        cms.InputTag("hltPFTau20","",hltTauValidationProcess_IDEAL),
                                        cms.InputTag("hltPFTau20Track","",hltTauValidationProcess_IDEAL),
                                        cms.InputTag("hltPFTau20TrackLooseIso","",hltTauValidationProcess_IDEAL),
                                        cms.InputTag("hltOverlapFilterIsoEle18IsoPFTau20","",hltTauValidationProcess_IDEAL)
                                        ),
            MatchDeltaR           = cms.untracked.vdouble(0.5,0.2,0.2,0.2,0.2,0.2),    #One per filter
            NTriggeredTaus        = cms.untracked.vuint32(1,0,0,1,1,1,1), #The first one is for the ref events
            NTriggeredLeptons     = cms.untracked.vuint32(1,1,1,0,0,0,1), #the first one is for the ref events
            TauType               = cms.untracked.vint32(0,0,84,84,84,84),
            LeptonType            = cms.untracked.vint32(-83,82,0,0,0,82)                            
        ),
        cms.PSet(
            ConfigType            = cms.untracked.string("Path"),
            triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryRAW","",hltTauValidationProcess_IDEAL),
            DQMFolder             = cms.untracked.string(hltTauValidationFolder_MC_IDEAL+'MuLooseTau'),
            Filter                = cms.untracked.VInputTag(
                                        cms.InputTag("hltL1sL1SingleMu10","",hltTauValidationProcess_IDEAL), 
                                        cms.InputTag("hltSingleMuIsoL3IsoFiltered15","",hltTauValidationProcess_IDEAL),
                                        cms.InputTag("hltPFTau20","",hltTauValidationProcess_IDEAL),
                                        cms.InputTag("hltPFTau20Track","",hltTauValidationProcess_IDEAL),
                                        cms.InputTag("hltPFTau20TrackLooseIso","",hltTauValidationProcess_IDEAL),
                                        cms.InputTag("hltOverlapFilterIsoMu15IsoPFTau20","",hltTauValidationProcess_IDEAL)
                                        ),
            MatchDeltaR           = cms.untracked.vdouble(0.5,0.2,0.2,0.2,0.2,0.2),    #One per filter
            NTriggeredTaus        = cms.untracked.vuint32(1,0,0,1,1,1,1), #The first one is for the ref events
            NTriggeredLeptons     = cms.untracked.vuint32(1,1,1,0,0,0,1), #the first one is for the ref events
            TauType               = cms.untracked.vint32(0,0,84,84,84,84),
            LeptonType            = cms.untracked.vint32(-81,83,0,0,0,83)                            
        ),
        cms.PSet(
            ConfigType            = cms.untracked.string("Path"),
            triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryRAW","",hltTauValidationProcess_IDEAL),
            DQMFolder             = cms.untracked.string(hltTauValidationFolder_MC_IDEAL+'MuTightTau'),
            Filter                = cms.untracked.VInputTag(
                                        cms.InputTag("hltL1sL1SingleMu10","",hltTauValidationProcess_IDEAL), 
                                        cms.InputTag("hltSingleMuIsoL3IsoFiltered15","",hltTauValidationProcess_IDEAL),
                                        cms.InputTag("hltPFTauTightIso20","",hltTauValidationProcess_IDEAL),
                                        cms.InputTag("hltPFTauTightIso20Track","",hltTauValidationProcess_IDEAL),
                                        cms.InputTag("hltPFTauTightIso20TrackTightIso","",hltTauValidationProcess_IDEAL),
                                        cms.InputTag("hltOverlapFilterIsoMu15TightIsoPFTau20","",hltTauValidationProcess_IDEAL)
                                        ),
            MatchDeltaR           = cms.untracked.vdouble(0.5,0.2,0.2,0.2,0.2,0.2),    #One per filter
            NTriggeredTaus        = cms.untracked.vuint32(1,0,0,1,1,1,1), #The first one is for the ref events
            NTriggeredLeptons     = cms.untracked.vuint32(1,1,1,0,0,0,1), #the first one is for the ref events
            TauType               = cms.untracked.vint32(0,0,84,84,84,84),
            LeptonType            = cms.untracked.vint32(-81,83,0,0,0,83)                            
        ),
       cms.PSet(
            ConfigType            = cms.untracked.string("LitePath"),
            triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryAOD","",hltTauValidationProcess_IDEAL),
            DQMFolder             = cms.untracked.string(hltTauValidationFolder_MC_IDEAL+'Summary'),
            Filter                = cms.untracked.VInputTag(
                                        cms.InputTag("hltFilterDoubleIsoPFTau40Trk5LeadTrack5IsolationL1HLTMatched","",hltTauValidationProcess_IDEAL),
                                        cms.InputTag("hltOverlapFilterIsoEle18IsoPFTau20","",hltTauValidationProcess_IDEAL),
                                        cms.InputTag("hltOverlapFilterIsoMu15IsoPFTau20","",hltTauValidationProcess_IDEAL),
                                        cms.InputTag("hltFilterSingleIsoPFTau35Trk20MET60LeadTrack20IsolationL1HLTMatched","",hltTauValidationProcess_IDEAL)
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
            ConfigType            = cms.untracked.string("Path"),
            triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryRAW","",hltTauValidationProcess_IDEAL),
            DQMFolder             = cms.untracked.string(hltTauValidationFolder_MC_IDEAL+'SingleTau'),
            Filter                = cms.untracked.VInputTag(
                                        cms.InputTag("hltL1sSingleIsoTau35Trk20MET60","",hltTauValidationProcess_IDEAL), 
                                        cms.InputTag("hltFilterL2EtCutSingleIsoPFTau35Trk20MET60","",hltTauValidationProcess_IDEAL),
                                        cms.InputTag("hltPFTauTightIso35","",hltTauValidationProcess_IDEAL),
                                        cms.InputTag("hltPFTauTightIso35Track","",hltTauValidationProcess_IDEAL),
                                        cms.InputTag("hltFilterSingleIsoPFTau35Trk20LeadTrackPt20","",hltTauValidationProcess_IDEAL),
                                        cms.InputTag("hltFilterSingleIsoPFTau35Trk20MET60LeadTrack20IsolationL1HLTMatched","",hltTauValidationProcess_IDEAL)
                                        ),
            MatchDeltaR           = cms.untracked.vdouble(0.5,0.2,0.2,0.2,0.2,0.2),    #One per filter
            NTriggeredTaus        = cms.untracked.vuint32(1,1,1,1,1,1,1), #The first one is for the ref events
            NTriggeredLeptons     = cms.untracked.vuint32(0,0,0,0,0,0,0), #the first one is for the ref events
            TauType               = cms.untracked.vint32(-86,84,84,84,84,84),
            LeptonType            = cms.untracked.vint32(0,0,0,0,0,0)                            
        ),
        cms.PSet(
            ConfigType            = cms.untracked.string("L1"),
            DQMFolder              = cms.string(hltTauValidationFolder_MC_IDEAL+'L1'),
            L1Taus                 = cms.InputTag("hltL1extraParticles","Tau"),
            L1Jets                 = cms.InputTag("hltL1extraParticles","Central"),
            L1Electrons            = cms.InputTag("hltL1extraParticles","NonIsolated"),
            L1Muons                = cms.InputTag("hltL1extraParticles")
        ),

        
   ),
    
   doMatching = cms.bool(True),
      refObjects = cms.untracked.VInputTag(
          cms.InputTag("TauMCProducer","HadronicTauOneAndThreeProng"),
          cms.InputTag("TauMCProducer","LeptonicTauElectrons"),
          cms.InputTag("TauMCProducer","LeptonicTauMuons")
     )
)


hltTauValIdealMonitorPF1E33 = cms.EDAnalyzer("HLTTauDQMOfflineSource",
    MonitorSetup = cms.VPSet(
        cms.PSet(
            ConfigType            = cms.untracked.string("Path"),
            triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryRAW","",hltTauValidationProcess_IDEAL),
            DQMFolder             = cms.untracked.string(hltTauValidationFolder_PF_IDEAL+'DoubleTau'),
            Filter                = cms.untracked.VInputTag(
                                        cms.InputTag("hltL1sDoubleIsoTau40Trk5","",hltTauValidationProcess_IDEAL), 
                                        cms.InputTag("hltFilterL2EtCutDoublePFIsoTau40Trk5","",hltTauValidationProcess_IDEAL),
                                        cms.InputTag("hltDoublePFTauTightIso40Track","",hltTauValidationProcess_IDEAL),
                                        cms.InputTag("hltDoublePFTauTightIso40Track5","",hltTauValidationProcess_IDEAL),
                                        cms.InputTag("hltFilterDoubleIsoPFTau40Trk5LeadTrack5IsolationL1HLTMatched","",hltTauValidationProcess_IDEAL)
                                        ),
            MatchDeltaR           = cms.untracked.vdouble(0.5,0.2,0.2,0.2,0.2),    #One per filter
            NTriggeredTaus        = cms.untracked.vuint32(2,2,2,2,2,2), #The first one is for the ref events
            NTriggeredLeptons     = cms.untracked.vuint32(0,0,0,0,0,0), #the first one is for the ref events
            TauType               = cms.untracked.vint32(-86,84,84,84,84),
            LeptonType            = cms.untracked.vint32(0,0,0,0,0)                            
        ),
        
        cms.PSet(
            ConfigType            = cms.untracked.string("Path"),
            triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryRAW","",hltTauValidationProcess_IDEAL),
            DQMFolder             = cms.untracked.string(hltTauValidationFolder_PF_IDEAL+'EleTau'),
            Filter                = cms.untracked.VInputTag(
                                        cms.InputTag("hltL1sL1SingleEG15","",hltTauValidationProcess_IDEAL), 
                                        cms.InputTag("hltEle18CaloIdVTCaloIsoTTrkIdTTrkIsoTTrackIsoFilter","",hltTauValidationProcess_IDEAL),
                                        cms.InputTag("hltPFTau20","",hltTauValidationProcess_IDEAL),
                                        cms.InputTag("hltPFTau20Track","",hltTauValidationProcess_IDEAL),
                                        cms.InputTag("hltPFTau20TrackLooseIso","",hltTauValidationProcess_IDEAL),
                                        cms.InputTag("hltOverlapFilterIsoEle18IsoPFTau20","",hltTauValidationProcess_IDEAL)
                                        ),
            MatchDeltaR           = cms.untracked.vdouble(0.5,0.2,0.2,0.2,0.2,0.2),    #One per filter
            NTriggeredTaus        = cms.untracked.vuint32(1,0,0,1,1,1,1), #The first one is for the ref events
            NTriggeredLeptons     = cms.untracked.vuint32(1,1,1,0,0,0,1), #the first one is for the ref events
            TauType               = cms.untracked.vint32(0,0,84,84,84,84),
            LeptonType            = cms.untracked.vint32(-83,82,0,0,0,82)                            
        ),
        
        cms.PSet(
            ConfigType            = cms.untracked.string("Path"),
            triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryRAW","",hltTauValidationProcess_IDEAL),
            DQMFolder             = cms.untracked.string(hltTauValidationFolder_PF_IDEAL+'MuLooseTau'),
            Filter                = cms.untracked.VInputTag(
                                        cms.InputTag("hltL1sL1SingleMu10","",hltTauValidationProcess_IDEAL), 
                                        cms.InputTag("hltSingleMuIsoL3IsoFiltered15","",hltTauValidationProcess_IDEAL),
                                        cms.InputTag("hltPFTau20","",hltTauValidationProcess_IDEAL),
                                        cms.InputTag("hltPFTau20Track","",hltTauValidationProcess_IDEAL),
                                        cms.InputTag("hltPFTau20TrackLooseIso","",hltTauValidationProcess_IDEAL),
                                        cms.InputTag("hltOverlapFilterIsoMu15IsoPFTau20","",hltTauValidationProcess_IDEAL)
                                        ),
            MatchDeltaR           = cms.untracked.vdouble(0.5,0.2,0.2,0.2,0.2,0.2),    #One per filter
            NTriggeredTaus        = cms.untracked.vuint32(1,0,0,1,1,1,1), #The first one is for the ref events
            NTriggeredLeptons     = cms.untracked.vuint32(1,1,1,0,0,0,1), #the first one is for the ref events
            TauType               = cms.untracked.vint32(0,0,84,84,84,84),
            LeptonType            = cms.untracked.vint32(-81,83,0,0,0,83)                            
        ),
        cms.PSet(
            ConfigType            = cms.untracked.string("Path"),
            triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryRAW","",hltTauValidationProcess_IDEAL),
            DQMFolder             = cms.untracked.string(hltTauValidationFolder_PF_IDEAL+'MuTightTau'),
            Filter                = cms.untracked.VInputTag(
                                        cms.InputTag("hltL1sL1SingleMu10","",hltTauValidationProcess_IDEAL), 
                                        cms.InputTag("hltSingleMuIsoL3IsoFiltered15","",hltTauValidationProcess_IDEAL),
                                        cms.InputTag("hltPFTauTightIso20","",hltTauValidationProcess_IDEAL),
                                        cms.InputTag("hltPFTauTightIso20Track","",hltTauValidationProcess_IDEAL),
                                        cms.InputTag("hltPFTauTightIso20TrackTightIso","",hltTauValidationProcess_IDEAL),
                                        cms.InputTag("hltOverlapFilterIsoMu15TightIsoPFTau20","",hltTauValidationProcess_IDEAL)
                                        ),
            MatchDeltaR           = cms.untracked.vdouble(0.5,0.2,0.2,0.2,0.2,0.2),    #One per filter
            NTriggeredTaus        = cms.untracked.vuint32(1,0,0,1,1,1,1), #The first one is for the ref events
            NTriggeredLeptons     = cms.untracked.vuint32(1,1,1,0,0,0,1), #the first one is for the ref events
            TauType               = cms.untracked.vint32(0,0,84,84,84,84),
            LeptonType            = cms.untracked.vint32(-81,83,0,0,0,83)                            
        ),   
        cms.PSet(
            ConfigType            = cms.untracked.string("LitePath"),
            triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryAOD","",hltTauValidationProcess_IDEAL),
            DQMFolder             = cms.untracked.string(hltTauValidationFolder_PF_IDEAL+'Summary'),
            Filter                = cms.untracked.VInputTag(
                                        cms.InputTag("hltFilterDoubleIsoPFTau40Trk5LeadTrack5IsolationL1HLTMatched","",hltTauValidationProcess_IDEAL),
                                        cms.InputTag("hltOverlapFilterIsoEle18IsoPFTau20","",hltTauValidationProcess_IDEAL),
                                        cms.InputTag("hltOverlapFilterIsoMu15IsoPFTau20","",hltTauValidationProcess_IDEAL),
                                        cms.InputTag("hltFilterSingleIsoPFTau35Trk20MET60LeadTrack20IsolationL1HLTMatched","",hltTauValidationProcess_IDEAL)
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
            ConfigType            = cms.untracked.string("Path"),
            triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryRAW","",hltTauValidationProcess_IDEAL),
            DQMFolder             = cms.untracked.string(hltTauValidationFolder_PF_IDEAL+'SingleTau'),
            Filter                = cms.untracked.VInputTag(
                                        cms.InputTag("hltL1sSingleIsoTau35Trk20MET60","",hltTauValidationProcess_IDEAL), 
                                        cms.InputTag("hltFilterL2EtCutSingleIsoPFTau35Trk20MET60","",hltTauValidationProcess_IDEAL),
                                        cms.InputTag("hltPFTauTightIso35","",hltTauValidationProcess_IDEAL),
                                        cms.InputTag("hltPFTauTightIso35Track","",hltTauValidationProcess_IDEAL),
                                        cms.InputTag("hltFilterSingleIsoPFTau35Trk20LeadTrackPt20","",hltTauValidationProcess_IDEAL),
                                        cms.InputTag("hltFilterSingleIsoPFTau35Trk20MET60LeadTrack20IsolationL1HLTMatched","",hltTauValidationProcess_IDEAL)
                                        ),
            MatchDeltaR           = cms.untracked.vdouble(0.5,0.2,0.2,0.2,0.2,0.2),    #One per filter
            NTriggeredTaus        = cms.untracked.vuint32(1,1,1,1,1,1,1), #The first one is for the ref events
            NTriggeredLeptons     = cms.untracked.vuint32(0,0,0,0,0,0,0), #the first one is for the ref events
            TauType               = cms.untracked.vint32(-86,84,84,84,84,84),
            LeptonType            = cms.untracked.vint32(0,0,0,0,0,0)                         
        ),
        cms.PSet(
            ConfigType            = cms.untracked.string("L1"),
            DQMFolder              = cms.string(hltTauValidationFolder_PF_IDEAL+'L1'),
            L1Taus                 = cms.InputTag("hltL1extraParticles","Tau"),
            L1Jets                 = cms.InputTag("hltL1extraParticles","Central"),
            L1Electrons            = cms.InputTag("hltL1extraParticles","NonIsolated"),
            L1Muons                = cms.InputTag("hltL1extraParticles")
        ),

   ),
    
   doMatching = cms.bool(True),
      refObjects = cms.untracked.VInputTag(
          cms.InputTag("TauRefCombiner",""),
          cms.InputTag("TauMCProducer","LeptonicTauElectrons"),
          cms.InputTag("TauMCProducer","LeptonicTauMuons")
     )
)


hltTauValIdeal1E33 = cms.Sequence(hltTauValIdealMonitorMC1E33+hltTauValIdealMonitorPF1E33)

