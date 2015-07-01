pathsAndFilters = {
    # 2011 TauMu
    "HLT_IsoMu12_LooseIsoPFTau10_v4": ("hltFilterIsoMu12IsoPFTau10LooseIsolation","hltSingleMuIsoL3IsoFiltered12"),
    "HLT_IsoMu15_LooseIsoPFTau15_v2": ("hltPFTau15TrackLooseIso","hltSingleMuIsoL3IsoFiltered15"),
    "HLT_IsoMu15_LooseIsoPFTau15_v4": ("hltPFTau15TrackLooseIso","hltSingleMuIsoL3IsoFiltered15"),
    "HLT_IsoMu15_LooseIsoPFTau15_v5": ("hltPFTau15TrackLooseIso","hltSingleMuIsoL3IsoFiltered15"),
    "HLT_IsoMu15_LooseIsoPFTau15_v6": ("hltPFTau15TrackLooseIso","hltSingleMuIsoL3IsoFiltered15"),
    "HLT_IsoMu15_LooseIsoPFTau15_v8": ("hltPFTau15TrackLooseIso","hltSingleMuIsoL3IsoFiltered15"),
    "HLT_IsoMu15_LooseIsoPFTau15_v9": ("hltPFTau15TrackLooseIso","hltSingleMuIsoL3IsoFiltered15"),
    # 2011 TauEle Ele15_*_LooseIsoPFTau15
    "HLT_Ele15_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_LooseIsoPFTau15_v1": ( ("hltOverlapFilterIsoEle15IsoPFTau15", [15]),      "hltEle15CaloIdVTTrkIdTCaloIsoTTrkIsoTTrackIsolFilter"),
    "HLT_Ele15_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_LooseIsoPFTau15_v2": ( ("hltOverlapFilterIsoEle15IsoPFTau15", [15]),      "hltEle15CaloIdVTTrkIdTCaloIsoTTrkIsoTTrackIsolFilter"),    
    "HLT_Ele15_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_LooseIsoPFTau15_v4": ( ("hltOverlapFilterIsoEle15IsoPFTau15", [15]),      "hltEle15CaloIdVTTrkIdTCaloIsoTTrkIsoTTrackIsolFilter"),
    "HLT_Ele15_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_LooseIsoPFTau15_v6": ( ("hltOverlapFilterIsoEle15IsoPFTau15", [15]),      "hltEle15CaloIdVTCaloIsoTTrkIdTTrkIsoTTrackIsoFilter"),
    "HLT_Ele15_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_LooseIsoPFTau15_v8": ( ("hltOverlapFilterIsoEle15IsoPFTau15", [15]),      "hltEle15CaloIdVTCaloIsoTTrkIdTTrkIsoTTrackIsoFilter"),
    "HLT_Ele15_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_LooseIsoPFTau15_v9": ( ("hltOverlapFilterIsoEle15IsoPFTau15", [15]),      "hltEle15CaloIdVTCaloIsoTTrkIdTTrkIsoTTrackIsoFilter"),                                                                                                    
    # 2011 TauEle Ele15_*_TightIsoPFTau20
    "HLT_Ele15_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_TightIsoPFTau20_v1": ( ("hltOverlapFilterIsoEle15TightIsoPFTau20", [15]), "hltEle15CaloIdVTCaloIsoTTrkIdTTrkIsoTTrackIsoFilter"),
    "HLT_Ele15_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_TightIsoPFTau20_v2": ( ("hltOverlapFilterIsoEle15TightIsoPFTau20", [15]), "hltEle15CaloIdVTCaloIsoTTrkIdTTrkIsoTTrackIsoFilter"),
    "HLT_Ele18_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_TightIsoPFTau20_v2": ( ("hltOverlapFilterIsoEle18TightIsoPFTau20", [15]), "hltEle18CaloIdVTCaloIsoTTrkIdTTrkIsoTTrackIsoFilter"),
    "HLT_Ele15_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_LooseIsoPFTau20_v4": ( ("hltOverlapFilterIsoEle15IsoPFTau20", [15]),      "hltEle15CaloIdVTTrkIdTCaloIsoTTrkIsoTTrackIsolFilter"),
    "HLT_Ele15_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_LooseIsoPFTau20_v6": ( ("hltOverlapFilterIsoEle15IsoPFTau20", [15]),      "hltEle15CaloIdVTCaloIsoTTrkIdTTrkIsoTTrackIsoFilter"),
    "HLT_Ele15_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_LooseIsoPFTau20_v8": ( ("hltOverlapFilterIsoEle15IsoPFTau20", [15]),      "hltEle15CaloIdVTCaloIsoTTrkIdTTrkIsoTTrackIsoFilter"),
    "HLT_Ele15_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_LooseIsoPFTau20_v9": ( ("hltOverlapFilterIsoEle15IsoPFTau20", [15]),      "hltEle15CaloIdVTCaloIsoTTrkIdTTrkIsoTTrackIsoFilter"),
    #PG old name for the lepton filter: hltEle15CaloIdVTTrkIdTCaloIsoTTrkIsoTTrackIsolFilter 
    #PG I found the new one in the trigger menu for 10^33
    # 2011 TauEle Ele18_*_MediumIsoPFTau20 
    "HLT_Ele18_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_MediumIsoPFTau20_v1": ( ("hltOverlapFilterIsoEle18MediumIsoPFTau20", [15]), "hltEle18CaloIdVTCaloIsoTTrkIdTTrkIsoTTrackIsoFilter"),
    # 2011 TauEle Ele20_*_MediumIsoPFTau20 
    "HLT_Ele20_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_MediumIsoPFTau20_v1": ( ("hltOverlapFilterIsoEle20MediumIsoPFTau20", [15]), "hltEle20CaloIdVTCaloIsoTTrkIdTTrkIsoTTrackIsoFilterL1SingleEG18orL1SingleEG20"),
    "HLT_Ele20_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_MediumIsoPFTau20_v5": ( ("hltOverlapFilterIsoEle20MediumIsoPFTau20", [15]), "hltEle20CaloIdVTCaloIsoTTrkIdTTrkIsoTTrackIsoFilterL1SingleEG18orL1SingleEG20"),
    "HLT_Ele20_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_MediumIsoPFTau20_v6": ( ("hltOverlapFilterIsoEle20MediumIsoPFTau20", [15]), "hltEle20CaloIdVTCaloIsoTTrkIdTTrkIsoTTrackIsoFilterL1SingleEG18orL1SingleEG20"),
    "HLT_Ele20_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_MediumIsoPFTau20_v7": ( ("hltOverlapFilterIsoEle20MediumIsoPFTau20", [15]), "hltEle20CaloIdVTCaloIsoTTrkIdTTrkIsoTTrackIsoFilterL1SingleEG18orL1SingleEG20"),
    # the following is used for synchronization
    "HLT_IsoMu15_eta2p1_LooseIsoPFTau20_v1": ("hltPFTau20TrackLooseIso","hltSingleMuIsoL3IsoFiltered15"), 
    "HLT_IsoMu15_eta2p1_LooseIsoPFTau20_v1": ("hltPFTau20TrackLooseIso","hltSingleMuIsoL1s14L3IsoFiltered15eta2p1"),
    "HLT_IsoMu15_eta2p1_LooseIsoPFTau20_v5": ("hltPFTau20TrackLooseIso","hltSingleMuIsoL1s14L3IsoFiltered15eta2p1"),
    "HLT_IsoMu15_eta2p1_LooseIsoPFTau20_v6": ("hltPFTau20TrackLooseIso","hltSingleMuIsoL1s14L3IsoFiltered15eta2p1"),
    "HLT_Ele18_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_MediumIsoPFTau20_v1": ("hltPFTauMediumIso20TrackMediumIso","hltEle18CaloIdVTCaloIsoTTrkIdTTrkIsoTTrackIsoFilter"),
    #
    # 2012 TauEle
    # 
    # for 2012A and 52 MC: 
    'HLT_Ele20_CaloIdVT_CaloIsoRhoT_TrkIdT_TrkIsoT_LooseIsoPFTau20_v*': ( ('hltOverlapFilterIsoEle20LooseIsoPFTau20', [15,0]),
                                                                          'hltEle20CaloIdVTCaloIsoTTrkIdTTrkIsoTTrackIsoFilterL1IsoEG18OrEG20'),
    #
    # 2012B and 53 MC
    #
    'HLT_Ele22_eta2p1_WP90Rho_LooseIsoPFTau20_v*': ( ('hltOverlapFilterIsoEle20WP90LooseIsoPFTau20', [15,0]),
                                                     'hltEle22WP90RhoTrackIsoFilter' ),
    #'HLT_Ele20_CaloIdVT_CaloIsoRhoT_TrkIdT_TrkIsoT_LooseIsoPFTau20_v*': ( ('hltOverlapFilterIsoEle20LooseIsoPFTau20', [15,0]),
    #                                                                      ('hltOverlapFilterIsoEle20LooseIsoPFTau20',[11])),    
    # note: pdgId for taus incorrectly set to 0...
    'HLT_IsoMu18_eta2p1_LooseIsoPFTau20_v*': ( ('hltOverlapFilterIsoMu18LooseIsoPFTau20', [15,0]), ('hltOverlapFilterIsoMu18LooseIsoPFTau20',[13])),
    #COLIN FIXME
    'HLT_IsoMu17_eta2p1_LooseIsoPFTau20_v*': ( ('hltOverlapFilterIsoMu17LooseIsoPFTau20', [15,0]), ('hltOverlapFilterIsoMu17LooseIsoPFTau20',[13])),
    # 'HLT_IsoMu17_eta2p1_LooseIsoPFTau20_v*': ( ('hltOverlapFilterIsoMu17LooseIsoPFTau20', [15,0]), 'hltL3crIsoL1sMu14erORMu16erL1f0L2f14QL3f17QL3crIsoRhoFiltered0p15'),
    # hadronic tau triggers 2011
    'HLT_DoubleIsoPFTau20_Trk5_v1': ("hltFilterDoubleIsoPFTau25Trk5LeadTrack5IsolationL1HLTMatched","hltFilterDoubleIsoPFTau25Trk5LeadTrack5IsolationL1HLTMatched"),
    'HLT_DoubleIsoPFTau20_Trk5_v2': ("hltFilterDoubleIsoPFTau25Trk5LeadTrack5IsolationL1HLTMatched","hltFilterDoubleIsoPFTau25Trk5LeadTrack5IsolationL1HLTMatched"),
    'HLT_DoubleIsoPFTau20_Trk5_v3': ("hltFilterDoubleIsoPFTau25Trk5LeadTrack5IsolationL1HLTMatched","hltFilterDoubleIsoPFTau25Trk5LeadTrack5IsolationL1HLTMatched"),
    'HLT_DoubleIsoPFTau20_Trk5_v4': ("hltFilterDoubleIsoPFTau25Trk5LeadTrack5IsolationL1HLTMatched","hltFilterDoubleIsoPFTau25Trk5LeadTrack5IsolationL1HLTMatched"),
    'HLT_DoubleIsoPFTau25_Trk5_eta2p1_v1': ("hltFilterDoubleIsoPFTau30Trk5LeadTrack5IsolationL1HLTMatched","hltFilterDoubleIsoPFTau30Trk5LeadTrack5IsolationL1HLTMatched"),
    'HLT_DoubleIsoPFTau25_Trk5_eta2p1_v2': ("hltFilterDoubleIsoPFTau30Trk5LeadTrack5IsolationL1HLTMatched","hltFilterDoubleIsoPFTau30Trk5LeadTrack5IsolationL1HLTMatched"),
    'HLT_DoubleIsoPFTau25_Trk5_eta2p1_v3': ("hltFilterDoubleIsoPFTau30Trk5LeadTrack5IsolationL1HLTMatched","hltFilterDoubleIsoPFTau30Trk5LeadTrack5IsolationL1HLTMatched"),
    'HLT_DoubleIsoPFTau25_Trk5_eta2p1_v4': ("hltFilterDoubleIsoPFTau30Trk5LeadTrack5IsolationL1HLTMatched","hltFilterDoubleIsoPFTau30Trk5LeadTrack5IsolationL1HLTMatched"),
    'HLT_DoubleIsoPFTau35_Trk5_eta2p1_v1': ("hltFilterDoubleIsoPFTau35Trk5LeadTrack5IsolationL1HLTMatched","hltFilterDoubleIsoPFTau35Trk5LeadTrack5IsolationL1HLTMatched"),
    'HLT_DoubleIsoPFTau35_Trk5_eta2p1_v2': ("hltFilterDoubleIsoPFTau35Trk5LeadTrack5IsolationL1HLTMatched","hltFilterDoubleIsoPFTau35Trk5LeadTrack5IsolationL1HLTMatched"),
    'HLT_DoubleIsoPFTau35_Trk5_eta2p1_v3': ("hltFilterDoubleIsoPFTau35Trk5LeadTrack5IsolationL1HLTMatched","hltFilterDoubleIsoPFTau35Trk5LeadTrack5IsolationL1HLTMatched"),
    'HLT_DoubleIsoPFTau35_Trk5_eta2p1_v4': ("hltFilterDoubleIsoPFTau35Trk5LeadTrack5IsolationL1HLTMatched","hltFilterDoubleIsoPFTau35Trk5LeadTrack5IsolationL1HLTMatched"),
    # hadronic tau triggers with jet 2012
    'HLT_DoubleMediumIsoPFTau25_Trk5_eta2p1_Jet30_v1': ("hltDoublePFTau25TrackPt5MediumIsolationProng4Dz02","hltDoublePFTau25TrackPt5MediumIsolationProng4Dz02","hltTripleL2Jets30eta3"),
    'HLT_DoubleMediumIsoPFTau25_Trk5_eta2p1_Jet30_v2': ("hltDoublePFTau25TrackPt5MediumIsolationProng4Dz02","hltDoublePFTau25TrackPt5MediumIsolationProng4Dz02","hltTripleL2Jets30eta3"),
    'HLT_DoubleMediumIsoPFTau25_Trk5_eta2p1_Jet30_v3': ("hltDoublePFTau25TrackPt5MediumIsolationProng4Dz02","hltDoublePFTau25TrackPt5MediumIsolationProng4Dz02","hltTripleL2Jets30eta3"),
    'HLT_DoubleMediumIsoPFTau25_Trk5_eta2p1_Jet30_v4': ("hltDoublePFTau25TrackPt5MediumIsolationProng4Dz02","hltDoublePFTau25TrackPt5MediumIsolationProng4Dz02","hltTripleL2Jets30eta3"),

    'HLT_DoubleMediumIsoPFTau30_Trk5_eta2p1_Jet30_v1': ("hltDoublePFTau30TrackPt5MediumIsolationProng4Dz02","hltDoublePFTau30TrackPt5MediumIsolationProng4Dz02","hltTripleL2Jets30eta3"),
    'HLT_DoubleMediumIsoPFTau30_Trk5_eta2p1_Jet30_v2': ("hltDoublePFTau30TrackPt5MediumIsolationProng4Dz02","hltDoublePFTau30TrackPt5MediumIsolationProng4Dz02","hltTripleL2Jets30eta3"),
    'HLT_DoubleMediumIsoPFTau30_Trk5_eta2p1_Jet30_v3': ("hltDoublePFTau30TrackPt5MediumIsolationProng4Dz02","hltDoublePFTau30TrackPt5MediumIsolationProng4Dz02","hltTripleL2Jets30eta3"),
    'HLT_DoubleMediumIsoPFTau30_Trk5_eta2p1_Jet30_v4': ("hltDoublePFTau30TrackPt5MediumIsolationProng4Dz02","hltDoublePFTau30TrackPt5MediumIsolationProng4Dz02","hltTripleL2Jets30eta3"),

    'HLT_DoubleMediumIsoPFTau30_Trk1_eta2p1_Jet30_v1': ("hltDoublePFTau30TrackPt1MediumIsolationProng4Dz02","hltDoublePFTau30TrackPt1MediumIsolationProng4Dz02","hltTripleL2Jets30eta3"),
    'HLT_DoubleMediumIsoPFTau30_Trk1_eta2p1_Jet30_v2': ("hltDoublePFTau30TrackPt1MediumIsolationProng4Dz02","hltDoublePFTau30TrackPt1MediumIsolationProng4Dz02","hltTripleL2Jets30eta3"),
    'HLT_DoubleMediumIsoPFTau30_Trk1_eta2p1_Jet30_v3': ("hltDoublePFTau30TrackPt1MediumIsolationProng4Dz02","hltDoublePFTau30TrackPt1MediumIsolationProng4Dz02","hltTripleL2Jets30eta3"),
    'HLT_DoubleMediumIsoPFTau30_Trk1_eta2p1_Jet30_v4': ("hltDoublePFTau30TrackPt1MediumIsolationProng4Dz02","hltDoublePFTau30TrackPt1MediumIsolationProng4Dz02","hltTripleL2Jets30eta3"),
    'HLT_DoubleMediumIsoPFTau30_Trk1_eta2p1_Jet30_v5': ("hltDoublePFTau30TrackPt1MediumIsolationProng4Dz02","hltDoublePFTau30TrackPt1MediumIsolationProng4Dz02","hltTripleL2Jets30eta3"),
    'HLT_DoubleMediumIsoPFTau30_Trk1_eta2p1_Reg_Jet30_v1': ("hltDoublePFTau30TrackPt1MediumIsolationProng4Dz02Reg","hltDoublePFTau30TrackPt1MediumIsolationProng4Dz02Reg","hltTripleL2Jets30eta3"),
    # hadronic tau triggers with 1 prong 2012
    'HLT_DoubleMediumIsoPFTau35_Trk5_eta2p1_Prong1_v1': ("hltDoublePFTau35TrackPt5MediumIsolationProng2Dz02","hltDoublePFTau35TrackPt5MediumIsolationProng2Dz02"),
    'HLT_DoubleMediumIsoPFTau35_Trk5_eta2p1_Prong1_v2': ("hltDoublePFTau35TrackPt5MediumIsolationProng2Dz02","hltDoublePFTau35TrackPt5MediumIsolationProng2Dz02"),
    'HLT_DoubleMediumIsoPFTau35_Trk5_eta2p1_Prong1_v3': ("hltDoublePFTau35TrackPt5MediumIsolationProng2Dz02","hltDoublePFTau35TrackPt5MediumIsolationProng2Dz02"),
    'HLT_DoubleMediumIsoPFTau35_Trk5_eta2p1_Prong1_v4': ("hltDoublePFTau35TrackPt5MediumIsolationProng2Dz02","hltDoublePFTau35TrackPt5MediumIsolationProng2Dz02"),
    'HLT_DoubleMediumIsoPFTau35_Trk5_eta2p1_Prong1_v5': ("hltDoublePFTau35TrackPt5MediumIsolationProng2Dz02","hltDoublePFTau35TrackPt5MediumIsolationProng2Dz02"),
    'HLT_DoubleMediumIsoPFTau35_Trk5_eta2p1_Prong1_v6': ("hltDoublePFTau35TrackPt5MediumIsolationProng2Dz02","hltDoublePFTau35TrackPt5MediumIsolationProng2Dz02"),

    'HLT_DoubleMediumIsoPFTau35_Trk1_eta2p1_Prong1_v1': ("hltDoublePFTau35TrackPt1MediumIsolationProng2Dz02","hltDoublePFTau35TrackPt1MediumIsolationProng2Dz02"),
    'HLT_DoubleMediumIsoPFTau35_Trk1_eta2p1_Prong1_v2': ("hltDoublePFTau35TrackPt1MediumIsolationProng2Dz02","hltDoublePFTau35TrackPt1MediumIsolationProng2Dz02"),
    'HLT_DoubleMediumIsoPFTau35_Trk1_eta2p1_Prong1_v3': ("hltDoublePFTau35TrackPt1MediumIsolationProng2Dz02","hltDoublePFTau35TrackPt1MediumIsolationProng2Dz02"),
    'HLT_DoubleMediumIsoPFTau35_Trk1_eta2p1_Prong1_v4': ("hltDoublePFTau35TrackPt1MediumIsolationProng2Dz02","hltDoublePFTau35TrackPt1MediumIsolationProng2Dz02"),
    'HLT_DoubleMediumIsoPFTau35_Trk1_eta2p1_Prong1_Reg_v1': ("hltDoublePFTau35TrackPt1MediumIsolationProng2Dz02Reg","hltDoublePFTau35TrackPt1MediumIsolationProng2Dz02Reg"),
    # hadronic tau triggers parked 2012
    ## first part with trk5
    'HLT_DoubleMediumIsoPFTau35_Trk5_eta2p1_v2': ("hltDoublePFTau35TrackPt5MediumIsolationProng4Dz02","hltDoublePFTau35TrackPt5MediumIsolationProng4Dz02"),
    'HLT_DoubleMediumIsoPFTau35_Trk5_eta2p1_v3': ("hltDoublePFTau35TrackPt5MediumIsolationProng4Dz02","hltDoublePFTau35TrackPt5MediumIsolationProng4Dz02"),
    'HLT_DoubleMediumIsoPFTau35_Trk5_eta2p1_v4': ("hltDoublePFTau35TrackPt5MediumIsolationProng4Dz02","hltDoublePFTau35TrackPt5MediumIsolationProng4Dz02"),
    'HLT_DoubleMediumIsoPFTau35_Trk5_eta2p1_v6': ("hltDoublePFTau35TrackPt5MediumIsolationProng4Dz02","hltDoublePFTau35TrackPt5MediumIsolationProng4Dz02"),
    ## second part with trk1
    'HLT_DoubleMediumIsoPFTau35_Trk1_eta2p1_v1': ("hltDoublePFTau35TrackPt1MediumIsolationProng4Dz02","hltDoublePFTau35TrackPt1MediumIsolationProng4Dz02"),
    'HLT_DoubleMediumIsoPFTau35_Trk1_eta2p1_v3': ("hltDoublePFTau35TrackPt1MediumIsolationProng4Dz02","hltDoublePFTau35TrackPt1MediumIsolationProng4Dz02"),
    'HLT_DoubleMediumIsoPFTau35_Trk1_eta2p1_v4': ("hltDoublePFTau35TrackPt1MediumIsolationProng4Dz02","hltDoublePFTau35TrackPt1MediumIsolationProng4Dz02"),
    # Embedded triggers: No filters
    'HLT_Mu17_Mu8_v16':('', ''),
    'HLT_Mu17_Mu8_v17':('', ''),
    'HLT_Mu17_Mu8_v18':('', ''),
    'HLT_Mu17_Mu8_v19':('', ''),
    'HLT_Mu17_Mu8_v21':('', ''),
    'HLT_Mu17_Mu8_v22':('', ''),
    }
    

if __name__ == '__main__':

    for path in sorted(pathsAndFilters):   
        print 'path:', path
        filters = pathsAndFilters[path]
        print '\tleg1 filter:', filters[0]
        print '\tleg2 filter:', filters[1]
        
   
