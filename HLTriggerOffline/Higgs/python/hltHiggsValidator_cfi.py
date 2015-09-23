import FWCore.ParameterSet.Config as cms


hltHiggsValidator = cms.EDAnalyzer("HLTHiggsValidator",
        
    hltProcessName = cms.string("HLT"),
    analysis       = cms.vstring("HWW", "HZZ", "Hgg", "HggControlPaths", "Htaunu", "H2tau", "VBFHbb_0btag", "VBFHbb_1btag", "VBFHbb_2btag",  "ZnnHbb","DoubleHinTaus","HiggsDalitz","X4b","TTHbbej","AHttH","WHToENuBB","MSSMHbb"),
    histDirectory  = cms.string("HLT/Higgs"),
    
    # -- The instance name of the reco::GenParticles collection 
    genParticleLabel = cms.string("genParticles"),
    
    # -- The instance name of the reco::GenJets collection
    genJetLabel = cms.string("ak4GenJets"),

    # -- The instance name of the reco::PFJetCollection collection
    recoHtJetLabel = cms.untracked.string("ak4PFJetsCHS"),

    # -- The number of interactions in the event
    pileUpInfoLabel  = cms.string("addPileupInfo"),

    # -- The binning of the Pt efficiency plots
    parametersTurnOn = cms.vdouble(0,
                                1, 8, 9, 10,
                                11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                22, 24, 26, 28, 30, 32, 34, 36, 38, 40,
                                45, 50, 55, 60, 65, 70,
                                80, 100,
                                ),

    # -- (NBins, minVal, maxValue) for the Eta,Phi and nInterations efficiency plots
    parametersEta      = cms.vdouble(48, -2.400, 2.400),
    parametersPhi      = cms.vdouble(50, -3.142, 3.142),
    parametersPu       = cms.vdouble(10, 0, 50),
    parametersHt       = cms.untracked.vdouble(100, 0, 1000),

    # TO BE DEPRECATED --------------------------------------------
    cutsDr = cms.vdouble(0.4, 0.4, 0.015), # TO BE DEPRECATED
    # parameters for attempting an L1 match using a propagator
    maxDeltaPhi = cms.double(0.4),  # TO BE DEPRECATED
    maxDeltaR   = cms.double(0.4),  # TO BE DEPRECATED
    # TO BE DEPRECATED --------------------------------------------

    # Definition of generic cuts on generated and reconstructed objects (note that
    # these cuts can be overloaded inside a particular analysis)
    # Objects recognized: Mu Ele Photon PFTau MET
    # Syntax in the strings: valid syntax of the StringCutObjectSelector class
    # --- Muons
    Mu_genCut     = cms.string("pt > 10 && abs(eta) < 2.4 && abs(pdgId) == 13 && status == 1"),
    Mu_recCut     = cms.string("pt > 10 && abs(eta) < 2.4 && isGlobalMuon"),
    Mu_cutMinPt   = cms.double(10),  # TO BE DEPRECATED
    Mu_cutMaxEta  = cms.double(2.4), # TO BE DEPRECATED
    
    # --- Electrons
    Ele_genCut      = cms.string("pt > 10 && abs(eta) < 2.5 && abs(pdgId) == 11 && status == 1"),
    Ele_recCut      = cms.string("pt > 10 && abs(eta) < 2.5 && hadronicOverEm < 0.05 && eSuperClusterOverP > 0.5 && eSuperClusterOverP < 2.5"),
    Ele_cutMinPt    = cms.double(10),  # TO BE DEPRECATED
    Ele_cutMaxEta   = cms.double(2.5), # TO BE DEPRECATED

    # --- Photons
    Photon_genCut     = cms.string("abs(pdgId) == 22 && status == 1"),
    Photon_recCut     = cms.string("pt > 20 && abs(eta) < 2.4 && hadronicOverEm < 0.1 && "+\
                                   " ( ( abs(eta) < 1.479 && r9 > 0.85 ) || "+\
                                   "   ( abs(eta) > 1.479 && r9 > 0.90 ) || "+\
                                   "   ( abs(eta) < 1.479 && r9 > 0.50 && sigmaIetaIeta < 0.014 && "+\
                                   "     ecalRecHitSumEtConeDR03 < (6.0+0.012*et) && trkSumPtSolidConeDR03 < (6.0 + 0.002*et) ) || "+\
                                   "   ( abs(eta) > 1.479 && r9 > 0.80 && sigmaIetaIeta < 0.035 && "+\
                                   "     ecalRecHitSumEtConeDR03 < (6.0+0.012*et) && trkSumPtSolidConeDR03 < (6.0 + 0.002*et) ) ) "
                                   ),
    Photon_cutMinPt   = cms.double(20), # TO BE DEPRECATED
    Photon_cutMaxEta  = cms.double(2.4),# TO BE DEPRECATED

    # --- Taus: 
    PFTau_genCut      = cms.string("pt > 20 && abs(eta) < 2.4 && abs(pdgId) == 15 && status == 3"),
    PFTau_recCut      = cms.string("pt > 20 && abs(eta) < 2.4"),  # STILL MISSING THIS INFO
    PFTau_cutMinPt    = cms.double(20), # TO BE DEPRECATED
    PFTau_cutMaxEta   = cms.double(2.5),# TO BE DEPRECATED

    # --- MET (calo)    
    MET_genCut      = cms.string("(abs(pdgId) == 12 || abs(pdgId)==14 || abs(pdgId) == 16 ) && status == 1"),
    MET_recCut      = cms.string("pt > 75."),  
    MET_cutMinPt    = cms.double(75), # TO BE DEPRECATED
    MET_cutMaxEta   = cms.double(0),  # TO BE DEPRECATED
    
    # --- PFMET    
    PFMET_genCut      = cms.string("(abs(pdgId) == 12 || abs(pdgId)==14 || abs(pdgId) == 16 ) && status == 1"),
    PFMET_recCut      = cms.string("pt > 75."),  
    PFMET_cutMinPt    = cms.double(75), # TO BE DEPRECATED
    PFMET_cutMaxEta   = cms.double(0),  # TO BE DEPRECATED
    
    # --- Jets: 
    Jet_genCut      = cms.string("pt > 10"),
    Jet_recCut      = cms.string("pt > 10"),  
    Jet_cutMinPt    = cms.double(0), # TO BE DEPRECATED
    Jet_cutMaxEta   = cms.double(0),  # TO BE DEPRECATED
    
    

    # The specific parameters per analysis: the name of the parameter set has to be 
    # the same as the defined ones in the 'analysis' datamember. Each analysis is a PSet
    # with the mandatory attributes:
    #    - hltPathsToCheck (cms.vstring) : a list of all the trigger pats to be checked 
    #                 in this analysis. Up to the version number _v, but not including 
    #                 the number in order to avoid this version dependence. Example: HLT_Mu18_v
    #    - recVarLabel (cms.string): where Var can be Muon, Elec, Photon, CaloMET, PFTau, Jet. This 
    #                 attribute is the name of the INSTANCE LABEL for each RECO collection to 
    #                 be considered in the analysis. Note that the trigger paths rely on some 
    #                 objects which need to be defined here, otherwise the code will complain. 
    #    - minCandidates (cms.uint32): the minimum number of GEN/RECO objects in the event
    # Besides the mandatory attributes, you can redefine the generation and reconstruction cuts
    # for any object you want.
    #    * Var_genCut, Var_recCut (cms.string): where Var=Mu, Ele, Photon, MET, PFTau (see above)

    HWW = cms.PSet( 
        hltPathsToCheck = cms.vstring(
              #dileptons for Hww and Hzz
              "HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ_v",
              "HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_v",
              "HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_v",
              "HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_v",
              "HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL_DZ_v",
              #dilepton path for the 7e33 menu at 25ns
              "HLT_Ele17_Ele12_CaloIdL_TrackIdL_IsoVL_DZ_v",
              "HLT_Mu8_TrkIsoVVL_Ele17_CaloIdL_TrackIdL_IsoVL_v",
              "HLT_Mu17_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_v",
              #prescaled control paths
              "HLT_Ele17_CaloIdL_TrackIdL_IsoVL_v",
              "HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_v",
              "HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_v",
              "HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL_v",
              "HLT_Ele23_CaloIdL_TrackIdL_IsoVL_v",
              "HLT_Ele12_CaloIdL_TrackIdL_IsoVL_v",
              "HLT_Ele17_Ele12_CaloIdL_TrackIdL_IsoVL_v"  
        ),
        recMuonLabel  = cms.string("muons"),
        recElecLabel  = cms.string("gedGsfElectrons"),
        # -- Analysis specific cuts
        minCandidates = cms.uint32(2),
        ),
    HZZ = cms.PSet( 
        hltPathsToCheck = cms.vstring(
        #tri-leptons for Hzz
            "HLT_Ele16_Ele12_Ele8_CaloIdL_TrackIdL_v",
            "HLT_Mu8_DiEle12_CaloIdL_TrackIdL_v",
            "HLT_DiMu9_Ele9_CaloIdL_TrackIdL_v",
            "HLT_TripleMu_12_10_5_v"
        ),
        recMuonLabel  = cms.string("muons"),
        recElecLabel  = cms.string("gedGsfElectrons"),
        #recTrackLabel = cms.string("globalMuons"),
        # -- Analysis specific cuts
        minCandidates = cms.uint32(4), 
        ),
    Hgg = cms.PSet( 
        hltPathsToCheck = cms.vstring(
            "HLT_Diphoton30_18_R9Id_OR_IsoCaloId_AND_HE_R9Id_Mass95_v",
            "HLT_Diphoton30PV_18PV_R9Id_AND_IsoCaloId_AND_HE_R9Id_DoublePixelVeto_Mass55_v",
            "HLT_Diphoton30_18_Solid_R9Id_AND_IsoCaloId_AND_HE_R9Id_Mass55_v",
            "HLT_Diphoton30EB_18EB_R9Id_OR_IsoCaloId_AND_HE_R9Id_DoublePixelVeto_Mass55_v",
            
            # frozen menu paths
            "HLT_Diphoton44_28_R9Id85_OR_Iso50T80LCaloId24b40e_AND_HE10P1_R9Id50b80e_v",
            "HLT_Diphoton30_18_R9Id85_OR_Iso50T80LCaloId24b40e_AND_HE10P0_R9Id50b80e_Mass95_v",
            "HLT_Diphoton28_14_R9Id85_OR_Iso50T80LCaloId24b40e_AND_HE10P0_R9Id50b80e_Mass50_Eta_1p5_v",
            "HLT_Diphoton30_18_R9Id85_AND_Iso50T80LCaloId24b40e_AND_HE10P0_R9Id50b80e_Solid_Mass30_v",
            "HLT_Diphoton30_18_R9Id85_AND_Iso50T80LCaloId24b40e_AND_HE10P0_R9Id50b80e_PV_v",
            "HLT_Diphoton30_18_R9Id85_AND_Iso50T80LCaloId24b40e_AND_HE10P0_R9Id50b80e_DoublePV_v"
        ),
        recPhotonLabel  = cms.string("photons"),
        # -- Analysis specific cuts
        minCandidates = cms.uint32(2), 
        ),
    # seperate directory because it needs a different relval    
    HggControlPaths = cms.PSet( 
        hltPathsToCheck = cms.vstring(
            "HLT_Diphoton30_18_R9Id_OR_IsoCaloId_AND_HE_R9Id_DoublePixelSeedMatch_Mass70_v",
            # frozen menu paths
            "HLT_Diphoton30_18_R9Id85_OR_Iso50T80LCaloId24b40e_AND_HE10P0_R9Id50b80e_PixelSeed_Mass70_v"
        ),
        recPhotonLabel  = cms.string("photons"),
        # -- Analysis specific cuts
        minCandidates = cms.uint32(2), 
        ),
     DoubleHinTaus = cms.PSet(
        hltPathsToCheck = cms.vstring(
            "HLT_Mu17_Mu8_v",
            "HLT_Mu17_Mu8_DZ_v",
            "HLT_Mu17_Mu8_SameSign_DZ_v",
            "HLT_Mu20_Mu10_v",
            "HLT_Mu20_Mu10_DZ_v",
            "HLT_Mu20_Mu10_SameSign_DZ_v",
            
            # frozen menu paths
            "HLT_Mu17_Mu8_SameSign_v",
            "HLT_Mu17_Mu8_SameSign_DPhi_v"
        ),
        recMuonLabel  = cms.string("muons"),
        # -- Analysis specific cuts
        minCandidates = cms.uint32(2),
        ),
     HiggsDalitz = cms.PSet(
        hltPathsToCheck = cms.vstring(
            "HLT_Mu17_Photon22_CaloIdL_L1ISO_v",
            "HLT_Mu17_Photon30_CaloIdL_L1ISO_v",
            "HLT_Mu12_Photon25_CaloIdL_v",
            "HLT_Mu12_Photon25_CaloIdL_L1ISO_v",
            "HLT_Mu12_Photon25_CaloIdL_L1OR_v"
        ),
        recMuonLabel  = cms.string("muons"),
        recPhotonLabel  = cms.string("photons"),
        # -- Analysis specific cuts
        minCandidates = cms.uint32(2),
        ),
     Htaunu = cms.PSet(
        hltPathsToCheck = cms.vstring(
            "HLT_LooseIsoPFTau50_Trk30_eta2p1_MET80_JetIdCleaned_v",
            "HLT_LooseIsoPFTau50_Trk30_eta2p1_MET120_JetIdCleaned_v",
            "HLT_LooseIsoPFTau50_Trk30_eta2p1_v",
            
            # frozen menu paths
            "HLT_LooseIsoPFTau50_Trk30_eta2p1_MET80_v",
            "HLT_LooseIsoPFTau50_Trk30_eta2p1_MET120_v",            
            "HLT_IsoMu16_eta2p1_CaloMET30_LooseIsoPFTau50_Trk30_eta2p1_v",
            "HLT_IsoMu16_eta2p1_CaloMET30_v"
            ),
        recPFTauLabel   = cms.string("hpsPFTauProducer"),
        recCaloMETLabel = cms.string("caloMet"),
        recMuonLabel  = cms.string("muons"),
        # -- Analysis specific cuts
        minCandidates = cms.uint32(1), 
        parametersTurnOn = cms.vdouble(0,
                                1, 8, 9, 10,
                                11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                22, 24, 26, 28, 30, 32, 34, 36, 38, 40,
                                45, 50, 55, 60, 65, 70, 
                                80, 100, 120, 140, 160, 180, 200,
                                ),
        ),
    H2tau  = cms.PSet( 
        hltPathsToCheck = cms.vstring(
            "HLT_IsoMu17_eta2p1_LooseIsoPFTau20_v",
            "HLT_DoubleMediumIsoPFTau40_Trk1_eta2p1_Reg_v",
            "HLT_IsoMu17_eta2p1_LooseIsoPFTau20_SingleL1_v",
            "HLT_IsoMu17_eta2p1_MediumIsoPFTau40_Trk1_eta2p1_Reg_v",
            "HLT_IsoMu17_eta2p1_v",
            "HLT_DoubleIsoMu17_eta2p1_v",
            "HLT_IsoMu16_eta2p1_CaloMET30_v",
            "HLT_Mu16_eta2p1_CaloMET30_v",
            "HLT_Ele27_eta2p1_WPLoose_Gsf_DoubleMediumIsoPFTau40_Trk1_eta2p1_Reg_v",
            "HLT_Ele32_eta2p1_WPLoose_Gsf_DoubleMediumIsoPFTau40_Trk1_eta2p1_Reg_v",
            "HLT_Ele27_eta2p1_WPLoose_Gsf_LooseIsoPFTau20_v",
            "HLT_Ele32_eta2p1_WPLoose_Gsf_LooseIsoPFTau20_v",
            "HLT_Ele22_eta2p1_WPLoose_Gsf_v",
            "HLT_Ele22_eta2p1_WPTight_Gsf_v",
            "HLT_DoubleEle24_22_eta2p1_WPLoose_Gsf_v",
            "HLT_IsoMu24_eta2p1_LooseIsoPFTau20_v",
            "HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_v",
            "HLT_Ele22_eta2p1_WPLoose_Gsf_LooseIsoPFTau20_v",
            "HLT_Ele27_eta2p1_WPLoose_Gsf_v",
            "HLT_Ele27_eta2p1_WPTight_Gsf_v",
            "HLT_Ele32_eta2p1_WPLoose_Gsf_v",
            "HLT_Ele32_eta2p1_WPTight_Gsf_v",
            ),
        recPFTauLabel  = cms.string("hpsPFTauProducer"),
        recMuonLabel   = cms.string("muons"),
        recElecLabel   = cms.string("gedGsfElectrons"),
        recCaloMETLabel = cms.string("caloMet"),
        # -- Analysis specific cuts
        minCandidates = cms.uint32(2), 
        ),
    VBFHbb_0btag  = cms.PSet( 
        hltPathsToCheck = cms.vstring(
            "HLT_QuadPFJet_VBF_v",
            "HLT_L1_TripleJet_VBF_v"
            ),
        recJetLabel  = cms.string("ak4PFJetsCHS"),
        jetTagLabel  = cms.string("pfCombinedSecondaryVertexV2BJetTags"),
        # -- Analysis specific cuts
        minCandidates = cms.uint32(4), 
        NminOneCuts = cms.untracked.vdouble(2.5, 240, 2.1, 0, 0, 0, 0, 0, 0, 95, 85, 70, 40), #dEtaqq, mqq, dPhibb, CSV1, CSV2, CSV3, maxCSV_jets, maxCSV_E, MET, pt1, pt2, pt3, pt4
        ),
    VBFHbb_2btag  = cms.PSet( 
        hltPathsToCheck = cms.vstring(
            "HLT_QuadPFJet_DoubleBTagCSV_VBF_Mqq200_v",
            "HLT_QuadPFJet_DoubleBTagCSV_VBF_Mqq240_v",
            ),
        recJetLabel  = cms.string("ak4PFJetsCHS"),
        jetTagLabel  = cms.string("pfCombinedSecondaryVertexV2BJetTags"),
        # -- Analysis specific cuts
        minCandidates = cms.uint32(4), 
        NminOneCuts = cms.untracked.vdouble(2.5, 240, 2.1, 0.8, 0.5, 0, 0, 0, 0, 95, 85, 70, 40), #dEtaqq, mqq, dPhibb, CSV1, CSV2, CSV3, maxCSV_jets, maxCSV_E, MET, pt1, pt2, pt3, pt4
        ),
    VBFHbb_1btag  = cms.PSet( 
        hltPathsToCheck = cms.vstring(
            "HLT_QuadPFJet_SingleBTagCSV_VBF_Mqq460_v",
            "HLT_QuadPFJet_SingleBTagCSV_VBF_Mqq500_v",
            ),
        recJetLabel  = cms.string("ak4PFJetsCHS"),
        jetTagLabel  = cms.string("pfCombinedSecondaryVertexV2BJetTags"),
        # -- Analysis specific cuts
        minCandidates = cms.uint32(4), 
        NminOneCuts = cms.untracked.vdouble(5, 550, 1.0, 0.8, 0, 0, 0, 0, 0, 95, 85, 70, 40), #dEtaqq, mqq, dPhibb, CSV1, CSV2, CSV3, maxCSV_jets, maxCSV_E, MET, pt1, pt2, pt3, pt4
        ),
    ZnnHbb = cms.PSet( 
        hltPathsToCheck = cms.vstring(
            "HLT_CaloMHTNoPU90_PFMET90_PFMHT90_IDTight_BTagCSV0p72_v",
            "HLT_CaloMHTNoPU90_PFMET90_PFMHT90_IDTight_v"
            "HLT_PFMET120_PFMHT120_IDTight_v",
            "HLT_PFMET110_PFMHT110_IDTight_v",
            "HLT_PFMET100_PFMHT100_IDTight_v",
            "HLT_PFMET90_PFMHT90_IDTight_v",
            
            # frozen menu paths
            "HLT_CaloMHTNoPU90_PFMET90_PFMHT90_IDLoose_BTagCSV0p7_v",
            "HLT_CaloMHTNoPU90_PFMET90_PFMHT90_IDLoose_v"
            "HLT_PFMET120_PFMHT120_IDLoose_v",
            "HLT_PFMET110_PFMHT110_IDLoose_v",
            "HLT_PFMET100_PFMHT100_IDLoose_v",
            "HLT_PFMET90_PFMHT90_IDLoose_v",
            ),
        Jet_recCut   = cms.string("pt > 10 && abs(eta) < 2.6"),
        recJetLabel  = cms.string("ak4PFJetsCHS"),
        jetTagLabel  = cms.string("pfCombinedSecondaryVertexV2BJetTags"),
        recPFMETLabel = cms.string("pfMet"),  
        # -- Analysis specific cuts
        minCandidates = cms.uint32(1), 
        NminOneCuts = cms.untracked.vdouble(0, 0, 0, 0.9, 0, 0, 8, 30, 100, 70), #dEtaqq, mqq, dPhibb, CSV1, CSV2, CSV3, maxCSV_jets, maxCSV_E, MET, pt1, pt2, pt3, pt4
        ),
    X4b  = cms.PSet( 
        hltPathsToCheck = cms.vstring(
            "HLT_DoubleJet90_Double30_TripleBTagCSV0p67_v",
            "HLT_DoubleJet90_Double30_DoubleBTagCSV0p67_v",
            "HLT_QuadJet45_TripleBTagCSV0p67_v",
            "HLT_QuadJet45_DoubleBTagCSV0p67_v",
            
            # frozen menu paths
            "HLT_DoubleJet90_Double30_TripleCSV0p5_v",
            "HLT_DoubleJet90_Double30_DoubleCSV0p5_v",
            "HLT_QuadJet45_TripleCSV0p5_v",
            "HLT_QuadJet45_DoubleCSV0p5_v"
            ),
        recJetLabel  = cms.string("ak4PFJetsCHS"),
        jetTagLabel  = cms.string("pfCombinedSecondaryVertexV2BJetTags"),
        # -- Analysis specific cuts
        minCandidates = cms.uint32(4), 
        NminOneCuts = cms.untracked.vdouble(0, 0, 0, 0.5, 0.5 , 0.5, 0, 0, 0, 0, 90, 0, 45), #dEtaqq, mqq, dPhibb, CSV1, CSV2, CSV3, maxCSV_jets, maxCSV_E, MET, pt1, pt2, pt3, pt4
        ),
    TTHbbej  = cms.PSet( 
        hltPathsToCheck = cms.vstring(
            "HLT_Ele27_eta2p1_WPLoose_Gsf_v",
            "HLT_Ele27_eta2p1_WPLoose_Gsf_HT200_v",
            
            # frozen menu paths
            "HLT_Ele27_WP85_Gsf_v",
            "HLT_Ele27_eta2p1_WP85_Gsf_HT200_v"
            ),
        recElecLabel   = cms.string("gedGsfElectrons"),
        #recJetLabel  = cms.string("ak4PFJetsCHS"),
        #jetTagLabel  = cms.string("pfCombinedSecondaryVertexV2BJetTags"),
        ## -- Analysis specific cuts
        minCandidates = cms.uint32(1),
        HtJetPtMin = cms.untracked.double(30),
        HtJetEtaMax = cms.untracked.double(3.0),
        ),
    AHttH  = cms.PSet(
        hltPathsToCheck = cms.vstring(
            "HLT_PFHT450_SixJet40_PFBTagCSV0p72_v",
            "HLT_PFHT400_SixJet30_BTagCSV0p55_2PFBTagCSV0p72_v",
            "HLT_PFHT450_SixJet40_v",
            "HLT_PFHT400_SixJet30_v",
            
            # frozen menu paths
            "HLT_PFHT450_SixJet40_PFBTagCSV_v",
            "HLT_PFHT400_SixJet30_BTagCSV0p5_2PFBTagCSV_v"
            ),
        #recElecLabel   = cms.string("gedGsfElectrons"),
        recJetLabel  = cms.string("ak4PFJetsCHS"),
        jetTagLabel  = cms.string("pfCombinedSecondaryVertexV2BJetTags"),
        ## -- Analysis specific cuts
        minCandidates = cms.uint32(6), 
        ),
    WHToENuBB  = cms.PSet( 
        hltPathsToCheck = cms.vstring(
            "HLT_Ele27_WPLoose_Gsf_WHbbBoost_v",
            "HLT_Ele23_WPLoose_Gsf_WHbbBoost_v"
            ),
        recElecLabel   = cms.string("gedGsfElectrons"),
        recJetLabel  = cms.string("ak4PFJetsCHS"),
        minCandidates = cms.uint32(1),
        ),

    MSSMHbb  = cms.PSet(
        hltPathsToCheck = cms.vstring(
            "HLT_DoubleJetsC100_DoubleBTagCSV0p85_DoublePFJetsC160_v",
            "HLT_DoubleJetsC100_DoubleBTagCSV0p9_DoublePFJetsC100MaxDeta1p6_v",
            "HLT_DoubleJetsC112_DoubleBTagCSV0p85_DoublePFJetsC172_v",
            "HLT_DoubleJetsC112_DoubleBTagCSV0p9_DoublePFJetsC112MaxDeta1p6_v",
            ),
        recJetLabel  = cms.string("ak4PFJetsCHS"),
        jetTagLabel  = cms.string("pfCombinedInclusiveSecondaryVertexV2BJetTags"),
        # -- Analysis specific cuts
        minCandidates = cms.uint32(3),
        NminOneCuts = cms.untracked.vdouble(0, 0, 0, 0.941, 0.941 , 0.00, 0, 0, 0, 100, 100, 0.0, 0.0),
        ),
)
