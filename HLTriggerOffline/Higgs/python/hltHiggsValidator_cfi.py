import FWCore.ParameterSet.Config as cms


hltHiggsValidator = cms.EDAnalyzer("HLTHiggsValidator",
        
    hltProcessName = cms.string("HLT"),
    analysis       = cms.vstring("HWW", "HZZ", "Hgg", "Htaunu", "H2tau", "VBFHbb", "ZnnHbb","DoubleHinTaus","HiggsDalitz","X4b","TTHbbej"), 
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
    parametersPu       = cms.vdouble(10, 0, 20),
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
    Photon_recCut     = cms.string("pt > 20 && abs(eta) < 2.4 && hadronicOverEm < 0.1 && ( r9 < 0.85 || ("+\
            " ( abs(eta) < 1.479 && sigmaIetaIeta < 0.014  || "+\
            "   abs(eta) > 1.479 && sigmaIetaIeta < 0.0035 ) && "+\
            " ecalRecHitSumEtConeDR03 < (5.0+0.012*et) && hcalTowerSumEtConeDR03 < (5.0+0.0005*et )  && trkSumPtSolidConeDR03 < (5.0 + 0.0002*et)"+\
            " )"+")" ),
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
              #prescaled control paths
              "HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_v",
              "HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_v",
              "HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL_v",
              "HLT_Ele23_CaloIdL_TrackIdL_IsoVL_v",
              "HLT_Ele12_CaloIdL_TrackIdL_IsoVL_v"
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
            "HLT_TripleMu_12_10_5_1PairDZ_v",
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
            "HLT_Photon36_R9Id85_OR_CaloId24b40e_Iso50T80L_Photon18_AND_HE10_R9Id65_Mass95_v",
            "HLT_Photon42_R9Id85_OR_CaloId24b40e_Iso50T80L_Photon22_AND_HE10_R9Id65_v",
            "HLT_Photon28_R9Id85_OR_CaloId24b40e_Iso50T80L_Photon14_AND_HE10_R9Id65_Mass50_Eta1p5_v",
            "HLT_Photon36_R9Id85_AND_CaloId24b40e_Iso50T80L_Photon18_AND_HE10_R9Id65_Mass30_v",
            "HLT_Photon36_R9Id85_AND_CaloId24b40e_Iso50T80L_Photon18_AND_HE10_R9Id65_v",
            "HLT_Photon36_R9Id85_OR_CaloId24b40e_Iso50T80L_Photon18_AND_HE10_R9Id65_Mass70_v"
        ),
        recPhotonLabel  = cms.string("photons"),
        # -- Analysis specific cuts
        minCandidates = cms.uint32(2), 
        ),
     DoubleHinTaus = cms.PSet(
        hltPathsToCheck = cms.vstring(
            "HLT_Mu17_Mu8_SameSign_v",
            "HLT_Mu17_Mu8_SameSign_DPhi_v"
        ),
        recMuonLabel  = cms.string("muons"),
        # -- Analysis specific cuts
        minCandidates = cms.uint32(2),
        ),
     HiggsDalitz = cms.PSet(
        hltPathsToCheck = cms.vstring(
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
            "HLT_LooseIsoPFTau35_Trk20_Prong1_MET70_v",
            "HLT_LooseIsoPFTau35_Trk20_Prong1_MET75_v",
            "HLT_LooseIsoPFTau50_Trk30_eta2p1_MET120_v",
            # monitoring triggers for efficiency measurement
            "HLT_LooseIsoPFTau50_Trk30_eta2p1_v",
            "HLT_IsoMu16_eta2p1_CaloMET30_LooseIsoPFTau50_Trk30_eta2p1_v",
            "HLT_IsoMu16_eta2p1_CaloMET30_v"
            ),
        recPFTauLabel   = cms.string("hpsPFTauProducer"),
        recCaloMETLabel = cms.string("caloMet"),
        recMuonLabel  = cms.string("muons"),
        # -- Analysis specific cuts
        minCandidates = cms.uint32(2), 
        ),
    H2tau  = cms.PSet( 
        hltPathsToCheck = cms.vstring(
            #"HLT_Ele22_eta2p1_WP90Rho_LooseIsoPFTau20_v",#?
            #"HLT_Ele22_eta2p1_WP90Rho_Gsf_LooseIsoPFTau20_v",
            #"HLT_Ele20_CaloIdVT_CaloIsoRhoT_TrkIdT_TrkIsoT_LooseIsoPFTau20_v",
            "HLT_IsoMu17_eta2p1_LooseIsoPFTau20_v",
            "HLT_DoubleMediumIsoPFTau40_Trk1_eta2p1_Reg_v",
            #"HLT_IsoMu24_eta2p1_IterTrk02_LooseIsoPFTau20_v",
            "HLT_IsoMu17_eta2p1_LooseIsoPFTau20_SingleL1_v",
            "HLT_IsoMu17_eta2p1_MediumIsoPFTau40_Trk1_eta2p1_Reg_v",
            "HLT_IsoMu17_eta2p1_v",
            "HLT_DoubleIsoMu17_eta2p1_v",
            "HLT_IsoMu16_eta2p1_CaloMET30_v",
            "HLT_Mu16_eta2p1_CaloMET30_v",
            "HLT_Ele27_eta2p1_WP85_Gsf_DoubleMediumIsoPFTau40_Trk1_eta2p1_Reg_v",
            "HLT_Ele32_eta2p1_WP85_Gsf_DoubleMediumIsoPFTau40_Trk1_eta2p1_Reg_v",
            "HLT_Ele27_eta2p1_WP85_Gsf_LooseIsoPFTau20_v",
            "HLT_Ele32_eta2p1_WP85_Gsf_LooseIsoPFTau20_v",
            "HLT_Ele22_eta2p1_WP85_Gsf_v",
            "HLT_DoubleEle24_22_eta2p1_WP85_Gsf_v",
            "HLT_IsoMu24_eta2p1_LooseIsoPFTau20_v",
            "HLT_IsoMu24_eta2p1_IterTrk02_v",
            "HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_v",
            "HLT_Ele22_eta2p1_WP85_Gsf_LooseIsoPFTau20_v",
            "HLT_Ele27_eta2p1_WP85_Gsf_v",
            "HLT_Ele32_eta2p1_WP85_Gsf_v",
            "HLT_Ele17_Ele8_Gsf_v"
            ),
        recPFTauLabel  = cms.string("hpsPFTauProducer"),
        recMuonLabel   = cms.string("muons"),
        recElecLabel   = cms.string("gedGsfElectrons"),
        recCaloMETLabel = cms.string("caloMet"),
        # -- Analysis specific cuts
        minCandidates = cms.uint32(2), 
        ),
    VBFHbb  = cms.PSet( 
        hltPathsToCheck = cms.vstring(
            "HLT_QuadPFJet_BTagCSV_VBF_v",
            "HLT_QuadPFJet_VBF_v",
            "HLT_L1_TripleJet_VBF_v"
            ),
        recJetLabel  = cms.string("ak4PFJetsCHS"),
        jetTagLabel  = cms.string("pfCombinedSecondaryVertexBJetTags"),
        # -- Analysis specific cuts
        minCandidates = cms.uint32(4), 
        NminOneCuts = cms.untracked.vdouble(2.6, 350, 2.6, 0.8, 0, 0, 0, 0, 0, 95, 85, 70, 40), #dEtaqq, mqq, dPhibb, CSV1, maxCSV_jets, maxCSV_E, MET, pt1, pt2, pt3, pt4
        ),
    ZnnHbb = cms.PSet( 
        hltPathsToCheck = cms.vstring(
            "HLT_CaloMHTNoPU90_PFMET90_PFMHT90_IDLoose_BTagCSV0p7_v",
            "HLT_PFMET120_PFMHT120_IDLoose_v",
            "HLT_CaloMHTNoPU90_PFMET90_PFMHT90_IDLoose_v"
            ),
        Jet_recCut   = cms.string("pt > 10 && abs(eta) < 2.6"),
        recJetLabel  = cms.string("ak4PFJetsCHS"),
        jetTagLabel  = cms.string("pfCombinedSecondaryVertexBJetTags"),
        recPFMETLabel = cms.string("pfMet"),  
        # -- Analysis specific cuts
        minCandidates = cms.uint32(1), 
        NminOneCuts = cms.untracked.vdouble(0, 0, 0, 0.9, 0, 0, 8, 30, 100, 70), #dEtaqq, mqq, dPhibb, CSV1, maxCSV_jets, maxCSV_E, MET, pt1
        ),
    X4b  = cms.PSet( 
        hltPathsToCheck = cms.vstring(
            "HLT_DoubleJet90_Double30_TripleCSV0p5_v",
            "HLT_DoubleJet90_Double30_DoubleCSV0p5_v",
            "HLT_QuadJet45_TripleCSV0p5_v",
            "HLT_QuadJet45_DoubleCSV0p5_v",
            ),
        recJetLabel  = cms.string("ak4PFJetsCHS"),
        jetTagLabel  = cms.string("pfCombinedSecondaryVertexBJetTags"),
        # -- Analysis specific cuts
        minCandidates = cms.uint32(4), 
        NminOneCuts = cms.untracked.vdouble(0, 0, 0, 0.5, 0.5 , 0.5, 0, 0, 0, 0, 90, 0, 45), #dEtaqq, mqq, dPhibb, CSV1, maxCSV_jets, maxCSV_E, MET, pt1, pt2, pt3, pt4
        ),
    TTHbbej  = cms.PSet( 
        hltPathsToCheck = cms.vstring(
            "HLT_Ele27_eta2p1_WP85_Gsf_v",
            "HLT_Ele27_eta2p1_WP85_Gsf_HT200_v"
            ),
        recElecLabel   = cms.string("gedGsfElectrons"),
        #recJetLabel  = cms.string("ak4PFJetsCHS"),
        #jetTagLabel  = cms.string("pfCombinedSecondaryVertexBJetTags"),
        ## -- Analysis specific cuts
        minCandidates = cms.uint32(1),
        HtJetPtMin = cms.untracked.double(30),
        HtJetEtaMax = cms.untracked.double(3.0),
        ),
)
