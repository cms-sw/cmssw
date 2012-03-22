import FWCore.ParameterSet.Config as cms


hltHiggsValidator = cms.EDAnalyzer("HLTHiggsValidator",
		
    hltProcessName = cms.string("HLT"),
    analysis       = cms.vstring("HWW", "HZZ", "Hgg", "Htaunu", "H2tau"),
    #analysis       = cms.vstring("HWW"),
    
    genParticleLabel = cms.string("genParticles"),
    #parametersTurnOn = cms.vdouble(0,
    #                               1,  2,  3,  4,  5,  6,  7,  8,  9, 10,
    #                               11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    #                               22, 24, 26, 28, 30, 32, 34, 36, 38, 40,
    #                               45, 50, 55, 60, 65, 70,
    #                               80, 100, 200, 500, 1000, 2000,
    #                               ), 
    parametersTurnOn = cms.vdouble(0,
                                   1, 8, 9, 10,
                                   11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                   22, 24, 26, 28, 30, 32, 34, 36, 38, 40,
                                   45, 50, 55, 60, 65, 70,
                                   80, 100,
                                   ),

    parametersEta      = cms.vdouble(48, -2.400, 2.400),
    parametersPhi      = cms.vdouble(50, -3.142, 3.142),

    cutsDr = cms.vdouble(0.4, 0.4, 0.015), # TO BE DEPRECATED
    # parameters for attempting an L1 match using a propagator
    maxDeltaPhi = cms.double(0.4),  # TO BE DEPRECATED
    maxDeltaR   = cms.double(0.4),  # TO BE DEPRECATED

    # set cuts on generated and reconstructed objects 
    # --- Muons
    Mu_genCut     = cms.string("pt > 10 && abs(eta) < 2.4 && abs(pdgId) == 13 && status == 1"),
    Mu_recCut     = cms.string("pt > 10 && abs(eta) < 2.4 && isGlobalMuon"),
    Mu_cutMinPt   = cms.double(10),
    Mu_cutMaxEta  = cms.double(2.4),
    
    # --- Electrons
    Ele_genCut      = cms.string("pt > 10 && abs(eta) < 2.5 && abs(pdgId) == 11 && status == 1"),
    Ele_recCut      = cms.string("pt > 10 && abs(eta) < 2.5 && hadronicOverEm < 0.05 && eSuperClusterOverP > 0.5 && eSuperClusterOverP < 2.5"),
    Ele_cutMinPt    = cms.double(10),
    Ele_cutMaxEta   = cms.double(2.5),

    # --- Photons
    Photon_genCut     = cms.string("abs(pdgId) == 22 && status == 1"),
    Photon_recCut     = cms.string("pt > 20 && abs(eta) < 2.4 && hadronicOverEm < 0.1 && ( r9 < 0.85 || ("+\
		    " ( abs(eta) < 1.479 && sigmaIetaIeta < 0.014  || "+\
		    "   abs(eta) > 1.479 && sigmaIetaIeta < 0.0035 ) && "+\
		    " ecalRecHitSumEtConeDR03 < (5.0+0.012*et) && hcalTowerSumEtConeDR03 < (5.0+0.0005*et )  && trkSumPtSolidConeDR03 < (5.0 + 0.0002*et)"+\
		    " )"+")" ),
    Photon_cutMinPt   = cms.double(20),
    Photon_cutMaxEta  = cms.double(2.4),

    # --- Taus: ----> ?? NOT SURE ABOUT THE CUTS
    PFTau_genCut      = cms.string("pt > 20 && abs(eta) < 2.5 && abs(pdgId) == 15 && status == 3"),
    PFTau_recCut      = cms.string(""),
    PFTau_cutMinPt    = cms.double(20),
    PFTau_cutMaxEta   = cms.double(2.5),

    # --- Muon Tracks    
    #TkMu_genCut      = cms.string("pt > 10 && abs(eta) < 2.4 && abs(pdgId) == 13 && status == 1"),
    #TkMu_recCut      = cms.string("pt > 10 && abs(eta) < 2.4"),
    #TkMu_cutMinPt    = cms.double(20),
    #TkMu_cutMaxEta   = cms.double(2.4),
    
    # --- MET (calo)    
    MET_genCut      = cms.string("(abs(pdgId) == 12 || abs(pdgId)==14 || abs(pdgId) == 16 ) && status == 1"),
    MET_recCut      = cms.string("pt > 75."),  
    MET_cutMinPt    = cms.double(75),
    MET_cutMaxEta   = cms.double(0),

    # The specific parameters per analysis
    HWW = cms.PSet( 
	    hltPathsToCheck = cms.vstring(
		    "HLT_Mu17_Mu8_v",
		    "HLT_Mu17_TkMu8_v",
		    "HLT_Mu17_Ele8_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_v",
		    "HLT_Mu8_Ele17_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_v",
		    "HLT_Ele17_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_Ele8_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_v",
		    ),
	    recMuonLabel  = cms.string("muons"),
	    recElecLabel  = cms.string("gsfElectrons"),
	    recTrackLabel = cms.string("globalMuons"),
	    # -- Analysis specific cuts
	    minCandidates = cms.uint32(2),
	    ),
    HZZ = cms.PSet( 
	    hltPathsToCheck = cms.vstring(
		    "HLT_Mu17_TkMu8_v",
		    "HLT_Ele17_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_Ele8_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_v",
		    ),
	    recMuonLabel  = cms.string("muons"),
	    recElecLabel  = cms.string("gsfElectrons"),
	    #recTrackLabel = cms.string("globalMuons"),
	    # -- Analysis specific cuts
	    minCandidates = cms.uint32(4), 
	    ),
    Hgg = cms.PSet( 
	    hltPathsToCheck = cms.vstring(
		    #   "HLT_Photon26_Photon18_v",
		    # "HLT_Photon36_Photon22_v",
		    "HLT_Photon26_R9Id85_OR_CaloId10_Iso50_Photon18_R9Id85_OR_CaloId10_Iso50_Mass60_v",
		    "HLT_Photon36_R9Id85_OR_CaloId10_Iso50_Photon22_R9Id85_OR_CaloId10_Iso50_v",
		    ),
	    recPhotonLabel  = cms.string("photons"),
	    # -- Analysis specific cuts
	    minCandidates = cms.uint32(2), 
	    ),
    Htaunu = cms.PSet( 
	    hltPathsToCheck = cms.vstring(
		    "HLT_LooseIsoPFTau35_Trk20_MET75_v",
		    ),
	    recPFTauLabel   = cms.string("hpsTancTaus"),
	    recCaloMETLabel = cms.string("met"),
	    # -- Analysis specific cuts
	    minCandidates = cms.uint32(2), 
	    ),
    H2tau  = cms.PSet( 
	    hltPathsToCheck = cms.vstring(
		    "HLT_Ele20_CaloIdVT_CaloIsoTRho_TrkIdT_TrkIsoT_LoosIsoPFTau20_v",
		    "HLT_IsoMu18_LooseIsoPFTau20_v"
		    ),
	    recPFTauLabel  = cms.string("hpsTancTaus"),
	    recMuonLabel   = cms.string("muons"),
	    recElecLabel   = cms.string("gsfElectrons"),
	    # -- Analysis specific cuts
	    minCandidates = cms.uint32(2), 
	    ),
)
