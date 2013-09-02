import FWCore.ParameterSet.Config as cms


hltHiggsValidator = cms.EDAnalyzer("HLTHiggsValidator",
		
    hltProcessName = cms.string("HLT"),
    analysis       = cms.vstring("HWW", "HZZ", "Hgg", "Htaunu", "H2tau"),
    
    # -- The instance name of the reco::GenParticles collection
    genParticleLabel = cms.string("genParticles"),

    # -- The binning of the Pt efficiency plots
    parametersTurnOn = cms.vdouble(0,
                                   1, 8, 9, 10,
                                   11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                   22, 24, 26, 28, 30, 32, 34, 36, 38, 40,
                                   45, 50, 55, 60, 65, 70,
                                   80, 100,
                                   ),

    # -- (NBins, minVal, maxValue) for the Eta and Phi efficiency plots
    parametersEta      = cms.vdouble(48, -2.400, 2.400),
    parametersPhi      = cms.vdouble(50, -3.142, 3.142),

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

    # The specific parameters per analysis: the name of the parameter set has to be 
    # the same as the defined ones in the 'analysis' datamember. Each analysis is a PSet
    # with the mandatory attributes:
    #    - hltPathsToCheck (cms.vstring) : a list of all the trigger pats to be checked 
    #                 in this analysis. Up to the version number _v, but not including 
    #                 the number in order to avoid this version dependence. Example: HLT_Mu18_v
    #    - recVarLabel (cms.string): where Var can be Muon, Elec, Photon, CaloMET, PFTau. This 
    #                 attribute is the name of the INSTANCE LABEL for each RECO collection to 
    #                 be considered in the analysis. Note that the trigger paths rely on some 
    #                 objects which need to be defined here, otherwise the code will complain. 
    #    - minCandidates (cms.uint32): the minimum number of GEN/RECO objects in the event
    # Besides the mandatory attributes, you can redefine the generation and reconstruction cuts
    # for any object you want.
    #    * Var_genCut, Var_recCut (cms.string): where Var=Mu, Ele, Photon, MET, PFTau (see above)

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
		    "HLT_Photon26_R9Id85_OR_CaloId10_Iso50_Photon18_R9Id85_OR_CaloId10_Iso50_Mass60_v",
		    "HLT_Photon36_R9Id85_OR_CaloId10_Iso50_Photon22_R9Id85_OR_CaloId10_Iso50_v",
		    ),
	    recPhotonLabel  = cms.string("photons"),
	    # -- Analysis specific cuts
	    minCandidates = cms.uint32(2), 
	    ),
    Htaunu = cms.PSet( 
	    hltPathsToCheck = cms.vstring(
		    #"HLT_LooseIsoPFTau35_Trk20_MET75_v",
		    "HLT_LooseIsoPFTau35_Trk20_Prong1_MET75_v",
		    ),
	    recPFTauLabel   = cms.string("hpsPFTauProducer"),
	    recCaloMETLabel = cms.string("met"),
	    # -- Analysis specific cuts
	    minCandidates = cms.uint32(2), 
	    ),
    H2tau  = cms.PSet( 
	    hltPathsToCheck = cms.vstring(
		    "HLT_Ele20_CaloIdVT_CaloIsoRhoT_TrkIdT_TrkIsoT_LooseIsoPFTau20_v",
		    "HLT_IsoMu18_eta2p1_LooseIsoPFTau20_v"
		    ),
	    recPFTauLabel  = cms.string("hpsPFTauProducer"),
	    recMuonLabel   = cms.string("muons"),
	    recElecLabel   = cms.string("gsfElectrons"),
	    # -- Analysis specific cuts
	    minCandidates = cms.uint32(2), 
	    ),
)
