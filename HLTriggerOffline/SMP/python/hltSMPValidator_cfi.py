import FWCore.ParameterSet.Config as cms

# taken from hltHiggsValidator_cfi.py in HLTriggerOffline/Higgs/python
hltSMPValidator = cms.EDAnalyzer("HLTHiggsValidator",
		
    hltProcessName = cms.string("HLT"),
    histDirectory  = cms.string("HLT/SMP"),
    analysis       = cms.vstring("SinglePhoton"),
    
    # -- The instance name of the reco::GenParticles collection
    genParticleLabel = cms.string("genParticles"),

    # -- The instance name of the reco::GenJets collection
    # (not used but required to be set)
    genJetLabel = cms.string("ak5GenJets"),

    # -- The nomber of interactions in the event
    pileUpInfoLabel  = cms.string("addPileupInfo"),

    # -- The binning of the Pt efficiency plots
    parametersTurnOn = cms.vdouble(0,
                                   1, 8, 9, 10,
                                   11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                   22, 24, 26, 28, 30, 32, 34, 36, 38, 40,
                                   45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 
                                   110, 120, 130, 140, 150, 160, 170, 180, 190, 200,
                                   220, 250, 300, 400, 500
                                   ),

    # -- (NBins, minVal, maxValue) for the Eta,Phi and nInterations efficiency plots
    parametersEta      = cms.vdouble(48, -2.400, 2.400),
    parametersPhi      = cms.vdouble(50, -3.142, 3.142),
    parametersPu       = cms.vdouble(10, 0, 20),

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
    Photon_recCut     = cms.string("pt > 20 && abs(eta) < 2.4 && hadronicOverEm < 0.1 && ("+\
		    "   abs(eta) < 1.479 && sigmaIetaIeta < 0.010  || "+\
		    "   abs(eta) > 1.479 && sigmaIetaIeta < 0.027 ) && "+\
		    " ecalRecHitSumEtConeDR03 < (5.0+0.012*et) && hcalTowerSumEtConeDR03 < (5.0+0.0005*et )  && trkSumPtSolidConeDR03 < (5.0 + 0.0002*et)" ),
    Photon_cutMinPt   = cms.double(20), # TO BE DEPRECATED
    Photon_cutMaxEta  = cms.double(2.4),# TO BE DEPRECATED

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

    SinglePhoton = cms.PSet( 
	    hltPathsToCheck = cms.vstring(
		    "HLT_Photon22_v",
		    "HLT_Photon30_v",
		    "HLT_Photon36_v",
		    "HLT_Photon50_v",
		    "HLT_Photon75_v",
		    "HLT_Photon90_v",
		    "HLT_Photon120_v",
		    "HLT_Photon165_HE10_v",
		    "HLT_Photon22_R9Id90_HE10_IsoM_v",
		    "HLT_Photon30_R9Id90_HE10_IsoM_v",
		    "HLT_Photon36_R9Id90_HE10_IsoM_v",
		    "HLT_Photon50_R9Id90_HE10_IsoM_v",
		    "HLT_Photon75_R9Id90_HE10_IsoM_v",
		    "HLT_Photon90_R9Id90_HE10_IsoM_v",
		    "HLT_Photon120_R9Id90_HE10_IsoM_v",
		    "HLT_Photon165_R9Id90_HE10_IsoM_v",
		    ),
	    recPhotonLabel  = cms.string("photons"),
	    # -- Analysis specific cuts
	    minCandidates = cms.uint32(1), 
	    ),
)
