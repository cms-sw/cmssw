### Configuration Fragment Include for HLTExoticaValidator module.
### In this file we instantiate the HLTExoticaValidator, with
### some default configurations. The specific analyses are loaded
### as cms.PSets, which are then added to this module with
### specific names. The canonical example is 
#
# from HLTriggerOffline.Exotica.hltExoticaHighPtDimuon_cff import HighPtDimuonPSet
#
# which is then made known to the module by the line
#
# analysis       = cms.vstring("HighPtDimuon"),
#

import FWCore.ParameterSet.Config as cms

# The specific analyses to be loaded
from HLTriggerOffline.Exotica.analyses.hltExoticaHighPtDimuon_cff import HighPtDimuonPSet
from HLTriggerOffline.Exotica.analyses.hltExoticaHighPtDielectron_cff import HighPtDielectronPSet
from HLTriggerOffline.Exotica.analyses.hltExoticaEleMu_cff import EleMuPSet
from HLTriggerOffline.Exotica.analyses.hltExoticaPureMET_cff import PureMETPSet
from HLTriggerOffline.Exotica.analyses.hltExoticaMonojet_cff import MonojetPSet
from HLTriggerOffline.Exotica.analyses.hltExoticaHT_cff import HTPSet


hltExoticaValidator = cms.EDAnalyzer("HLTExoticaValidator",
		
    hltProcessName = cms.string("HLT"),
    
    # -- The name of the analysis. This is the name that
    # appears in Run summary/Exotica/ANALYSIS_NAME
    analysis       = cms.vstring("HighPtDimuon",
                                 "HighPtDielectron"),
#                                 "EleMu",
#                                 "PureMET",
#                                 "Monojet",
#                                 "HT"),
    
    # -- The instance name of the reco::GenParticles collection
    genParticleLabel = cms.string("genParticles"),

    # -- The binning of the Pt efficiency plots
    # NOTICE: these DEFINITELY should be tuned for the different analyses.
    # What we have there is a generic, 0-100 GeV uniform binning.
    parametersTurnOn = cms.vdouble( 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20,
                                   22, 24, 26, 28, 30, 32, 34, 36, 38, 40,
                                   42, 44, 46, 48, 50, 52, 54, 56, 58, 60,
                                   62, 64, 66, 68, 70, 72, 74, 76, 78, 80,
                                   82, 84, 86, 88, 90, 92, 94, 96, 98, 100,
                                   ),

    # -- (NBins, minVal, maxValue) for the Eta and Phi efficiency plots
    parametersEta      = cms.vdouble(48, -2.400, 2.400),
    parametersPhi      = cms.vdouble(50, -3.142, 3.142),

    # Definition of generic cuts on generated and reconstructed objects (note that
    # these cuts can be overloaded inside a particular analysis)
    # Objects recognized: Mu Ele Photon PFTau Jet MET
    # Syntax in the strings: valid syntax of the StringCutObjectSelector class
    # --- Muons
    Mu_genCut     = cms.string("pt > 10 && abs(eta) < 2.4 && abs(pdgId) == 13 && status == 1"),
    Mu_recCut     = cms.string("pt > 10 && abs(eta) < 2.4 && isPFMuon && (isTrackerMuon || isGlobalMuon)"), # Loose Muon
    
    # --- Electrons
    Ele_genCut      = cms.string("pt > 10 && abs(eta) < 2.5 && abs(pdgId) == 11 && status == 1"),
    Ele_recCut      = cms.string("pt > 10 && abs(eta) < 2.5 && hadronicOverEm < 0.05 &&"+\
                                     "eSuperClusterOverP > 0.5 && eSuperClusterOverP < 2.5"), # Loose-like electron

    # --- Photons
    Photon_genCut     = cms.string("pt > 20 && abs(eta) < 2.4 && abs(pdgId) == 22 && status == 1"),
    Photon_recCut     = cms.string("pt > 20 && abs(eta) < 2.4"), # STILL MISSING THIS INFO
   
    # --- Taus: 
    PFTau_genCut      = cms.string("pt > 20 && abs(eta) < 2.4 && abs(pdgId) == 15 && status == 3"),
    PFTau_recCut      = cms.string("pt > 20 && abs(eta) < 2.4"),  # STILL MISSING THIS INFO
   
    # --- Jets: 
    Jet_genCut      = cms.string("pt > 30 && abs(eta) < 2.4"),
    Jet_recCut      = cms.string("pt > 30 && abs(eta) < 2.4 &&"+\
                                     "(neutralHadronEnergy + HFHadronEnergy)/energy < 0.99 &&"+\
                                     "neutralEmEnergyFraction < 0.99 &&"+\
                                     "numberOfDaughters > 1 &&"+\
                                     "chargedHadronEnergyFraction > 0 &&"+\
                                     "chargedMultiplicity > 0 && "+\
                                     "chargedEmEnergyFraction < 0.99"),  # Loose PFJet
   
    # --- MET (PF)    
    MET_genCut      = cms.string("pt > 75"),
    MET_recCut      = cms.string("pt > 75"),  
   
    # The specific parameters per analysis: the name of the parameter set has to be 
    # the same as the defined ones in the 'analysis' datamember. Each analysis is a PSet
    # with the mandatory attributes:
    #    - hltPathsToCheck (cms.vstring) : a list of all the trigger pats to be checked 
    #                 in this analysis. Up to the version number _v, but not including 
    #                 the number in order to avoid this version dependence. Example: HLT_Mu18_v
    #    - recVarLabel (cms.string): where Var is Mu, Ele, Photon, MET, Jet, PFTau, MET. This
    #                 attribute is the name of the INSTANCE LABEL for each RECO collection to 
    #                 be considered in the analysis. Note that the trigger paths rely on some 
    #                 objects which need to be defined here, otherwise the code will complain. 
    #    - minCandidates (cms.uint32): the minimum number of GEN/RECO objects in the event
    # Besides the mandatory attributes, you can redefine the generation and reconstruction cuts
    # for any object you want.
    #    * Var_genCut, Var_recCut (cms.string): where Var=Mu, Ele, Photon, Jet, PFTau, MET (see above)
    HighPtDimuon     = HighPtDimuonPSet,
    HighPtDielectron = HighPtDielectronPSet,
    EleMu            = EleMuPSet,
    PureMET          = PureMETPSet,                                 
    Monojet          = MonojetPSet,
    HT               = HTPSet
)
