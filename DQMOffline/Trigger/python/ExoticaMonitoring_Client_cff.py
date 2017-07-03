import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

metEfficiency = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/MET/*"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "effic_met          'MET turnON;            PF MET [GeV]; efficiency'     met_numerator          met_denominator",
        "effic_met_variable 'MET turnON;            PF MET [GeV]; efficiency'     met_variable_numerator met_variable_denominator",
        "effic_metPhi       'MET efficiency vs phi; PF MET phi [rad]; efficiency' metPhi_numerator       metPhi_denominator",
    ),
    efficiencyProfile = cms.untracked.vstring(
        "effic_met_vs_LS 'MET efficiency vs LS; LS; PF MET efficiency' metVsLS_numerator metVsLS_denominator"
    ),
  
)

NoBPTXEfficiency = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/NoBPTX/*"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "effic_jetE          'Calo jet energy turnON;            Jet E [GeV]; Efficiency'     jetE_numerator          jetE_denominator",
        "effic_jetE_variable 'Calo jet energy turnON;            Jet E [GeV]; Efficiency'     jetE_variable_numerator jetE_variable_denominator",
        "effic_jetEta          'Calo jet eta eff;            Jet #eta; Efficiency'     jetEta_numerator          jetEta_denominator",
        "effic_jetPhi          'Calo jet phi eff;            Jet #phi; Efficiency'     jetPhi_numerator          jetPhi_denominator",
        "effic_muonPt          'Muon pt turnON; DisplacedStandAlone Muon p_{T} [GeV]; Efficiency'     muonPt_numerator          muonPt_denominator",
        "effic_muonPt_variable 'Muon pt turnON; DisplacedStandAlone Muon p_{T} [GeV]; Efficiency'     muonPt_variable_numerator muonPt_variable_denominator",
        "effic_muonEta          'Muon eta eff; DisplacedStandAlone Muon #eta; Efficiency'     muonEta_numerator          muonEta_denominator",
        "effic_muonPhi          'Muon phi eff; DisplacedStandAlone Muon #phi; Efficiency'     muonPhi_numerator          muonPhi_denominator",
    ),
    efficiencyProfile = cms.untracked.vstring(
        "effic_jetE_vs_LS 'Calo jet energy efficiency vs LS; LS; Jet p_{T} Efficiency' jetEVsLS_numerator jetEVsLS_denominator",
    ), 
)

muonEfficiency = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/Muon/*"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages                                                                                                           \
                                                                                                                                                                                     
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "effic_muon         'Muon turnON;            Muon pt [GeV]; efficiency'     muon_pt_numerator          muon_pt_denominator",
        "effic_muon_variable 'Muon turnON;            Muon pt [GeV]; efficiency'     muon_pt_variable_numerator muon_pt_variable_denominator",
        "effic_muonPhi       'efficiency vs phi; Muon phi [rad]; efficiency' muon_phi_numerator       muon_phi_denominator",
        "effic_muonEta       'efficiency vs eta; Muon eta; efficiency' muon_eta_numerator       muon_eta_denominator",
        "effic_muonEtaPhi       'Muon phi; Muon eta; efficiency' muon_etaphi_numerator       muon_etaphi_denominator",
        "effic_muondxy       'efficiency vs dxy; Muon dxy; efficiency' muon_dxy_numerator       muon_dxy_denominator",
        "effic_muondz       'efficiency vs dz; Muon dz; efficiency' muon_dz_numerator       muon_dz_denominator",
        "effic_muonetaVB       'efficiency vs eta; Muon eta; efficiency' muon_eta_variablebinning_numerator       muon_eta_variablebinning_denominator",
    ),
    efficiencyProfile = cms.untracked.vstring(
        "effic_muon_vs_LS 'Muon pt efficiency vs LS; LS; PF MET efficiency' muonVsLS_numerator muonVsLS_denominator"
    ),

)

exoticaClient = cms.Sequence(
    metEfficiency
    + NoBPTXEfficiency
    + muonEfficiency
)
