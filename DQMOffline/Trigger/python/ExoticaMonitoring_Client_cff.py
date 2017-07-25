import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDHarvester import DQMEDHarvester
from DQMOffline.Trigger.HTMonitoring_Client_cff import *
from DQMOffline.Trigger.METMonitoring_Client_cff import *

photonEfficiency = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/Photon/*"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "effic_photon         'Photon turnON;            Photon pt [GeV]; efficiency'     photon_pt_numerator          photon_pt_denominator",
        "effic_photon_variable 'Photon turnON;            Photon pt [GeV]; efficiency'     photon_pt_variable_numerator photon_pt_variable_denominator",
        "effic_photonPhi       'efficiency vs phi; Photon phi [rad]; efficiency' photon_phi_numerator       photon_phi_denominator",
        "effic_photonEta       'efficiency vs eta; Photon eta; efficiency' photon_eta_numerator       photon_eta_denominator",
        "effic_photonr9       'efficiency vs r9; Photon r9; efficiency' photon_r9_numerator       photon_r9_denominator",
        "effic_photonhoE       'efficiency vs hoE; Photon hoE; efficiency' photon_hoE_numerator       photon_hoE_denominator",
        "effic_photonEtaPhi       'Photon phi; Photon eta; efficiency' photon_etaphi_numerator       photon_etaphi_denominator",
    ),
    efficiencyProfile = cms.untracked.vstring(
        "effic_photon_vs_LS 'Photon pt efficiency vs LS; LS; PF MET efficiency' photonVsLS_numerator photonVsLS_denominator"
    ),
)

muonEfficiency = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/Muon/*"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages                                                                                                                                          
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

METplusTrackEfficiency = DQMEDHarvester("DQMGenericClient",
    subDirs = cms.untracked.vstring("HLT/MET/MET105_IsoTrk50/", "HLT/MET/MET120_IsoTrk50/"),
    verbose = cms.untracked.uint32(0), # Set to 2 for all messages
    resolution = cms.vstring(),
    efficiency = cms.vstring(
        "effic_met_variable    'MET leg turnON;              CaloMET [GeV]; efficiency'     met_variable_numerator    met_variable_denominator",
        "effic_metPhi          'MET leg efficiency vs phi;   CaloMET phi [rad]; efficiency' metPhi_numerator          metPhi_denominator",
        "effic_muonPt_variable 'Track leg turnON;            Muon p_{T} [GeV]; efficiency'  muonPt_variable_numerator muonPt_variable_denominator",
        "effic_muonEta         'Track leg efficiency vs eta; Muon #eta; efficiency'         muonEta_numerator         muonEta_denominator",
    ),
    efficiencyProfile = cms.untracked.vstring(
        "effic_met_vs_LS     'MET leg efficiency vs LS; LS; CaloMET leg efficiency' metVsLS_numerator metVsLS_denominator",
        "effic_muonPt_vs_LS 'Track leg efficiency vs LS; LS; Track leg efficiency'  muonPtVsLS_numerator muonPtVsLS_denominator",
    ),

)

exoticaClient = cms.Sequence(
    NoBPTXEfficiency
  + photonEfficiency
  + htClient
  + metClient
  + METplusTrackEfficiency
  + muonEfficiency
)
