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

exoticaClient = cms.Sequence(
    NoBPTXEfficiency
  + photonEfficiency
  + htClient
  + metClient
)
