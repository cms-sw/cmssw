import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

from DQMOffline.Trigger.VBFMETMonitor_Client_cff import *

from DQMOffline.Trigger.VBFTauMonitor_Client_cff import *

diphotonEfficiency = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/photon/HLT_Diphoton30_22_R9Id_OR_IsoCaloId_AND_HE_R9Id_Mass90_v","HLT/photon/HLT_Diphoton30_22_R9Id_OR_IsoCaloId_AND_HE_R9Id_Mass95_v","HLT_Diphoton30PV_18PV_R9Id_AND_IsoCaloId_AND_HE_R9Id_PixelVeto_Mass55_v","HLT_Diphoton30PV_18PV_R9Id_AND_IsoCaloId_AND_HE_R9Id_NoPixelVeto_Mass55_v","HLT_Diphoton30EB_18EB_R9Id_OR_IsoCaloId_AND_HE_R9Id_NoPixelVeto_Mass55_v","HLT_Diphoton30EB_18EB_R9Id_OR_IsoCaloId_AND_HE_R9Id_PixelVeto_Mass55_v"),
                                    #subDirs        = cms.untracked.vstring("HLT/Higgs/*"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "eff_diphoton_pt       'efficiency vs lead pt;             Photon pt [GeV]; efficiency'     photon_pt_numerator          photon_pt_denominator",
        "eff_diphoton_variable 'efficiency vs lead pt;             Photon pt [GeV]; efficiency'     photon_pt_variable_numerator photon_pt_variable_denominator",
        "eff_diphoton_eta      'efficiency vs lead eta;            Photon eta; efficiency'          photon_eta_numerator         photon_eta_denominator",
        "eff_diphoton_subpt    'efficiency vs sublead pt;          Photon subpt [GeV]; efficiency'  subphoton_pt_numerator       subphoton_pt_denominator",
        "eff_diphoton_subeta   'efficiency vs sublead eta;         Photon subeta; efficiency'       subphoton_eta_numerator      subphoton_eta_denominator",
        "eff_diphoton_mass     'efficiency vs diphoton mass;       Diphoton mass; efficiency'       diphoton_mass_numerator      diphoton_mass_denominator",
        "eff_photon_phi        'efficiency vs lead phi;            Photon phi [rad]; efficiency'    photon_phi_numerator         photon_phi_denominator",
        "eff_photon_subphi     'efficiency vs sublead phi;         Photon subphi [rad]; efficiency' subphoton_phi_numerator      subphoton_phi_denominator",
        "eff_photonr9          'efficiency vs r9;                  Photon r9; efficiency'           photon_r9_numerator          photon_r9_denominator",
        "eff_photonhoE         'efficiency vs hoE;                 Photon hoE; efficiency'          photon_hoE_numerator         photon_hoE_denominator",
        "eff_photonEtaPhi      'Photon phi;                        Photon eta; efficiency'          photon_etaphi_numerator      photon_etaphi_denominator",
        "eff_photon_subr9      'efficiency vs sublead r9;          Photon subr9; efficiency'        subphoton_r9_numerator       subphoton_r9_denominator",
        "eff_photon_subhoE     'efficiency vs sublead hoE;         Photon subhoE; efficiency'       subphoton_hoE_numerator      subphoton_hoE_denominator",
        "eff_photon_subEtaPhi  'Photon sublead phi;                Photon sublead eta; efficiency'  subphoton_etaphi_numerator   subphoton_etaphi_denominator",

    ),
    efficiencyProfile = cms.untracked.vstring(
        "eff_photon_vs_LS 'Photon pt efficiency vs LS; LS' photonVsLS_numerator photonVsLS_denominator"
    ),
                                    )
higgsClient = cms.Sequence(
    diphotonEfficiency
    +vbfmetClient
    +vbftauClient
    )
