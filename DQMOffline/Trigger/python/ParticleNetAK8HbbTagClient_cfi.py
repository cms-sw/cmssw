import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

particleNetAK8HbbTagEfficiency = DQMEDHarvester("DQMGenericClient",
    subDirs    = cms.untracked.vstring("HLT/HIG/PNETAK8/HLT_IsoMu50_AK8PFJet230_SoftDropMass40_PFAK8ParticleNetBB0p35_or_HLT_Ele50_CaloIdVT_GsfTrkIdT_AK8PFJet230_SoftDropMass40/"),
    verbose    = cms.untracked.uint32(0), # Set to 2 for all messages
    resolution = cms.vstring(),
    efficiency = cms.vstring(
        "eff_muon_pt 'Efficiency vs p_{T}(#mu); p_{T}(#mu); efficiency' muon_pt_numerator muon_pt_denominator",
        "eff_muon_eta 'Efficiency vs #eta(#mu); #eta(#mu); efficiency' muon_eta_numerator muon_eta_denominator",
        "eff_electron_pt 'Efficiency vs p_{T}(ele); p_{T}(ele); efficiency' electron_pt_numerator electron_pt_denominator",
        "eff_electron_eta 'Efficiency vs #eta(ele); #eta(ele); efficiency' electron_eta_numerator electron_eta_denominator",
        "eff_ht 'Efficiency vs H_{T}; H_{T}; efficiency' ht_numerator ht_denominator",
        "eff_njets 'Efficiency vs N_{jets}; N_{jets}; efficiency' njets_numerator njets_denominator",
        "eff_nbjets 'Efficiency vs N_{bjets}; N_{bjets}; efficiency' nbjets_numerator nbjets_denominator",
        "eff_jet1_pt 'Efficiency vs p_{T}(j1); p_{T}(j1); efficiency' jet1_pt_numerator jet1_pt_denominator",
        "eff_jet1_eta 'Efficiency vs #eta(j1); #eta(j1); efficiency' jet1_eta_numerator jet1_eta_denominator",
        "eff_jet1_pnetscore 'Efficiency vs Lead PNET-score; Lead PNET-score; efficiency' jet1_pnetscore_numerator jet1_pnetscore_denominator",
        "eff_jet1_pnetscore_trans 'Efficiency vs Lead atanh(PNET-score); Lead atanh(PNET-score); efficiency' jet1_pnetscore_trans_numerator jet1_pnetscore_trans_denominator",
        "eff_jet1_pt_eta 'Efficiency vs j1 p_{T} and #eta; p_{T}(j1); #eta(j1); efficiency' jet1_pt_eta_numerator jet1_pt_eta_denominator",
        "eff_jet1_pt_pnetscore1 'Efficiency vs j1 p_{T} and Lead PNET-score; p_{T}(j1); Lead PNET-score; efficiency' jet1_pt_pnetscore1_numerator jet1_pt_pnetscore1_denominator",
        "eff_jet1_pt_pnetscore1_trans 'Efficiency vs j1 p_{T} and Lead atanh(PNET-score); p_{T}(j1); Lead atanh(PNET-score); efficiency' jet1_pt_pnetscore1_trans_numerator jet1_pt_pnetscore1_trans_denominator",
    )
)
