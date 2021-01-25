from .adapt_to_new_backend import *
dqmitems={}

moreInfoStr = "For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftL1T\">here</a>."

def l1toccupancy(i, p, *rows): i["00 Shift/L1T/Occupancy/" + p] = rows

l1toccupancy(dqmitems,"00 - CaloLayer1 ECAL occupancy",
  [{'path': "L1T/L1TStage2CaloLayer1/ecalOccRecdEtWgt", 'description': "CaloLayer1 ECAL Et-weighted occupancy. "+moreInfoStr, 'draw': { 'withref': "no" }}])
l1toccupancy(dqmitems,"01 - CaloLayer1 HCAL occupancy",
  [{'path': "L1T/L1TStage2CaloLayer1/hcalOccRecdEtWgt", 'description': "CaloLayer1 HCAL Et-weighted occupancy. "+moreInfoStr, 'draw': { 'withref': "no" }}])
l1toccupancy(dqmitems,"02 - CaloLayer2 central jet E_T eta phi",
    [{'path': "L1T/L1TStage2CaloLayer2/shifter/CenJetsEtEtaPhi_shift", 'description': "CaloLayer2 central jet E_T eta phi. x-axis: CaloLayer2 central jet E_T eta; y-axis: CaloLayer2 central jet E_T phi. "+moreInfoStr, 'draw': { 'withref': "no" }}])
l1toccupancy(dqmitems,"03 - CaloLayer2 forward jet E_T eta phi",
    [{'path': "L1T/L1TStage2CaloLayer2/shifter/ForJetsEtEtaPhi_shift", 'description': "CaloLayer2 forward jet E_T eta phi. x-axis: CaloLayer2 forward jet E_T eta; y-axis: CaloLayer2 forward jet E_T phi. "+moreInfoStr, 'draw': { 'withref': "no" }}])
l1toccupancy(dqmitems,"04 - CaloLayer2 iso EG E_T eta phi",
    [{'path': "L1T/L1TStage2CaloLayer2/shifter/IsoEGsEtEtaPhi_shift", 'description': "CaloLayer2 iso EG E_T eta phi. x-axis: CaloLayer2 iso EG E_T eta; y-axis: CaloLayer2 iso EG E_T phi. "+moreInfoStr, 'draw': { 'withref': "no" }}])
l1toccupancy(dqmitems,"05 - CaloLayer2 noniso EG E_T eta phi",
    [{'path': "L1T/L1TStage2CaloLayer2/shifter/NonIsoEGsEtEtaPhi_shift", 'description': "CaloLayer2 noniso EG E_T eta phi. x-axis: CaloLayer2 noniso EG E_T eta; y-axis: CaloLayer2 noniso EG E_T phi. "+moreInfoStr, 'draw': { 'withref': "no" }}])
l1toccupancy(dqmitems,"06 - CaloLayer2 iso tau E_T eta phi",
    [{'path': "L1T/L1TStage2CaloLayer2/shifter/IsoTausEtEtaPhi_shift", 'description': "CaloLayer2 iso tau E_T eta phi. x-axis: CaloLayer2 iso tau E_T eta; y-axis: CaloLayer2 iso tau E_T phi. "+moreInfoStr, 'draw': { 'withref': "no" }}])
l1toccupancy(dqmitems,"07 - CaloLayer2 noniso tau E_T eta phi",
    [{'path': "L1T/L1TStage2CaloLayer2/shifter/TausEtEtaPhi_shift", 'description': "CaloLayer2 noniso tau E_T eta phi. x-axis: CaloLayer2 noniso tau E_T eta; y-axis: CaloLayer2 noniso tau E_T phi. "+moreInfoStr, 'draw': { 'withref': "no" }}])
l1toccupancy(dqmitems,"08 - uGMT muon p_T",
    [{'path': "L1T/L1TStage2uGMT/ugmtMuonPt", 'description': "uGMT muon p_T. x-axis: uGMT muon p_T. "+moreInfoStr, 'draw': { 'withref': "yes" }}])
l1toccupancy(dqmitems,"09 - uGMT muon eta",
    [{'path': "L1T/L1TStage2uGMT/ugmtMuonEta", 'description': "uGMT muon eta. x-axis: uGMT muon eta. "+moreInfoStr, 'draw': { 'withref': "yes" }}])
l1toccupancy(dqmitems,"10 - uGMT muon phi",
    [{'path': "L1T/L1TStage2uGMT/ugmtMuonPhi", 'description': "uGMT muon phi. x-axis: uGMT muon phi. "+moreInfoStr, 'draw': { 'withref': "yes" }}])
l1toccupancy(dqmitems,"11 - uGMT muon eta phi",
    [{'path': "L1T/L1TStage2uGMT/ugmtMuonPhivsEta", 'description': "uGMT muon eta phi. x-axis: uGMT muon eta; y-axis: uGMT muon phi. "+moreInfoStr, 'draw': { 'withref': "no" }}])
l1toccupancy(dqmitems,"12 - uGMT muon eta phi at vertex",
    [{'path': "L1T/L1TStage2uGMT/ugmtMuonPhiAtVtxvsEtaAtVtx", 'description': "uGMT muon eta phi at vertex. x-axis: uGMT muon eta at vertex; y-axis: uGMT muon phi at vertex. "+moreInfoStr, 'draw': { 'withref': "no" }}])
l1toccupancy(dqmitems,"13 - uGMT input link vs. input muon BX",
    [{'path': "L1T/L1TStage2uGMT/ugmtBXvsLink", 'description': "uGMT input link vs. input muon BX. x-axis: uGMT input link from (B)MTF, O(MTF), or (E)MTF; y-axis: uGMT input muon BX. "+moreInfoStr, 'draw': { 'withref': "no" }}])
l1toccupancy(dqmitems,"14 - uGMT input link vs. muon charge",
    [{'path': "L1T/L1TStage2uGMT/ugmtMuonChargevsLink", 'description': "uGMT muon charge vs. input link. x-axis: uGMT input link from (B)MTF, O(MTF), or (E)MTF; y-axis: uGMT muon charge. "+moreInfoStr, 'draw': { 'withref': "no" }}])
l1toccupancy(dqmitems,"15 - EMTF Chamber Occupancy",
    [{'path': "L1T/L1TStage2EMTF/cscLCTOccupancy", 'description': "EMTF CSC chamber Occupancy. "+moreInfoStr, 'draw': { 'withref': "no" }}])
l1toccupancy(dqmitems,"16 - EMTF Track Bunch Crossing",
    [{'path': "L1T/L1TStage2EMTF/emtfTrackBX", 'description': "EMTF Track Bunch Crossing. "+moreInfoStr, 'draw': { 'withref': "no" }}])
#l1toccupancy(dqmitems,"17 - uGT Algorithm Trigger Bits (after prescale) vs. Global BX Number",
#    [{'path': "L1T/L1TStage2uGT/algoBits_after_prescale_bx_global", 'description': "uGT Algorithm Trigger Bits (after prescale) vs. Global BX Number. "+moreInfoStr, 'draw': { 'withref': "no" }}])
#l1toccupancy(dqmitems,"18 - uGT Algorithm Trigger Bits (after prescale) vs. BX Number in Event",
#    [{'path': "L1T/L1TStage2uGT/algoBits_after_prescale_bx_inEvt", 'description': "uGT Algorithm Trigger Bits (after prescale) vs. BX Number in Event. "+moreInfoStr, 'draw': { 'withref': "no" }}])
l1toccupancy(dqmitems,"19 - uGT # of unprescaled algo accepts relative to # of all algo accepts in +-2 BX vs. BX number in event for first bunch in train",
    [{'path': "L1T/L1TStage2uGT/Ratio_Unprescaled_First_Bunch_In_Train", 'description': "uGT number of unprescaled algo accepts relative to number of all unprescaled algo accepts in +/-2 BX vs. BX number in event for first bunch in train. x-axis: BX number in event for first bunch in a bunch train; y-axis: uGT algorithms (before prescale); z-axis: number of algo accepts relative to number of all algo accepts in +/-2 BX. "+moreInfoStr, 'draw': { 'withref': "no" }}])
l1toccupancy(dqmitems,"20 - uGT # of unprescaled algo accepts relative to # of all algo accepts in +-2 BX vs. BX number in event for last bunch in train",
    [{'path': "L1T/L1TStage2uGT/Ratio_Unprescaled_Last_Bunch_In_Train", 'description': "uGT number of unprescaled algo accepts relative to number of all unprescaled algo accepts in +/-2 BX vs. BX number in event last bunch in train. x-axis: BX number in event for last bunch in a bunch train; y-axis: uGT algorithms (before prescale); z-axis: number of algo accepts relative to number of all algo accepts in +/-2 BX. "+moreInfoStr, 'draw': { 'withref': "no" }}])
l1toccupancy(dqmitems,"21 - uGT # of prescaled algo accepts relative to # of all algo accepts in +-2 BX vs. BX number in event for first bunch in train",
    [{'path': "L1T/L1TStage2uGT/Ratio_Prescaled_First_Bunch_In_Train", 'description': "uGT number of prescaled algo accepts relative to number of all prescaled algo accepts in +/-2 BX vs. BX number in event for first bunch in train. x-axis: BX number in event for first bunch in a bunch train; y-axis: uGT algorithms (before prescale); z-axis: number of algo accepts relative to number of all algo accepts in +/-2 BX. "+moreInfoStr, 'draw': { 'withref': "no" }}])
l1toccupancy(dqmitems,"22 - uGT # of prescaled algo accepts relative to # of all algo accepts in +-2 BX vs. BX number in event for last bunch in train",
    [{'path': "L1T/L1TStage2uGT/Ratio_Prescaled_Last_Bunch_In_Train", 'description': "uGT number of prescaled algo accepts relative to number of all prescaled algo accepts in +/-2 BX vs. BX number in event last bunch in train. x-axis: BX number in event for last bunch in a bunch train; y-axis: uGT algorithms (before prescale); z-axis: number of algo accepts relative to number of all algo accepts in +/-2 BX. "+moreInfoStr, 'draw': { 'withref': "no" }}])

def l1tefficiency(i, p, *rows): i["00 Shift/L1T/Efficiency/" + p] = rows

l1tefficiency(dqmitems, "00 - Reco Muon L1T Efficiency",
  [{'path': "L1T/L1TObjects/L1TMuon/L1TriggerVsReco/eff_pt_25_etaMin0_etaMax2p4_qualSingle", 'description': "", 'draw': { 'withref': "yes" }},
   {'path': "L1T/L1TObjects/L1TMuon/L1TriggerVsReco/eff_pt_15_etaMin0_etaMax2p4_qualDouble", 'description': "", 'draw': { 'withref': "yes" }},
   {'path': "L1T/L1TObjects/L1TMuon/L1TriggerVsReco/eff_pt_7_etaMin0_etaMax2p4_qualDouble", 'description': "", 'draw': { 'withref': "yes" }}],
  [{'path': "L1T/L1TObjects/L1TMuon/L1TriggerVsReco/eff_eta_25_qualSingle", 'description': "", 'draw': { 'withref': "yes" }},
   {'path': "L1T/L1TObjects/L1TMuon/L1TriggerVsReco/eff_eta_15_qualDouble", 'description': "", 'draw': { 'withref': "yes" }},
   {'path': "L1T/L1TObjects/L1TMuon/L1TriggerVsReco/eff_eta_7_qualDouble", 'description': "", 'draw': { 'withref': "yes" }}],
  [{'path': "L1T/L1TObjects/L1TMuon/L1TriggerVsReco/eff_phi_25_etaMin0_etaMax2p4_qualSingle", 'description': "", 'draw': { 'withref': "yes" }},
   {'path': "L1T/L1TObjects/L1TMuon/L1TriggerVsReco/eff_phi_15_etaMin0_etaMax2p4_qualDouble", 'description': "", 'draw': { 'withref': "yes" }},
   {'path': "L1T/L1TObjects/L1TMuon/L1TriggerVsReco/eff_phi_7_etaMin0_etaMax2p4_qualDouble", 'description': "", 'draw': { 'withref': "yes" }}])

l1tefficiency(dqmitems, "01 - Reco Muon L1T Efficiency per Track Finder Region",
  [{'path': "L1T/L1TObjects/L1TMuon/L1TriggerVsReco/eff_pt_25_etaMin0_etaMax0p83_qualSingle", 'description': "", 'draw': { 'withref': "yes" }},
   {'path': "L1T/L1TObjects/L1TMuon/L1TriggerVsReco/eff_pt_25_etaMin0p83_etaMax1p24_qualSingle", 'description': "", 'draw': { 'withref': "yes" }},
   {'path': "L1T/L1TObjects/L1TMuon/L1TriggerVsReco/eff_pt_25_etaMin1p24_etaMax2p4_qualSingle", 'description': "", 'draw': { 'withref': "yes" }}],
  [{'path': "L1T/L1TObjects/L1TMuon/L1TriggerVsReco/eff_phi_25_etaMin0_etaMax0p83_qualSingle", 'description': "", 'draw': { 'withref': "yes" }},
   {'path': "L1T/L1TObjects/L1TMuon/L1TriggerVsReco/eff_phi_25_etaMin0p83_etaMax1p24_qualSingle", 'description': "", 'draw': { 'withref': "yes" }},
   {'path': "L1T/L1TObjects/L1TMuon/L1TriggerVsReco/eff_phi_25_etaMin1p24_etaMax2p4_qualSingle", 'description': "", 'draw': { 'withref': "yes" }}],
  [{'path': "L1T/L1TObjects/L1TMuon/L1TriggerVsReco/eff_vtx_25_etaMin0_etaMax0p83_qualSingle", 'description': "", 'draw': { 'withref': "yes" }},
   {'path': "L1T/L1TObjects/L1TMuon/L1TriggerVsReco/eff_vtx_25_etaMin0p83_etaMax1p24_qualSingle", 'description': "", 'draw': { 'withref': "yes" }},
   {'path': "L1T/L1TObjects/L1TMuon/L1TriggerVsReco/eff_vtx_25_etaMin1p24_etaMax2p4_qualSingle", 'description': "", 'draw': { 'withref': "yes" }}])

# remove the photon plots for the time being
#l1tefficiency(dqmitems, "02 - Reco Photon L1T Efficiency",
#  [{'path': "L1T/L1TObjects/L1TEGamma/L1TriggerVsReco/efficiencyPhotonET_EB_EE_threshold_36", 'description': "", 'draw': { 'withref': "yes" }},
#   {'path': "L1T/L1TObjects/L1TEGamma/L1TriggerVsReco/efficiencyPhotonET_EB_threshold_36", 'description': "", 'draw': { 'withref': "yes" }},
#   {'path': "L1T/L1TObjects/L1TEGamma/L1TriggerVsReco/efficiencyPhotonET_EE_threshold_36", 'description': "", 'draw': { 'withref': "yes" }}])

l1tefficiency(dqmitems, "02 - Reco Electron L1T Efficiency",
  [{'path': "L1T/L1TObjects/L1TEGamma/L1TriggerVsReco/efficiencyElectronET_EB_EE_threshold_36", 'description': "", 'draw': { 'withref': "yes" }},
   {'path': "L1T/L1TObjects/L1TEGamma/L1TriggerVsReco/efficiencyElectronET_EB_threshold_36", 'description': "", 'draw': { 'withref': "yes" }},
   {'path': "L1T/L1TObjects/L1TEGamma/L1TriggerVsReco/efficiencyElectronET_EE_threshold_36", 'description': "", 'draw': { 'withref': "yes" }}],
  [{'path': "L1T/L1TObjects/L1TEGamma/L1TriggerVsReco/efficiencyElectronEta_threshold_48", 'description': "", 'draw': { 'withref': "yes" }},
   {'path': "L1T/L1TObjects/L1TEGamma/L1TriggerVsReco/efficiencyElectronPhi_threshold_48", 'description': "", 'draw': { 'withref': "yes" }},
   {'path': "L1T/L1TObjects/L1TEGamma/L1TriggerVsReco/efficiencyElectronNVertex_threshold_48", 'description': "", 'draw': { 'withref': "yes" }}])

l1tefficiency(dqmitems, "03 - Reco IsoTau L1T Efficiency",
  [{'path': "L1T/L1TObjects/L1TTau/L1TriggerVsReco/efficiencyIsoTauET_EB_EE_threshold_32", 'description': "", 'draw': { 'withref': "yes" }},
   {'path': "L1T/L1TObjects/L1TTau/L1TriggerVsReco/efficiencyIsoTauET_EB_threshold_32", 'description': "", 'draw': { 'withref': "yes" }},
   {'path': "L1T/L1TObjects/L1TTau/L1TriggerVsReco/efficiencyIsoTauET_EE_threshold_32", 'description': "", 'draw': { 'withref': "yes" }}],
  [{'path': "L1T/L1TObjects/L1TTau/L1TriggerVsReco/efficiencyIsoTauET_EB_EE_threshold_128", 'description': "", 'draw': { 'withref': "yes" }}])

l1tefficiency(dqmitems, "04 - Reco NonIsoTau L1T Efficiency",
  [{'path': "L1T/L1TObjects/L1TTau/L1TriggerVsReco/efficiencyNonIsoTauET_EB_EE_threshold_32", 'description': "", 'draw': { 'withref': "yes" }},
   {'path': "L1T/L1TObjects/L1TTau/L1TriggerVsReco/efficiencyNonIsoTauET_EB_threshold_32", 'description': "", 'draw': { 'withref': "yes" }},
   {'path': "L1T/L1TObjects/L1TTau/L1TriggerVsReco/efficiencyNonIsoTauET_EE_threshold_32", 'description': "", 'draw': { 'withref': "yes" }}],
  [{'path': "L1T/L1TObjects/L1TTau/L1TriggerVsReco/efficiencyNonIsoTauET_EB_EE_threshold_128", 'description': "", 'draw': { 'withref': "yes" }}])

l1tefficiency(dqmitems, "05 - Reco Jet L1T Efficiency",
  [{'path': "L1T/L1TObjects/L1TJet/L1TriggerVsReco/efficiencyJetEt_HB_HE_threshold_36", 'description': "", 'draw': { 'withref': "yes" }},
   {'path': "L1T/L1TObjects/L1TJet/L1TriggerVsReco/efficiencyJetEt_HB_HE_threshold_68", 'description': "", 'draw': { 'withref': "yes" }}],
  [{'path': "L1T/L1TObjects/L1TJet/L1TriggerVsReco/efficiencyJetEt_HB_HE_threshold_128", 'description': "", 'draw': { 'withref': "yes" }},
   {'path': "L1T/L1TObjects/L1TJet/L1TriggerVsReco/efficiencyJetEt_HB_HE_threshold_176", 'description': "", 'draw': { 'withref': "yes" }}])

l1tefficiency(dqmitems, "06 - Reco MET L1T Efficiency",
  [{'path': "L1T/L1TObjects/L1TEtSum/L1TriggerVsReco/efficiencyMET_threshold_40", 'description': "", 'draw': { 'withref': "yes" }},
   {'path': "L1T/L1TObjects/L1TEtSum/L1TriggerVsReco/efficiencyMET_threshold_60", 'description': "", 'draw': { 'withref': "yes" }}],
  [{'path': "L1T/L1TObjects/L1TEtSum/L1TriggerVsReco/efficiencyMET_threshold_80", 'description': "", 'draw': { 'withref': "yes" }},
   {'path': "L1T/L1TObjects/L1TEtSum/L1TriggerVsReco/efficiencyMET_threshold_100", 'description': "", 'draw': { 'withref': "yes" }},
   {'path': "L1T/L1TObjects/L1TEtSum/L1TriggerVsReco/efficiencyMET_threshold_120", 'description': "", 'draw': { 'withref': "yes" }}])


def l1tresolution(i, p, *rows): i["00 Shift/L1T/Resolution/" + p] = rows

l1tresolution(dqmitems, "00 - Muon per Track Finder Region",
  [{'path': "L1T/L1TObjects/L1TMuon/L1TriggerVsReco/resolution/resolution_qoverpt_etaMin0_etaMax0p83_qualAll", 'description': "", 'draw': { 'withref': "yes" }},
   {'path': "L1T/L1TObjects/L1TMuon/L1TriggerVsReco/resolution/resolution_qoverpt_etaMin0p83_etaMax1p24_qualAll", 'description': "", 'draw': { 'withref': "yes" }},
   {'path': "L1T/L1TObjects/L1TMuon/L1TriggerVsReco/resolution/resolution_qoverpt_etaMin1p24_etaMax2p4_qualAll", 'description': "", 'draw': { 'withref': "yes" }}],
  [{'path': "L1T/L1TObjects/L1TMuon/L1TriggerVsReco/resolution/resolution_eta_etaMin0_etaMax0p83_qualAll", 'description': "", 'draw': { 'withref': "yes" }},
   {'path': "L1T/L1TObjects/L1TMuon/L1TriggerVsReco/resolution/resolution_eta_etaMin0p83_etaMax1p24_qualAll", 'description': "", 'draw': { 'withref': "yes" }},
   {'path': "L1T/L1TObjects/L1TMuon/L1TriggerVsReco/resolution/resolution_eta_etaMin1p24_etaMax2p4_qualAll", 'description': "", 'draw': { 'withref': "yes" }}],
  [{'path': "L1T/L1TObjects/L1TMuon/L1TriggerVsReco/resolution/resolution_phi_etaMin0_etaMax0p83_qualAll", 'description': "", 'draw': { 'withref': "yes" }},
   {'path': "L1T/L1TObjects/L1TMuon/L1TriggerVsReco/resolution/resolution_phi_etaMin0p83_etaMax1p24_qualAll", 'description': "", 'draw': { 'withref': "yes" }},
   {'path': "L1T/L1TObjects/L1TMuon/L1TriggerVsReco/resolution/resolution_phi_etaMin1p24_etaMax2p4_qualAll", 'description': "", 'draw': { 'withref': "yes" }}])

# remove the photon plots for the time being
#l1tresolution(dqmitems, "01 - Photon",
#  [{'path': "L1T/L1TObjects/L1TEGamma/L1TriggerVsReco/resolutionPhotonET_EB", 'description': "", 'draw': { 'withref': "yes" }},
#   {'path': "L1T/L1TObjects/L1TEGamma/L1TriggerVsReco/resolutionPhotonET_EE", 'description': "", 'draw': { 'withref': "yes" }}],
#  [{'path': "L1T/L1TObjects/L1TEGamma/L1TriggerVsReco/resolutionPhotonEta", 'description': "", 'draw': { 'withref': "yes" }},
#   {'path': "L1T/L1TObjects/L1TEGamma/L1TriggerVsReco/resolutionPhotonPhi_EB", 'description': "", 'draw': { 'withref': "yes" }},
#   {'path': "L1T/L1TObjects/L1TEGamma/L1TriggerVsReco/resolutionPhotonPhi_EE", 'description': "", 'draw': { 'withref': "yes" }}])

l1tresolution(dqmitems, "01 - Electron",
  [{'path': "L1T/L1TObjects/L1TEGamma/L1TriggerVsReco/resolutionElectronET_EB", 'description': "", 'draw': { 'withref': "yes" }},
   {'path': "L1T/L1TObjects/L1TEGamma/L1TriggerVsReco/resolutionElectronET_EE", 'description': "", 'draw': { 'withref': "yes" }}],
  [{'path': "L1T/L1TObjects/L1TEGamma/L1TriggerVsReco/resolutionElectronEta", 'description': "", 'draw': { 'withref': "yes" }},
   {'path': "L1T/L1TObjects/L1TEGamma/L1TriggerVsReco/resolutionElectronPhi_EB", 'description': "", 'draw': { 'withref': "yes" }},
   {'path': "L1T/L1TObjects/L1TEGamma/L1TriggerVsReco/resolutionElectronPhi_EE", 'description': "", 'draw': { 'withref': "yes" }}])

l1tresolution(dqmitems, "02 - Tau",
  [{'path': "L1T/L1TObjects/L1TTau/L1TriggerVsReco/resolutionTauET_EB", 'description': "", 'draw': { 'withref': "yes" }},
   {'path': "L1T/L1TObjects/L1TTau/L1TriggerVsReco/resolutionTauET_EE", 'description': "", 'draw': { 'withref': "yes" }}],
  [{'path': "L1T/L1TObjects/L1TTau/L1TriggerVsReco/resolutionTauEta", 'description': "", 'draw': { 'withref': "yes" }},
   {'path': "L1T/L1TObjects/L1TTau/L1TriggerVsReco/resolutionTauPhi_EB", 'description': "", 'draw': { 'withref': "yes" }},
   {'path': "L1T/L1TObjects/L1TTau/L1TriggerVsReco/resolutionTauPhi_EE", 'description': "", 'draw': { 'withref': "yes" }}])

l1tresolution(dqmitems, "03 - Jet",
  [{'path': "L1T/L1TObjects/L1TJet/L1TriggerVsReco/resolutionJetET_HB", 'description': "", 'draw': { 'withref': "yes" }},
   {'path': "L1T/L1TObjects/L1TJet/L1TriggerVsReco/resolutionJetET_HE", 'description': "", 'draw': { 'withref': "yes" }},
   {'path': "L1T/L1TObjects/L1TJet/L1TriggerVsReco/resolutionJetET_HF", 'description': "", 'draw': { 'withref': "yes" }}],
  [{'path': "L1T/L1TObjects/L1TJet/L1TriggerVsReco/resolutionJetPhi_HB", 'description': "", 'draw': { 'withref': "yes" }},
   {'path': "L1T/L1TObjects/L1TJet/L1TriggerVsReco/resolutionJetPhi_HE", 'description': "", 'draw': { 'withref': "yes" }},
   {'path': "L1T/L1TObjects/L1TJet/L1TriggerVsReco/resolutionJetPhi_HF", 'description': "", 'draw': { 'withref': "yes" }}],
  [{'path': "L1T/L1TObjects/L1TJet/L1TriggerVsReco/resolutionJetEta", 'description': "", 'draw': { 'withref': "yes" }}])

l1tresolution(dqmitems, "04 - MET",
  [{'path': "L1T/L1TObjects/L1TEtSum/L1TriggerVsReco/resolutionETT", 'description': "", 'draw': { 'withref': "yes" }},
   {'path': "L1T/L1TObjects/L1TEtSum/L1TriggerVsReco/resolutionHTT", 'description': "", 'draw': { 'withref': "yes" }}],
  [{'path': "L1T/L1TObjects/L1TEtSum/L1TriggerVsReco/resolutionMET", 'description': "", 'draw': { 'withref': "yes" }},
   {'path': "L1T/L1TObjects/L1TEtSum/L1TriggerVsReco/resolutionMETPhi", 'description': "", 'draw': { 'withref': "yes" }}],
  [{'path': "L1T/L1TObjects/L1TEtSum/L1TriggerVsReco/resolutionMHT", 'description': "", 'draw': { 'withref': "yes" }},
   {'path': "L1T/L1TObjects/L1TEtSum/L1TriggerVsReco/resolutionMHTPhi", 'description': "", 'draw': { 'withref': "yes" }}])


apply_dqm_items_to_new_back_end(dqmitems, __file__)
