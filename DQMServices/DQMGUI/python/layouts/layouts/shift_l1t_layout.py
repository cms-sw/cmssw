from .adapt_to_new_backend import *
dqmitems={}

moreInfoStr = "For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftL1T\">here</a>."

def l1tlayout(i, p, *rows): i["00 Shift/L1T/" + p] = rows

l1tlayout(dqmitems,"00 - CaloLayer1 ECAL occupancy",
  [{'path': "L1T/L1TStage2CaloLayer1/ecalOccRecdEtWgt", 'description': "CaloLayer1 ECAL Et-weighted occupancy. "+moreInfoStr, 'draw': { 'withref': "no" }}])
l1tlayout(dqmitems,"01 - CaloLayer1 HCAL occupancy",
  [{'path': "L1T/L1TStage2CaloLayer1/hcalOccRecdEtWgt", 'description': "CaloLayer1 HCAL Et-weighted occupancy. "+moreInfoStr, 'draw': { 'withref': "no" }}])
l1tlayout(dqmitems,"02 - CaloLayer2 central jet E_T eta phi",
    [{'path': "L1T/L1TStage2CaloLayer2/shifter/CenJetsEtEtaPhi_shift", 'description': "CaloLayer2 central jet E_T eta phi. x-axis: CaloLayer2 central jet E_T eta; y-axis: CaloLayer2 central jet E_T phi. "+moreInfoStr, 'draw': { 'withref': "no" }}])
l1tlayout(dqmitems,"03 - CaloLayer2 forward jet E_T eta phi",
    [{'path': "L1T/L1TStage2CaloLayer2/shifter/ForJetsEtEtaPhi_shift", 'description': "CaloLayer2 forward jet E_T eta phi. x-axis: CaloLayer2 forward jet E_T eta; y-axis: CaloLayer2 forward jet E_T phi. "+moreInfoStr, 'draw': { 'withref': "no" }}])
l1tlayout(dqmitems,"04 - CaloLayer2 iso EG E_T eta phi",
    [{'path': "L1T/L1TStage2CaloLayer2/shifter/IsoEGsEtEtaPhi_shift", 'description': "CaloLayer2 iso EG E_T eta phi. x-axis: CaloLayer2 iso EG E_T eta; y-axis: CaloLayer2 iso EG E_T phi. "+moreInfoStr, 'draw': { 'withref': "no" }}])
l1tlayout(dqmitems,"05 - CaloLayer2 noniso EG E_T eta phi",
    [{'path': "L1T/L1TStage2CaloLayer2/shifter/NonIsoEGsEtEtaPhi_shift", 'description': "CaloLayer2 noniso EG E_T eta phi. x-axis: CaloLayer2 noniso EG E_T eta; y-axis: CaloLayer2 noniso EG E_T phi. "+moreInfoStr, 'draw': { 'withref': "no" }}])
l1tlayout(dqmitems,"06 - CaloLayer2 iso tau E_T eta phi",
    [{'path': "L1T/L1TStage2CaloLayer2/shifter/IsoTausEtEtaPhi_shift", 'description': "CaloLayer2 iso tau E_T eta phi. x-axis: CaloLayer2 iso tau E_T eta; y-axis: CaloLayer2 iso tau E_T phi. "+moreInfoStr, 'draw': { 'withref': "no" }}])
l1tlayout(dqmitems,"07 - CaloLayer2 noniso tau E_T eta phi",
    [{'path': "L1T/L1TStage2CaloLayer2/shifter/TausEtEtaPhi_shift", 'description': "CaloLayer2 noniso tau E_T eta phi. x-axis: CaloLayer2 noniso tau E_T eta; y-axis: CaloLayer2 noniso tau E_T phi. "+moreInfoStr, 'draw': { 'withref': "no" }}])
l1tlayout(dqmitems,"08 - uGMT muon p_T",
    [{'path': "L1T/L1TStage2uGMT/ugmtMuonPt", 'description': "uGMT muon p_T. x-axis: uGMT muon p_T. "+moreInfoStr, 'draw': { 'withref': "yes" }}])
l1tlayout(dqmitems,"09 - uGMT muon eta",
    [{'path': "L1T/L1TStage2uGMT/ugmtMuonEta", 'description': "uGMT muon eta. x-axis: uGMT muon eta. "+moreInfoStr, 'draw': { 'withref': "yes" }}])
l1tlayout(dqmitems,"10 - uGMT muon phi",
    [{'path': "L1T/L1TStage2uGMT/ugmtMuonPhi", 'description': "uGMT muon phi. x-axis: uGMT muon phi. "+moreInfoStr, 'draw': { 'withref': "yes" }}])
l1tlayout(dqmitems,"11 - uGMT muon eta phi",
    [{'path': "L1T/L1TStage2uGMT/ugmtMuonPhivsEta", 'description': "uGMT muon eta phi. x-axis: uGMT muon eta; y-axis: uGMT muon phi. "+moreInfoStr, 'draw': { 'withref': "no" }}])
l1tlayout(dqmitems,"12 - uGMT muon eta phi at vertex",
    [{'path': "L1T/L1TStage2uGMT/ugmtMuonPhiAtVtxvsEtaAtVtx", 'description': "uGMT muon eta phi at vertex. x-axis: uGMT muon eta at vertex; y-axis: uGMT muon phi at vertex. "+moreInfoStr, 'draw': { 'withref': "no" }}])
l1tlayout(dqmitems,"13 - uGMT input link vs. input muon BX",
    [{'path': "L1T/L1TStage2uGMT/ugmtBXvsLink", 'description': "uGMT input link vs. input muon BX. x-axis: uGMT input link from (B)MTF, O(MTF), or (E)MTF; y-axis: uGMT input muon BX. "+moreInfoStr, 'draw': { 'withref': "no" }}])
l1tlayout(dqmitems,"14 - uGMT input link vs. muon charge",
    [{'path': "L1T/L1TStage2uGMT/ugmtMuonChargevsLink", 'description': "uGMT muon charge vs. input link. x-axis: uGMT input link from (B)MTF, O(MTF), or (E)MTF; y-axis: uGMT muon charge. "+moreInfoStr, 'draw': { 'withref': "no" }}])
l1tlayout(dqmitems,"15 - EMTF Chamber Occupancy",
    [{'path': "L1T/L1TStage2EMTF/cscLCTOccupancy", 'description': "EMTF CSC chamber Occupancy. "+moreInfoStr, 'draw': { 'withref': "no" }}])
l1tlayout(dqmitems,"16 - EMTF Track Bunch Crossing",
    [{'path': "L1T/L1TStage2EMTF/emtfTrackBX", 'description': "EMTF Track Bunch Crossing. "+moreInfoStr, 'draw': { 'withref': "no" }}])
l1tlayout(dqmitems,"17 - uGT Algorithm Trigger Bits (after prescale) vs. Global BX Number",
    [{'path': "L1T/L1TStage2uGT/algoBits_after_prescale_bx_global", 'description': "uGT Algorithm Trigger Bits (after prescale) vs. Global BX Number. "+moreInfoStr, 'draw': { 'withref': "no" }}])
l1tlayout(dqmitems,"18 - uGT Algorithm Trigger Bits (after prescale) vs. BX Number in Event",
    [{'path': "L1T/L1TStage2uGT/algoBits_after_prescale_bx_inEvt", 'description': "uGT Algorithm Trigger Bits (after prescale) vs. BX Number in Event. "+moreInfoStr, 'draw': { 'withref': "no" }}])
l1tlayout(dqmitems,"19 - uGT # of unprescaled algo accepts relative to # of all algo accepts in +-2 BX vs. BX number in event for first bunch in train",
    [{'path': "L1T/L1TStage2uGT/Ratio_Unprescaled_First_Bunch_In_Train", 'description': "uGT number of unprescaled algo accepts relative to number of all unprescaled algo accepts in +/-2 BX vs. BX number in event for first bunch in train. x-axis: BX number in event for first bunch in a bunch train; y-axis: uGT algorithms (before prescale); z-axis: number of algo accepts relative to number of all algo accepts in +/-2 BX. "+moreInfoStr, 'draw': { 'withref': "no" }}])
l1tlayout(dqmitems,"20 - uGT # of unprescaled algo accepts relative to # of all algo accepts in +-2 BX vs. BX number in event for last bunch in train",
    [{'path': "L1T/L1TStage2uGT/Ratio_Unprescaled_Last_Bunch_In_Train", 'description': "uGT number of unprescaled algo accepts relative to number of all unprescaled algo accepts in +/-2 BX vs. BX number in event last bunch in train. x-axis: BX number in event for last bunch in a bunch train; y-axis: uGT algorithms (before prescale); z-axis: number of algo accepts relative to number of all algo accepts in +/-2 BX. "+moreInfoStr, 'draw': { 'withref': "no" }}])
l1tlayout(dqmitems,"21 - uGT # of prescaled algo accepts relative to # of all algo accepts in +-2 BX vs. BX number in event for first bunch in train",
    [{'path': "L1T/L1TStage2uGT/Ratio_Prescaled_First_Bunch_In_Train", 'description': "uGT number of prescaled algo accepts relative to number of all prescaled algo accepts in +/-2 BX vs. BX number in event for first bunch in train. x-axis: BX number in event for first bunch in a bunch train; y-axis: uGT algorithms (before prescale); z-axis: number of algo accepts relative to number of all algo accepts in +/-2 BX. "+moreInfoStr, 'draw': { 'withref': "no" }}])
l1tlayout(dqmitems,"22 - uGT # of prescaled algo accepts relative to # of all algo accepts in +-2 BX vs. BX number in event for last bunch in train",
    [{'path': "L1T/L1TStage2uGT/Ratio_Prescaled_Last_Bunch_In_Train", 'description': "uGT number of prescaled algo accepts relative to number of all prescaled algo accepts in +/-2 BX vs. BX number in event last bunch in train. x-axis: BX number in event for last bunch in a bunch train; y-axis: uGT algorithms (before prescale); z-axis: number of algo accepts relative to number of all algo accepts in +/-2 BX. "+moreInfoStr, 'draw': { 'withref': "no" }}])


apply_dqm_items_to_new_back_end(dqmitems, __file__)
