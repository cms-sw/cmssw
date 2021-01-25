from .adapt_to_new_backend import *
dqmitems={}

def l1tlayout(i, p, *rows): i["L1T/Layouts/" + p] = rows

# The quick collection is defined in ../workspaces-online.py
# for Online DQM, but we want to also include descriptions
# So we reference the 'QuickCollection' layout created here
def l1t_quickCollection(i, name, *rows):
  i["L1T/Layouts/Stage2-QuickCollection/%s" % name] = rows

# If you add a plot here, remember to add the reference to ../workspaces-online.py
l1t_quickCollection(dqmitems, "00 - Calo Layer1 ECAL Input Occupancy",
  [{
    'path': "L1T/L1TStage2CaloLayer1/ecalOccRecdEtWgt",
    'description': "This should be well populated in normal collision conditions, shaded areas represent parts of the geometry that have no associated trigger tower",
    'draw': { 'withref': "no" }
  }])
l1t_quickCollection(dqmitems, "01 - Calo Layer1 HCAL Input Occupancy",
  [{
    'path': "L1T/L1TStage2CaloLayer1/hcalOccRecdEtWgt",
    'description': "This should be well populated in normal collision conditions, shaded areas represent parts of the geometry that have no associated trigger tower",
    'draw': { 'withref': "no" }
  }])
l1t_quickCollection(dqmitems, "02 - Calo Layer1 Input Link Errors and event mismatches",
  [{
    'path': "L1T/L1TStage2CaloLayer1/maxEvtLinkErrorsByLumi",
    'description': "This should be empty at all times.",
    'draw': { 'withref': "no" }
  }],
  [{
    'path': "L1T/L1TStage2CaloLayer1/maxEvtMismatchByLumi",
    'description': "This should be empty at all times.",
    'draw': { 'withref': "no" }
  }])
l1t_quickCollection(dqmitems, "03 - uGMT MUON BX and Link vs BX",
  [{
    'path': "L1T/L1TStage2uGMT/ugmtMuonBX",
    'description': "This should have a peak at BX=0.",
    'draw': { 'withref': "yes" }
  }],
  [{
    'path': "L1T/L1TStage2uGMT/ugmtMuonBXvsLink",
    'description': "This should have a peak at BX=0 for all links.",
    'draw': { 'withref': "no" }
  }])
l1t_quickCollection(dqmitems, "04 - uGMT MUON P_{T}",
  [{
    'path': "L1T/L1TStage2uGMT/ugmtMuonPt",
    'description': "This should show normal pT spectrum (spikes at 140 GeV, 200 GeV and 255 GeV expected from max TF pT assigned)",
    'draw': { 'withref': "yes" }
  }])
l1t_quickCollection(dqmitems, "05 - uGMT MUON ETA",
  [{
    'path': "L1T/L1TStage2uGMT/ugmtMuonEta",
    'description': "This should have no spikes.",
    'draw': { 'withref': "yes" }
  }])
l1t_quickCollection(dqmitems, "06 - uGMT MUON PHI",
  [{
    'path': "L1T/L1TStage2uGMT/ugmtMuonPhi",
    'description': "This should have no spikes or dips.",
    'draw': { 'withref': "yes" }
  }])
l1t_quickCollection(dqmitems, "07 - uGMT MUON PHI ETA",
  [{
    'path': "L1T/L1TStage2uGMT/ugmtMuonPhivsEta",
    'description': "This should have no big holes.",
    'draw': { 'withref': "no" }
  }])
l1t_quickCollection(dqmitems, "08 - uGT Algoritm Trigger Bits (before prescale) vs Global BX Number",
  [{
    'path': "L1T/L1TStage2uGT/algoBits_before_prescale_bx_global",
    'description': "",
    'draw': { 'withref': "no" }
  }])
l1t_quickCollection(dqmitems, "09 - uGT Algorithm Trigger Bits (after prescale) vs. Global BX Number",
  [{
    'path': "L1T/L1TStage2uGT/algoBits_after_prescale_bx_global",
    'description': "",
    'draw': { 'withref': "no" }
  }])
l1t_quickCollection(dqmitems, "10 - uGT Algorithm Trigger Bits (after prescale) vs. BX Number in Event",
  [{
    'path': "L1T/L1TStage2uGT/algoBits_after_prescale_bx_inEvt",
    'description': "",
    'draw': { 'withref': "no" }
  }])
l1t_quickCollection(dqmitems, "11 - uGT Algorithm Trigger Bits (after prescale)",
  [{
    'path': "L1T/L1TStage2uGT/algoBits_after_prescale",
    'description': "",
    'draw': { 'withref': "no" }
  }])
l1t_quickCollection(dqmitems, "12 - Calo Layer2 Bx Occupancy distributions",
  [{
    'path': "L1T/L1TStage2CaloLayer2/Central-Jets/CenJetsBxOcc",
    'description': "",
    'draw': { 'withref': "no" }
  },
  {
    'path': "L1T/L1TStage2CaloLayer2/Forward-Jets/ForJetsBxOcc",
    'description': "",
    'draw': { 'withref': "no" }
  }],
  [{
    'path': "L1T/L1TStage2CaloLayer2/Isolated-EG/IsoEGsBxOcc",
    'description': "",
    'draw': { 'withref': "no" }
  },
  {
    'path': "L1T/L1TStage2CaloLayer2/NonIsolated-EG/NonIsoEGsBxOcc",
    'description': "",
    'draw': { 'withref': "no" }
  }],
  [{
    'path': "L1T/L1TStage2CaloLayer2/Isolated-Tau/IsoTausBxOcc",
    'description': "",
    'draw': { 'withref': "no" }
  },
  {
    'path': "L1T/L1TStage2CaloLayer2/NonIsolated-Tau/TausBxOcc",
    'description': "",
    'draw': { 'withref': "no" }
  }])
l1t_quickCollection(dqmitems, "13 - Calo Layer2 Central Jet Et Eta vs Phi distribution",
  [{
    'path': "L1T/L1TStage2CaloLayer2/shifter/CenJetsEtEtaPhi_shift",
    'description': "",
    'draw': { 'withref': "no" }
  }])
l1t_quickCollection(dqmitems, "14 - Calo Layer2 Central Jets Pt distribution",
  [{
    'path': "L1T/L1TStage2CaloLayer2/Central-Jets/CenJetsRank",
    'description': "",
    'draw': { 'withref': "yes" }
  }])
l1t_quickCollection(dqmitems, "15 - Calo Layer2 Forward Jet Et Eta vs Phi distribution",
  [{
    'path': "L1T/L1TStage2CaloLayer2/shifter/ForJetsEtEtaPhi_shift",
    'description': "",
    'draw': { 'withref': "no" }
  }])
l1t_quickCollection(dqmitems, "16 - Calo Layer2 Forward Jet Pt distribution",
  [{
    'path': "L1T/L1TStage2CaloLayer2/Forward-Jets/ForJetsRank",
    'description': "",
    'draw': { 'withref': "yes" }
  }])
l1t_quickCollection(dqmitems, "17 - Calo Layer2 Isolated EG Et Eta vs Phi distribution",
  [{
    'path': "L1T/L1TStage2CaloLayer2/shifter/IsoEGsEtEtaPhi_shift",
    'description': "",
    'draw': { 'withref': "no" }
  }])
l1t_quickCollection(dqmitems, "18 - Calo Layer2 Isolated EG Pt distribution",
  [{
    'path': "L1T/L1TStage2CaloLayer2/Isolated-EG/IsoEGsRank",
    'description': "",
    'draw': { 'withref': "yes" }
  }])
l1t_quickCollection(dqmitems, "19 - Calo Layer2 Non-Isolated EG Et Eta vs Phi distribution",
  [{
    'path': "L1T/L1TStage2CaloLayer2/shifter/NonIsoEGsEtEtaPhi_shift",
    'description': "",
    'draw': { 'withref': "no" }
  }])
l1t_quickCollection(dqmitems, "20 - Calo Layer2 Non-Isolated EG Pt distribution",
  [{
    'path': "L1T/L1TStage2CaloLayer2/NonIsolated-EG/NonIsoEGsRank",
    'description': "",
    'draw': { 'withref': "yes" }
  }])
l1t_quickCollection(dqmitems, "21 - Calo Layer2 Isolated Tau Et Eta vs Phi distribution",
  [{
    'path': "L1T/L1TStage2CaloLayer2/shifter/IsoTausEtEtaPhi_shift",
    'description': "",
    'draw': { 'withref': "no" }
  }])
l1t_quickCollection(dqmitems, "22 - Calo Layer2 Isolated Tau Pt distribution",
  [{
    'path': "L1T/L1TStage2CaloLayer2/Isolated-Tau/IsoTausRank",
    'description': "",
    'draw': { 'withref': "yes" }
  }])
l1t_quickCollection(dqmitems, "23 - Calo Layer2 Non-Isolated Tau Et Eta vs Phi distribution",
  [{
    'path': "L1T/L1TStage2CaloLayer2/shifter/TausEtEtaPhi_shift",
    'description': "",
    'draw': { 'withref': "no" }
  }])
l1t_quickCollection(dqmitems, "24 - Calo Layer2 Non-Isolated Tau Pt distribution",
  [{
    'path': "L1T/L1TStage2CaloLayer2/NonIsolated-Tau/TausRank",
    'description': "",
    'draw': { 'withref': "yes" }
  }])
l1t_quickCollection(dqmitems, "25 - Calo Layer2 EtSum Bx Occupancy distribution",
  [{
    'path': "L1T/L1TStage2CaloLayer2/Energy-Sums/EtSumBxOcc",
    'description': "",
    'draw': { 'withref': "no" }
  }])
l1t_quickCollection(dqmitems, "26 - Calo Layer2 ETT Et distribution",
  [{
    'path': "L1T/L1TStage2CaloLayer2/Energy-Sums/ETTRank",
    'description': "",
    'draw': { 'withref': "yes" }
  }])
l1t_quickCollection(dqmitems, "27 - Calo Layer2 ETTEM Et distribution",
  [{
    'path': "L1T/L1TStage2CaloLayer2/Energy-Sums/ETTEMRank",
    'description': "",
    'draw': { 'withref': "yes" }
  }])
l1t_quickCollection(dqmitems, "28 - Calo Layer2 HTT Et distribution",
  [{
    'path': "L1T/L1TStage2CaloLayer2/Energy-Sums/HTTRank",
    'description': "",
    'draw': { 'withref': "yes" }
  }])
l1t_quickCollection(dqmitems, "29 - Calo Layer2 MET Et distribution",
  [{
    'path': "L1T/L1TStage2CaloLayer2/Energy-Sums/METRank",
    'description': "",
    'draw': { 'withref': "yes" }
  }])
l1t_quickCollection(dqmitems, "30 - Calo Layer2 MET Phi distribution",
  [{
    'path': "L1T/L1TStage2CaloLayer2/Energy-Sums/METPhi",
    'description': "",
    'draw': { 'withref': "yes" }
  }])
l1t_quickCollection(dqmitems, "31 - Calo Layer2 METHF Et distribution",
  [{
    'path': "L1T/L1TStage2CaloLayer2/Energy-Sums/METHFRank",
    'description': "",
    'draw': { 'withref': "yes" }
  }])
l1t_quickCollection(dqmitems, "32 - Calo Layer2 METHF Phi distribution",
  [{
    'path': "L1T/L1TStage2CaloLayer2/Energy-Sums/METHFPhi",
    'description': "",
    'draw': { 'withref': "yes" }
  }])
l1t_quickCollection(dqmitems, "33 - Calo Layer2 MHT Et distribution",
  [{
    'path': "L1T/L1TStage2CaloLayer2/Energy-Sums/MHTRank",
    'description': "",
    'draw': { 'withref': "yes" }
  }])
l1t_quickCollection(dqmitems, "34 - Calo Layer2 MHT Phi distribution",
  [{
    'path': "L1T/L1TStage2CaloLayer2/Energy-Sums/MHTPhi",
    'description': "",
    'draw': { 'withref': "yes" }
  }])
l1t_quickCollection(dqmitems, "35 - Calo Layer2 MHTHF Et distribution",
  [{
    'path': "L1T/L1TStage2CaloLayer2/Energy-Sums/MHTHFRank",
    'description': "",
    'draw': { 'withref': "yes" }
  }])
l1t_quickCollection(dqmitems, "36 - Calo Layer2 MHTHF Phi distribution",
  [{
    'path': "L1T/L1TStage2CaloLayer2/Energy-Sums/MHTHFPhi",
    'description': "",
    'draw': { 'withref': "yes" }
  }])
l1t_quickCollection(dqmitems, "37 - uGMT BMTF BX and Wedge vs BX",
  [{
    'path': "L1T/L1TStage2uGMT/BMTFInput/ugmtBMTFBX",
    'description': "",
    'draw': { 'withref': "yes" }
  }],
  [{
    'path': "L1T/L1TStage2BMTF/bmtf_wedge_bx",
    'description': "",
    'draw': { 'withref': "no" }
  }])
l1t_quickCollection(dqmitems, "38 - BMTF Muon HW p_{T}",
  [{
    'path': "L1T/L1TStage2BMTF/bmtf_hwPt",
    'description': "",
    'draw': { 'withref': "yes" }
  }])
l1t_quickCollection(dqmitems, "39 - uGMT BMTF HW Eta",
  [{
    'path': "L1T/L1TStage2uGMT/BMTFInput/ugmtBMTFhwEta",
    'description': "",
    'draw': { 'withref': "yes" }
  }])
l1t_quickCollection(dqmitems, "40 - uGMT BMTF HW Phi",
  [{
    'path': "L1T/L1TStage2uGMT/BMTFInput/ugmtBMTFglbhwPhi",
    'description': "",
    'draw': { 'withref': "yes" }
  }])
l1t_quickCollection(dqmitems, "41 - uGMT BMTF HW vs Sign",
  [{
    'path': "L1T/L1TStage2uGMT/BMTFInput/ugmtBMTFhwSign",
    'description': "",
    'draw': { 'withref': "yes" }
  }])
l1t_quickCollection(dqmitems, "42 - uGMT OMTF BX and Sector vs. BX",
  [{
    'path': "L1T/L1TStage2uGMT/OMTFInput/ugmtOMTFBX",
    'description': "",
    'draw': { 'withref': "yes" }
  }],
  [{
    'path': "L1T/L1TStage2uGMT/ugmtBXvsProcessorOMTF",
    'description': "",
    'draw': { 'withref': "no" }
  }])
l1t_quickCollection(dqmitems, "43 - uGMT OMTF HW p_{T}",
  [{
    'path': "L1T/L1TStage2uGMT/OMTFInput/ugmtOMTFhwPt",
    'description': "",
    'draw': { 'withref': "yes" }
  }])
l1t_quickCollection(dqmitems, "44 - uGMT OMTF HW Eta",
  [{
    'path': "L1T/L1TStage2uGMT/OMTFInput/ugmtOMTFhwEta",
    'description': "",
    'draw': { 'withref': "yes" }
  }])
l1t_quickCollection(dqmitems, "45 - uGMT OMTF HW Phi (top: positive, bottom: negative)",
  [{
    'path': "L1T/L1TStage2uGMT/OMTFInput/ugmtOMTFglbhwPhiPos",
    'description': "",
    'draw': { 'withref': "yes" }
  }],
  [{
    'path': "L1T/L1TStage2uGMT/OMTFInput/ugmtOMTFglbhwPhiNeg",
    'description': "",
    'draw': { 'withref': "yes" }
  }])
l1t_quickCollection(dqmitems, "46 - uGMT OMTF HW Sign",
  [{
    'path': "L1T/L1TStage2uGMT/OMTFInput/ugmtOMTFhwSign",
    'description': "",
    'draw': { 'withref': "yes" }
  }])
l1t_quickCollection(dqmitems, "47 - uGMT EMTF BX and Track vs BX",
  [{
    'path': "L1T/L1TStage2uGMT/EMTFInput/ugmtEMTFBX",
    'description': "",
    'draw': { 'withref': "yes" }
  }],
  [{
    'path': "L1T/L1TStage2EMTF/emtfTrackBX",
    'description': "",
    'draw': { 'withref': "no" }
  }])
l1t_quickCollection(dqmitems, "48 - uGMT EMTF Phi",
  [{
    'path': "L1T/L1TStage2uGMT/ugmtMuonPhiEmtf",
    'description': "",
    'draw': { 'withref': "yes" }
  }])
l1t_quickCollection(dqmitems, "49 - EMTF LCT Occupancy",
  [{
    'path': "L1T/L1TStage2EMTF/cscLCTOccupancy",
    'description': "",
    'draw': { 'withref': "no" }
  }])
l1t_quickCollection(dqmitems, "50 - Calo TPG Link Errors and event mismatches (top: ECAL, bottom: HCAL)",
  [{
    'path': "L1T/L1TStage2CaloLayer1/MismatchDetail/maxEvtLinkErrorsByLumiECAL",
    'description': "This should be empty at all times.",
    'draw': { 'withref': "no" }
  },
  {
    'path': "L1T/L1TStage2CaloLayer1/MismatchDetail/maxEvtMismatchByLumiECAL",
    'description': "This should be empty at all times.",
    'draw': { 'withref': "no" }
  }],
  [{
    'path': "L1T/L1TStage2CaloLayer1/MismatchDetail/maxEvtLinkErrorsByLumiHCAL",
    'description': "This should be empty at all times.",
    'draw': { 'withref': "no" }
  },
  {
    'path': "L1T/L1TStage2CaloLayer1/MismatchDetail/maxEvtMismatchByLumiHCAL",
    'description': "This should be empty at all times.",
    'draw': { 'withref': "no" }
  }])
l1t_quickCollection(dqmitems, "51 - uGT CaloLayer2 Inputs Board 2-6 misMatch Ratios",
  [{
    'path': "L1T/L1TStage2uGT/uGTBoardComparisons/Board1vsBoard2/CaloLayer2/mismatchRatio",
    'description': "This should be empty at all times.",
    'draw': { 'withref': "no" }
  },
  {
    'path': "L1T/L1TStage2uGT/uGTBoardComparisons/Board1vsBoard3/CaloLayer2/mismatchRatio",
    'description': "This should be empty at all times.",
    'draw': { 'withref': "no" }
  },
  {
    'path': "L1T/L1TStage2uGT/uGTBoardComparisons/Board1vsBoard4/CaloLayer2/mismatchRatio",
    'description': "This should be empty at all times.",
    'draw': { 'withref': "no" }
  }],
  [{
    'path': "L1T/L1TStage2uGT/uGTBoardComparisons/Board1vsBoard5/CaloLayer2/mismatchRatio",
    'description': "This should be empty at all times.",
    'draw': { 'withref': "no" }
  },
  {
    'path': "L1T/L1TStage2uGT/uGTBoardComparisons/Board1vsBoard6/CaloLayer2/mismatchRatio",
    'description': "This should be empty at all times.",
    'draw': { 'withref': "no" }
  }])
l1t_quickCollection(dqmitems, "52 - uGT Muon Inputs Board 2-6 misMatch Ratios",
  [{
    'path': "L1T/L1TStage2uGT/uGTBoardComparisons/Board1vsBoard2/Muons/mismatchRatio",
    'description': "This should be empty at all times.",
    'draw': { 'withref': "no" }
  },
  {
    'path': "L1T/L1TStage2uGT/uGTBoardComparisons/Board1vsBoard3/Muons/mismatchRatio",
    'description': "This should be empty at all times.",
    'draw': { 'withref': "no" }
  },
  {
    'path': "L1T/L1TStage2uGT/uGTBoardComparisons/Board1vsBoard4/Muons/mismatchRatio",
    'description': "This should be empty at all times.",
    'draw': { 'withref': "no" }
  }],
  [{
    'path': "L1T/L1TStage2uGT/uGTBoardComparisons/Board1vsBoard5/Muons/mismatchRatio",
    'description': "This should be empty at all times.",
    'draw': { 'withref': "no" }
  },
  {
    'path': "L1T/L1TStage2uGT/uGTBoardComparisons/Board1vsBoard6/Muons/mismatchRatio",
    'description': "This should be empty at all times.",
    'draw': { 'withref': "no" }
  }])
l1t_quickCollection(dqmitems, "53 - uGMT Muon Copy 1-5 misMatch Ratios",
  [{
    'path': "L1T/L1TStage2uGMT/uGMTMuonCopies/uGMTMuonCopy1/mismatchRatio",
    'description': "This should be empty at all times.",
    'draw': { 'withref': "no" }
  },
  {
    'path': "L1T/L1TStage2uGMT/uGMTMuonCopies/uGMTMuonCopy2/mismatchRatio",
    'description': "This should be empty at all times.",
    'draw': { 'withref': "no" }
  },
  {
    'path': "L1T/L1TStage2uGMT/uGMTMuonCopies/uGMTMuonCopy3/mismatchRatio",
    'description': "This should be empty at all times.",
    'draw': { 'withref': "no" }
  }],
  [{
    'path': "L1T/L1TStage2uGMT/uGMTMuonCopies/uGMTMuonCopy4/mismatchRatio",
    'description': "This should be empty at all times.",
    'draw': { 'withref': "no" }
  },
  {
    'path': "L1T/L1TStage2uGMT/uGMTMuonCopies/uGMTMuonCopy5/mismatchRatio",
    'description': "This should be empty at all times.",
    'draw': { 'withref': "no" }
  }])
l1t_quickCollection(dqmitems, "54 - Input vs Output misMatch Ratios (clockwise from top left: uGT vs. uGMT, uGT vs. caloL2, uGMT vs. EMTF, uGMT vs. OMTF, uGMT vs. BMTF)",
  [{
    'path': "L1T/L1TStage2uGT/uGMToutput_vs_uGTinput/mismatchRatio",
    'description': "This should be empty at all times.",
    'draw': { 'withref': "no" }
  },
  {
    'path': "L1T/L1TStage2uGT/calol2ouput_vs_uGTinput/mismatchRatio",
    'description': "This should be empty at all times.",
    'draw': { 'withref': "no" }
  }],
  [{
    'path': "L1T/L1TStage2uGMT/BMTFoutput_vs_uGMTinput/mismatchRatio",
    'description': "This should be empty at all times.",
    'draw': { 'withref': "no" }
  },
  {
    'path': "L1T/L1TStage2uGMT/OMTFoutput_vs_uGMTinput/mismatchRatio",
    'description': "This should be empty at all times.",
    'draw': { 'withref': "no" }
  },
  {
    'path': "L1T/L1TStage2uGMT/EMTFoutput_vs_uGMTinput/mismatchRatio",
    'description': "This should be empty at all times.",
    'draw': { 'withref': "no" }
  }])
l1t_quickCollection(dqmitems, "55 - uGMT Zero Suppression misMatch Ratio (left: all events, right: fat events)",
  [{
    'path': "L1T/L1TStage2uGMT/zeroSuppression/AllEvts/mismatchRatio",
    'description': "This should be empty at all times.",
    'draw': { 'withref': "no" }
  },
  {
    'path': "L1T/L1TStage2uGMT/zeroSuppression/FatEvts/mismatchRatio",
    'description': "This should be empty at all times.",
    'draw': { 'withref': "no" }
  }])
l1t_quickCollection(dqmitems, "56 - BMTF Zero Suppression misMatch Ratio (left: all events, right: fat events)",
  [{
    'path': "L1T/L1TStage2BMTF/zeroSuppression/AllEvts/mismatchRatio",
    'description': "This should be empty at all times.",
    'draw': { 'withref': "no" }
  },
  {
    'path': "L1T/L1TStage2BMTF/zeroSuppression/FatEvts/mismatchRatio",
    'description': "This should be empty at all times.",
    'draw': { 'withref': "no" }
  }])


###############################################
### From here down is legacy/stage1 trigger ###
###           All in Legacy folder          ###
###############################################

# def l1t_gt_single(i, dir, name):
#   i["L1T/Layouts/Legacy/00-GT-Summary/%s" % name] = \
#     DQMItem(layout=[["L1T/%s/%s" % (dir, name)]])

# def l1t_gmt_single(i, dir, name):
#   i["L1T/Layouts/Legacy/01-GMT-Summary/%s" % name] = \
#     DQMItem(layout=[["L1T/%s/%s" % (dir, name)]])

# def l1t_gct_single(i, dir, name):
#   i["L1T/Layouts/Legacy/02-Stage1Layer2-Summary/%s" % name] = \
#     DQMItem(layout=[["L1T/%s/%s" % (dir, name)]])

# #def l1t_rct_single(i, dir, name):
# #  i["L1T/Layouts/Legacy/03-RCT-Summary/%s" % name] = \
# #    DQMItem(layout=[["L1T/%s/%s" % (dir, name)]])

# def l1t_csctf_single(i, dir, name):
#   i["L1T/Layouts/Legacy/04-CSCTF-Summary/%s" % name] = \
#     DQMItem(layout=[["L1T/%s/%s" % (dir, name)]])

# #def l1t_dttf_single(i, dir, name):
# #  i["L1T/Layouts/Legacy/05-DTTF-Summary/%s" % name] = DQMItem(layout=[["L1T/%s/%s" % (dir, name)]])
# def l1t_dttf_single(i, p, *rows):
#   i["L1T/Layouts/Legacy/05-DTTF-Summary/" + p] = rows

# def l1t_rpctf_single(i, dir, name):
#   i["L1T/Layouts/Legacy/06-RPCTF-Summary/%s" % name] = \
#     DQMItem(layout=[["L1T/%s/%s" % (dir, name)]])

# def l1t_scal_single(i, p, *rows): i["L1T/Layouts/Legacy/07-SCAL4Cosmics-Summary/" + p] = rows

# def l1t_rct_expert(i, p, *rows): i["L1T/Layouts/Legacy/03-RCT-Summary/" + p] = rows
# l1t_rct_expert(dqmitems, "RctEmIsoEmEtEtaPhi",
#   [{ 'path': "L1T/L1TRCT/RctEmIsoEmEtEtaPhi", 'description': "For details see - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/RCTDataQualityMonitoring>RCTDQM</a> CAL/RCT/GCT mapping is here <a href=https://twiki.cern.ch/twiki/pub/CMS/RCTDataQualityMonitoring/RCTGCTCAL.jpeg> mapping </a>" }])

# l1t_rct_expert(dqmitems, "RctEmNonIsoEmEtEtaPhi",
#   [{ 'path': "L1T/L1TRCT/RctEmNonIsoEmEtEtaPhi", 'description': "For description see - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/RCTDataQualityMonitoring>RCTDQM</a>  CAL/RCT/GCT mapping is here <a href=https://twiki.cern.ch/twiki/pub/CMS/RCTDataQualityMonitoring/RCTGCTCAL.jpeg> mapping </a>" }])

# l1t_rct_expert(dqmitems, "RctRegionsEtEtaPhi",
#   [{ 'path': "L1T/L1TRCT/RctRegionsEtEtaPhi", 'description': "For description see - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/RCTDataQualityMonitoring>RCTDQM</a> CAL/RCT/GCT mapping is here <a href=https://twiki.cern.ch/twiki/pub/CMS/RCTDataQualityMonitoring/RCTGCTCAL.jpeg> mapping </a>" }])

# def l1t_summary(i, p, *rows): i["L1T/Layouts/Legacy/08-L1T-Summary/" + p] = rows

# l1t_summary(dqmitems,"00 Physics Trigger Rate",
#     [{'path': "L1T/L1TScalersSCAL/Level1TriggerRates/Physics Trigger Rate", 'description': "Physics Trigger Rate. x-axis: Time(lumisection); y-axis: Rate (Hz).  For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftTrigger\">here</a>."}])

# l1t_summary(dqmitems,"01 Random Trigger Rate",
#     [{'path': "L1T/L1TScalersSCAL/Level1TriggerRates/Random Trigger Rate", 'description': "Random Trigger Rate. x-axis: Time(lumisection); y-axis: Rate (Hz).  For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftTrigger\">here</a>."}])

# # list of summary GT histograms (dqmitems, dirPath , histoName)
# l1t_gt_single(dqmitems, "L1TGT", "algo_bits")
# l1t_gt_single(dqmitems, "L1TGT", "tt_bits")
# l1t_gt_single(dqmitems, "L1TGT", "gtfe_bx")
# l1t_gt_single(dqmitems, "L1Scalers_SM", "l1AlgoBits_Vs_Bx")
# l1t_gt_single(dqmitems, "L1Scalers_SM", "l1TechBits_Vs_Bx")
# l1t_gt_single(dqmitems, "BXSynch", "BxOccyGtTrigType1")

# # list of summary GMT histograms (dqmitems, dirPath , histoName)
# l1t_gmt_single(dqmitems, "L1TGMT", "DTTF_phi")
# l1t_gmt_single(dqmitems, "L1TGMT", "CSC_eta")
# l1t_gmt_single(dqmitems, "L1TGMT", "RPCb_phi")
# l1t_gmt_single(dqmitems, "L1TGMT", "GMT_phi")
# l1t_gmt_single(dqmitems, "L1TGMT", "DTTF_candlumi")
# l1t_gmt_single(dqmitems, "L1TGMT", "CSCTF_candlumi")
# l1t_gmt_single(dqmitems, "L1TGMT", "RPCb_candlumi")
# l1t_gmt_single(dqmitems, "L1TGMT", "GMT_candlumi")
# l1t_gmt_single(dqmitems, "L1TGMT", "GMT_etaphi")
# l1t_gmt_single(dqmitems, "L1TGMT", "GMT_qty")
# l1t_gmt_single(dqmitems, "L1TGMT", "n_RPCb_vs_DTTF")
# l1t_gmt_single(dqmitems, "L1TGMT", "Regional_trigger")

# # list of summary Layer2 histograms (dqmitems, dirPath , histoName)
# l1t_gct_single(dqmitems, "L1TStage1Layer2", "AllJetsEtEtaPhi")
# l1t_gct_single(dqmitems, "L1TStage1Layer2", "TauJetsEtEtaPhi")
# l1t_gct_single(dqmitems, "L1TStage1Layer2", "IsoEmRankEtaPhi")
# l1t_gct_single(dqmitems, "L1TStage1Layer2", "NonIsoEmRankEtaPhi")
# l1t_gct_single(dqmitems, "L1TStage1Layer2", "CenJetsRank")
# l1t_gct_single(dqmitems, "L1TStage1Layer2", "ForJetsRank")
# l1t_gct_single(dqmitems, "L1TStage1Layer2", "TauJetsRank")
# l1t_gct_single(dqmitems, "L1TStage1Layer2", "IsoTauJetsRank")
# l1t_gct_single(dqmitems, "L1TStage1Layer2", "IsoEmRank")
# l1t_gct_single(dqmitems, "L1TStage1Layer2", "NonIsoEmRank")
# l1t_gct_single(dqmitems, "L1TStage1Layer2", "EtMiss")
# l1t_gct_single(dqmitems, "L1TStage1Layer2", "HtMissHtTotal")
# l1t_gct_single(dqmitems, "L1TStage1Layer2", "EtTotal")
# l1t_gct_single(dqmitems, "L1TStage1Layer2", "EtHad")

# # list of summary RCT histograms (dqmitems, dirPath , histoName)
# #l1t_rct_single(dqmitems, "L1TRCT", "RctIsoEmOccEtaPhi")
# #l1t_rct_single(dqmitems, "L1TRCT", "RctNonIsoEmOccEtaPhi")
# #l1t_rct_single(dqmitems, "L1TRCT", "RctIsoEmRank")
# #l1t_rct_single(dqmitems, "L1TRCT", "RctNonIsoEmRank")

# # list of summary CSCTF histograms (dqmitems, dirPath , histoName)
# l1t_csctf_single(dqmitems, "L1TCSCTF", "CSCTF_errors")
# l1t_csctf_single(dqmitems, "L1TCSCTF", "CSCTF_occupancies")

# # list of summary RPC histograms (dqmitems, dirPath , histoName)
# l1t_rpctf_single(dqmitems, "L1TRPCTF", "RPCTF_muons_tower_phipacked")
# l1t_rpctf_single(dqmitems, "L1TRPCTF", "RPCTF_phi_valuepacked")
# l1t_rpctf_single(dqmitems, "L1TRPCTF", "RPCTF_ntrack")
# l1t_rpctf_single(dqmitems, "L1TRPCTF", "RPCTF_quality")
# l1t_rpctf_single(dqmitems, "L1TRPCTF", "RPCTF_charge_value")
# l1t_rpctf_single(dqmitems, "L1TRPCTF", "RPCTF_pt_value")
# l1t_rpctf_single(dqmitems, "L1TRPCTF", "RPCTF_bx")
# l1t_rpctf_single(dqmitems, "L1TRPCTF", "RPCDigi_bx")

# #### list of summary DTTF histograms (dqmitems, dirPath , histoName)
# ## l1t_dttf_single(dqmitems, "EventInfo/errorSummarySegments", "DT_TPG_phi_map")
# ## l1t_dttf_single(dqmitems, "L1TDTTF/DTTF_TRACKS/INTEG", "Occupancy Summary")
# ## l1t_dttf_single(dqmitems, "L1TDTTF/DTTF_TRACKS/INTEG", "Occupancy Phi vs Eta")
# ## l1t_dttf_single(dqmitems, "L1TDTTF/DTTF_TRACKS/INTEG", "Integrated Packed Phi")
# ## l1t_dttf_single(dqmitems, "L1TDTTF/DTTF_TRACKS/INTEG", "Integrated Packed Eta")
# ## l1t_dttf_single(dqmitems, "L1TDTTF/DTTF_TRACKS/INTEG", "Integrated Packed Pt")
# ## l1t_dttf_single(dqmitems, "L1TDTTF/DTTF_TRACKS/INTEG", "Integrated Packed Charge")
# ## l1t_dttf_single(dqmitems, "L1TDTTF/DTTF_TRACKS/INTEG", "Integrated Packed Quality")
# ## l1t_dttf_single(dqmitems, "L1TDTTF/DTTF_TRACKS/INTEG", "Integrated BX")
# ## l1t_dttf_single(dqmitems, "L1TDTTF/DTTF_TRACKS/INTEG", "Num Tracks Per Event")
# ## l1t_dttf_single(dqmitems, "L1TDTTF/DTTF_TRACKS/INTEG", "BX Summary")
# ## l1t_dttf_single(dqmitems, "L1TDTTF/DTTF_TRACKS/INTEG", "2nd Track Summary")
# ## l1t_dttf_single(dqmitems, "L1TDTTF/DTTF_TRACKS/INTEG", "Fractional High Quality Summary")

# #dqmitems["dttf_03_tracks_distr_summary"]['description'] = "DTTF Tracks distribution by Sector and Wheel. N0 contains usually 5-10% tracks w.r.t. P0: the violet band is normal. Normalized to total DTTF tracks number at BX=0. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DTTFDataQualityMonitoring#\"  target=\"_blank\">here</a>."

# l1t_dttf_single(dqmitems,  "01 - Number of Tracks per Event",
#            [{'path' : "L1T/L1TDTTF/01-INCLUSIVE/dttf_01_nTracksPerEvent_integ", 'description' : "Number of DTTF Tracks Per Event. Normalized to total DTTF tracks number at BX=0. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DTTFDataQualityMonitoring#01_Number_of_Tracks_per_Event\"  target=\"_blank\">here</a>."}])

# l1t_dttf_single(dqmitems,  "02 - Fraction of tracks per wheel",
#                 [{'path' : "L1T/L1TDTTF/01-INCLUSIVE/dttf_02_nTracks", 'description' : "Distribution of DTTF Tracks per Wheel. N0 contains usually 5-10% tracks w.r.t. P0. Normalized to total DTTF tracks number at BX=0. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DTTFDataQualityMonitoring#02_Fraction_of_tracks_per_wheel\"  target=\"_blank\">here</a>."}])

# l1t_dttf_single(dqmitems,  "03 - DTTF Tracks Occupancy",
#                 [{'path' : "L1T/L1TDTTF/01-INCLUSIVE/dttf_03_tracks_occupancy_summary", 'description' : "DTTF Tracks distribution by Sector and Wheel. N0 contains usually 5-10% tracks w.r.t. P0: the violet band is normal. Normalized to total DTTF tracks number at BX=0. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DTTFDataQualityMonitoring#03_DTTF_Tracks_Occupancy\"  target=\"_blank\">here</a>."}])

# l1t_dttf_single(dqmitems,  "04 - DTTF Tracks Occupancy In the Last LumiSections",
#                 [{'path' : "L1T/L1TDTTF/01-INCLUSIVE/dttf_04_tracks_occupancy_by_lumi", 'description' : "DTTF Tracks distribution by Sector and Wheel in the last Luminosity Sections. Normalized to total DTTF tracks number at BX=0. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DTTFDataQualityMonitoring#04_Tracks_Occupancy_In_the_Last\"  target=\"_blank\">here</a>."}])

# l1t_dttf_single(dqmitems,  "05 - Tracks BX Distribution by Wheel",
#                 [{'path' : "L1T/L1TDTTF/01-INCLUSIVE/dttf_05_bx_occupancy", 'description' : "DTTF Tracks BX Distribution by Wheel. Normalized to total DTTF tracks number. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DTTFDataQualityMonitoring#05_Tracks_BX_Distribution_by_Whe\"  target=\"_blank\">here</a>."}])

# l1t_dttf_single(dqmitems,  "06 - Fraction of Tracks BX w.r.t. Tracks with BX=0",
#                 [{'path' : "L1T/L1TDTTF/01-INCLUSIVE/dttf_06_bx", 'description' : "Fraction of DTTF Tracks BX w.r.t. Tracks with BX=0. By definition, Bx=0 bin is 1. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DTTFDataQualityMonitoring#06_Fraction_of_Tracks_BX_w_r_t_T\"  target=\"_blank\">here</a>."}])

# l1t_dttf_single(dqmitems,  "07 - Tracks Quality distribution",
#                 [{'path' : "L1T/L1TDTTF/01-INCLUSIVE/dttf_07_quality", 'description' : "DTTF Tracks Quality distribution. Normalized to total DTTF tracks number at BX=0. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DTTFDataQualityMonitoring#07_Tracks_Quality_distribution\"  target=\"_blank\">here</a>."}])

# l1t_dttf_single(dqmitems,  "08 - Quality distribution by wheel",
#                 [{'path' : "L1T/L1TDTTF/01-INCLUSIVE/dttf_08_quality_occupancy", 'description' : "DTTF Tracks Quality distribution by wheel. Normalized to total DTTF tracks number at BX=0. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DTTFDataQualityMonitoring#08_Tracks_Quality_distribution_b\"  target=\"_blank\">here</a>."}])

# l1t_dttf_single(dqmitems,  "09 - High Quality Tracks Occupancy",
#                 [{'path' : "L1T/L1TDTTF/01-INCLUSIVE/dttf_09_highQuality_Tracks", 'description' : "Fraction of DTTF Tracks with Quality>3 by Sector and Wheel. Relatively lower occupancy foreseen in chimney: S3 N0 (no tracks going to N1) and S4 P0 (no tracks going to P1). For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DTTFDataQualityMonitoring#09_High_Quality_Tracks_Occupancy\"  target=\"_blank\">here</a>."}])

# l1t_dttf_single(dqmitems,  "10 - Occupancy Phi vs Eta-Coarse",
#                 [{'path' : "L1T/L1TDTTF/01-INCLUSIVE/dttf_10_phi_vs_etaCoarse", 'description' : "#eta-#phi distribution of DTTF Tracks with coarse #eta assignment (packed values) normalized to total DTTF tracks number at BX=0. A sector roughly covers 12 #eta bins, starting from -6. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DTTFDataQualityMonitoring#10_Occupancy_Phi_vs_Eta_Coarse\"  target=\"_blank\">here</a>."}])

# l1t_dttf_single(dqmitems,  "11 - Occupancy Phi vs Eta-Fine",
#                 [{'path' : "L1T/L1TDTTF/01-INCLUSIVE/dttf_11_phi_vs_etaFine", 'description' : "#eta-#phi Distribution of DTTF Tracks with fine #eta assignment (packed values) normalized to total DTTF tracks number at BX=0. A sector roughly covers 12 #eta bins, starting from -6. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DTTFDataQualityMonitoring#11_Occupancy_Phi_vs_Eta_Fine\"  target=\"_blank\">here</a>."}])

# l1t_dttf_single(dqmitems,  "12 - Occupancy Phi vs Eta",
#                 [{'path' : "L1T/L1TDTTF/01-INCLUSIVE/dttf_12_phi_vs_eta", 'description' : "#eta-#phi Distribution of DTTF Tracks normalized to total DTTF tracks number at BX=0. A sector roughly covers 30deg #eta bins, starting from -15. Wheel separation are at #eta about +/-0.3 and +/-0.74. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DTTFDataQualityMonitoring#12_Occupancy_Phi_vs_Eta\"  target=\"_blank\">here</a>."}])

# l1t_dttf_single(dqmitems,  "13 - Fraction of tracks with Eta fine assigment",
#                 [{'path' : "L1T/L1TDTTF/01-INCLUSIVE/dttf_13_eta_fine_fraction", 'description' : "Fraction of DTTF Tracks with Fine #eta Assignment per Wheel. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DTTFDataQualityMonitoring#13_Fraction_of_tracks_with_Eta_f\"  target=\"_blank\">here</a>."}])

# l1t_dttf_single(dqmitems,  "14 - Integrated Packed Eta",
#                 [{'path' : "L1T/L1TDTTF/01-INCLUSIVE/dttf_14_eta", 'description' : "#eta distribution (Packed values) of DTTF Tracks normalized to total DTTF tracks number at BX=0. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DTTFDataQualityMonitoring#14_Integrated_Packed_Eta\"  target=\"_blank\">here</a>."}])

# l1t_dttf_single(dqmitems,  "15 - Integrated Packed Phi",
#                 [{'path' : "L1T/L1TDTTF/01-INCLUSIVE/dttf_15_phi", 'description' : "Phi distribution (Packed values) of DTTF Tracks normalized to total DTTF tracks number at BX=0. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DTTFDataQualityMonitoring#15_Integrated_Packed_Phi\"  target=\"_blank\">here</a>."}])

# l1t_dttf_single(dqmitems,  "16 - Integrated Packed pT",
#                 [{'path' : "L1T/L1TDTTF/01-INCLUSIVE/dttf_16_pt", 'description' : "p_{T} distribution (Packed values) of DTTF Tracks normalized to total DTTF tracks number at BX=0. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DTTFDataQualityMonitoring#16_Integrated_Packed_pT\"  target=\"_blank\">here</a>."}])

# l1t_dttf_single(dqmitems,  "17 - Integrated Packed Charge",
#                 [{'path' : "L1T/L1TDTTF/01-INCLUSIVE/dttf_17_charge", 'description' : "Charge distribution of DTTF Tracks normalized to total DTTF tracks number at BX=0. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DTTFDataQualityMonitoring#17_Integrated_Packed_Charge\"  target=\"_blank\">here</a>."}])

# l1t_dttf_single(dqmitems,  "18 - 2nd Track Summary",
#                 [{'path' : "L1T/L1TDTTF/01-INCLUSIVE/dttf_18_2ndTrack_occupancy_summary", 'description' : "DTTF 2nd Tracks Only Distribution by Sector and Wheel normalized to the total Number of tracks. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DTTFDataQualityMonitoring#18_2nd_Track_Summary\"  target=\"_blank\">here</a>."}])

# # list of summary SCAL histograms (dqmitems, dirPath , histoName)
# l1t_scal_single(dqmitems, "Rate_AlgoBit_002",
#                [{'path':"L1T/L1TScalersSCAL/Level1TriggerRates/AlgorithmRates/Rate_AlgoBit_002", 'description' :  "none"}])
# l1t_scal_single(dqmitems, "Rate_AlgoBit_003",
#                [{'path':"L1T/L1TScalersSCAL/Level1TriggerRates/AlgorithmRates/Rate_AlgoBit_003", 'description' :  "none"}])
# l1t_scal_single(dqmitems, "Rate_AlgoBit_004",
#                [{'path':"L1T/L1TScalersSCAL/Level1TriggerRates/AlgorithmRates/Rate_AlgoBit_004", 'description' :  "none"}])
# l1t_scal_single(dqmitems, "Rate_AlgoBit_005",
#                [{'path':"L1T/L1TScalersSCAL/Level1TriggerRates/AlgorithmRates/Rate_AlgoBit_005", 'description' :  "none"}])
# l1t_scal_single(dqmitems, "Rate_AlgoBit_006",
#                [{'path':"L1T/L1TScalersSCAL/Level1TriggerRates/AlgorithmRates/Rate_AlgoBit_006", 'description' :  "none"}])
# l1t_scal_single(dqmitems, "Rate_AlgoBit_007",
#                [{'path':"L1T/L1TScalersSCAL/Level1TriggerRates/AlgorithmRates/Rate_AlgoBit_007", 'description' :  "none"}])
# l1t_scal_single(dqmitems, "Rate_AlgoBit_008",
#                [{'path':"L1T/L1TScalersSCAL/Level1TriggerRates/AlgorithmRates/Rate_AlgoBit_008", 'description' :  "none"}])
# l1t_scal_single(dqmitems, "Rate_AlgoBit_009",
#                [{'path':"L1T/L1TScalersSCAL/Level1TriggerRates/AlgorithmRates/Rate_AlgoBit_009", 'description' :  "none"}])
# l1t_scal_single(dqmitems, "Rate_AlgoBit_010",
#                [{'path':"L1T/L1TScalersSCAL/Level1TriggerRates/AlgorithmRates/Rate_AlgoBit_010", 'description' :  "none"}])
# l1t_scal_single(dqmitems, "Rate_AlgoBit_011",
#                [{'path':"L1T/L1TScalersSCAL/Level1TriggerRates/AlgorithmRates/Rate_AlgoBit_011", 'description' :  "none"}])
# l1t_scal_single(dqmitems, "Rate_AlgoBit_012",
#                [{'path':"L1T/L1TScalersSCAL/Level1TriggerRates/AlgorithmRates/Rate_AlgoBit_012", 'description' :  "none"}])
# l1t_scal_single(dqmitems, "Rate_AlgoBit_013",
#                [{'path':"L1T/L1TScalersSCAL/Level1TriggerRates/AlgorithmRates/Rate_AlgoBit_013", 'description' :  "none"}])
# l1t_scal_single(dqmitems, "Rate_AlgoBit_015",
#                [{'path':"L1T/L1TScalersSCAL/Level1TriggerRates/AlgorithmRates/Rate_AlgoBit_015", 'description' :  "none"}])
# l1t_scal_single(dqmitems, "Rate_AlgoBit_016",
#                [{'path':"L1T/L1TScalersSCAL/Level1TriggerRates/AlgorithmRates/Rate_AlgoBit_016", 'description' :  "none"}])
# l1t_scal_single(dqmitems, "Rate_AlgoBit_045",
#                [{'path':"L1T/L1TScalersSCAL/Level1TriggerRates/AlgorithmRates/Rate_AlgoBit_045", 'description' :  "none"}])
# l1t_scal_single(dqmitems, "Rate_AlgoBit_054",
#                [{'path':"L1T/L1TScalersSCAL/Level1TriggerRates/AlgorithmRates/Rate_AlgoBit_054", 'description' :  "none"}])
# l1t_scal_single(dqmitems, "Rate_AlgoBit_055",
#                [{'path':"L1T/L1TScalersSCAL/Level1TriggerRates/AlgorithmRates/Rate_AlgoBit_055", 'description' :  "none"}])
# l1t_scal_single(dqmitems, "Rate_AlgoBit_056",
#                [{'path':"L1T/L1TScalersSCAL/Level1TriggerRates/AlgorithmRates/Rate_AlgoBit_056", 'description' :  "none"}])
# l1t_scal_single(dqmitems, "Rate_AlgoBit_057",
#                [{'path':"L1T/L1TScalersSCAL/Level1TriggerRates/AlgorithmRates/Rate_AlgoBit_057", 'description' :  "none"}])
# l1t_scal_single(dqmitems, "Rate_AlgoBit_058",
#                [{'path':"L1T/L1TScalersSCAL/Level1TriggerRates/AlgorithmRates/Rate_AlgoBit_058", 'description' :  "none"}])
# l1t_scal_single(dqmitems, "Rate_AlgoBit_059",
#                [{'path':"L1T/L1TScalersSCAL/Level1TriggerRates/AlgorithmRates/Rate_AlgoBit_059", 'description' :  "none"}])
# l1t_scal_single(dqmitems, "Rate_AlgoBit_060",
#                [{'path':"L1T/L1TScalersSCAL/Level1TriggerRates/AlgorithmRates/Rate_AlgoBit_060", 'description' :  "none"}])
# l1t_scal_single(dqmitems, "Rate_AlgoBit_061",
#                [{'path':"L1T/L1TScalersSCAL/Level1TriggerRates/AlgorithmRates/Rate_AlgoBit_061", 'description' :  "none"}])
# l1t_scal_single(dqmitems, "Rate_AlgoBit_062",
#                [{'path':"L1T/L1TScalersSCAL/Level1TriggerRates/AlgorithmRates/Rate_AlgoBit_062", 'description' :  "none"}])
# l1t_scal_single(dqmitems, "Rate_AlgoBit_063",
#                [{'path':"L1T/L1TScalersSCAL/Level1TriggerRates/AlgorithmRates/Rate_AlgoBit_063", 'description' :  "none"}])
# l1t_scal_single(dqmitems, "Rate_AlgoBit_065",
#                [{'path':"L1T/L1TScalersSCAL/Level1TriggerRates/AlgorithmRates/Rate_AlgoBit_065", 'description' :  "none"}])
# l1t_scal_single(dqmitems, "Rate_AlgoBit_068",
#                [{'path':"L1T/L1TScalersSCAL/Level1TriggerRates/AlgorithmRates/Rate_AlgoBit_068", 'description' :  "none"}])
# l1t_scal_single(dqmitems, "Rate_AlgoBit_070",
#                [{'path':"L1T/L1TScalersSCAL/Level1TriggerRates/AlgorithmRates/Rate_AlgoBit_070", 'description' :  "none"}])
# l1t_scal_single(dqmitems, "Rate_AlgoBit_088",
#                [{'path':"L1T/L1TScalersSCAL/Level1TriggerRates/AlgorithmRates/Rate_AlgoBit_088", 'description' :  "none"}])


apply_dqm_items_to_new_back_end(dqmitems, __file__)
