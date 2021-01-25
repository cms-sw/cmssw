from .adapt_to_new_backend import *
dqmitems={}

def errorlayout(i, p, *rows): i["00 Shift/Errors/" + p] = rows

##errorlayout(dqmitems, "00 - HBHEHF Warning Plots",
## [{ 'path': "Hcal/DeadCellMonitor_Hcal/TotalDeadCells_HBHEHF_vs_LS",
##    'description': "This plot represents the number of HCAL dead Cells in HB,HE,HF as a function of the LumiSection. The shift leader must be immediately informed if this number is greater than 50.", 'draw': { 'withref': "no" }}])
##errorlayout(dqmitems, "01 - HO Warning Plots",
## [{ 'path': "Hcal/DeadCellMonitor_Hcal/TotalDeadCells_HO_vs_LS",
##    'description': "This plot represents the number of HO dead Cells as a function of the LumiSection.",'draw':{'withref':"no"}}])
##errorlayout(dqmitems, "02 - SiStrip FED errors",
## [{ 'path': "SiStrip/ReadoutView/FED/nFEDErrors",
##    'description': "# of FEDs in error per event - Call Tracker DOC 165503 if the mean value is above 10 - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftSiStrip>DQMShiftOnlineSiStrip</a>",'draw':{'withref':"no"}}])
##errorlayout(dqmitems, "03 - Hcal DQM Error",
## [{ 'path': "Hcal/DeadCellMonitor_Hcal/ProblemsInLastNLB_HBHEHF_alarm",
##    'description': "This plot shows the total number of HCAL dead Cells in HB,HE,HF in last N LS. The plot is filled if there are more than 50 DeadCells present for more than 10 consequtive LumiSections. The shift leader must be immediately informed if this number is greater than 50.", 'draw': { 'withref': "no" }}])
##errorlayout(dqmitems, "04 - Hcal DQM Error in HO01",
## [{ 'path': "Hcal/DeadCellMonitor_Hcal/ProblemsInLastNLB_HO01_alarm",
##    'description': "This plot shows the total number of HCAL dead Cells in HO rings 0 and 1 (|ieta|<=10) in last N LS. The plot is filled if there are more than 50 DeadCells present for more than 10 consequtive LumiSections. The shift leader must be immediately informed if this number is greater than 50.", 'draw': { 'withref': "no" }}])
##errorlayout(dqmitems, "05 - Hcal DQM BcN Mismatch",
## [{ 'path': "Hcal/DigiMonitor_Hcal/bad_digis/1D_digi_plots/ProblemDigisInLastNLB_HBHEHF_alarm",
##    'description': "This plot shows the total number of HCAL problematic digis in HB,HE,HF in last N LS. The plot is filled if there are more than 50 BadDigis present for more than 5 consequtive LumiSections. The shift leader must be immediately informed if this number is greater than 50.", 'draw': { 'withref': "no" }}])
errorlayout(dqmitems, "06 - BeamSpot missing from online",
 [{ 'path': "BeamMonitor/BeamSpotProblemMonitor/FitFromScalars/BeamSpotError",
    'description': "This plots shows if beamspot is running OK or have problems. If there is any filled entries then it will give alarm. This plot is filled when for more than 10 LSs continuously there is problem. The alarm is put off after 2 LSs. This alarm exclude no collision OR tracker OFF conditions.", 'draw': { 'withref': "no" }}])
##errorlayout(dqmitems, "07 - SiStrip DataPresentInLS",
## [{ 'path': "SiStrip/MechanicalView/DataPresentInLS",
##    'description': "Monitoring of SiStrip activity in each LS. Audio alarm is fired during stable beams if there are no digis in either SiStrip partition for some time (red box). In case of alarm, please verify with the shift leader that SiStrip HV is ON. If HV is ON during alarm, call Tracker DOC at 165503. For more details, please refer to <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftSiStrip>DQMShiftOnlineSiStrip</a> ", 'draw': { 'withref': "no" }}])

errorlayout(dqmitems, "08 - L1T Zero Suppression Error",
  [{'path': "L1T/L1TStage2uGMT/zeroSuppression/AllEvts/mismatchRatio", 'description': "Summary of the uGMT zero suppression checks for all events. Wrongly zero suppressed raw data can only be checked for the validation events. For normal events only errors where data was not zero suppressed but should have been will be found.", 'draw': { 'withref': "no" }},
  #{'path': "L1T/L1TStage2BMTF/zeroSuppression/AllEvts/mismatchRatio", 'description': "Summary of the BMTF zero suppression checks for all events. Wrongly zero suppressed raw data can only be checked for the validation events. For normal events only errors where data was not zero suppressed but should have been will be found.", 'draw': { 'withref': "no" }}
  ])

errorlayout(dqmitems, "09 - L1T Data Transmission Error",
  [{'path': "L1T/L1TStage2uGT/uGMToutput_vs_uGTinput/mismatchRatio", 'description': "Summary of the uGMT output to uGT input data transmission checks. The uGT must receive exactly the same objects that the uGMT sends. Mismatches are indicated in the ratio plot.", 'draw': { 'withref': "no" }},
  #{'path': "L1T/L1TStage2uGT/calol2ouput_vs_uGTinput/mismatchRatio", 'description': "Summary of the CaloLayer2 output to uGT input data transmission checks. The uGT must receive exactly the same objects that the CaloLayer2 sends. Mismatches are indicated in the ratio plot.", 'draw': { 'withref': "no" }}
  ])

errorlayout(dqmitems, "10 - L1T uGMT Output Integrity Error",
  [{'path': "L1T/L1TStage2uGMT/uGMTMuonCopies/uGMTMuonCopy1/mismatchRatio", 'description': "Summary of the comparison of the uGMT output muon collection with the copy #1 of the uGMT output muon collection. The two collections must contain identical data. Mismatches are indicated in the ratio plot.", 'draw': { 'withref': "no" }},
  {'path': "L1T/L1TStage2uGMT/uGMTMuonCopies/uGMTMuonCopy2/mismatchRatio", 'description': "Summary of the comparison of the uGMT output muon collection with the copy #2 of the uGMT output muon collection. The two collections must contain identical data. Mismatches are indicated in the ratio plot.", 'draw': { 'withref': "no" }},
  {'path': "L1T/L1TStage2uGMT/uGMTMuonCopies/uGMTMuonCopy3/mismatchRatio", 'description': "Summary of the comparison of the uGMT output muon collection with the copy #3 of the uGMT output muon collection. The two collections must contain identical data. Mismatches are indicated in the ratio plot.", 'draw': { 'withref': "no" }}],
  [{'path': "L1T/L1TStage2uGMT/uGMTMuonCopies/uGMTMuonCopy4/mismatchRatio", 'description': "Summary of the comparison of the uGMT output muon collection with the copy #4 of the uGMT output muon collection. The two collections must contain identical data. Mismatches are indicated in the ratio plot.", 'draw': { 'withref': "no" }},
  {'path': "L1T/L1TStage2uGMT/uGMTMuonCopies/uGMTMuonCopy5/mismatchRatio", 'description': "Summary of the comparison of the uGMT output muon collection with the copy #5 of the uGMT output muon collection. The two collections must contain identical data. Mismatches are indicated in the ratio plot.", 'draw': { 'withref': "no" }}])

errorlayout(dqmitems, "11 - HCAL uHTR-L1T Layer1 Mismatch",
  [{'path': "L1T/L1TStage2CaloLayer1/MismatchDetail/maxEvtMismatchByLumiHCAL", 'description': "Trigger primitive mismatches between HCAL uHTR and L1T", 'draw': { 'withref': "no" }},
  #{'path': "L1T/L1TStage2uGT/calol2ouput_vs_uGTinput/mismatchRatio", 'description': "Summary of the CaloLayer2 output to uGT input data transmission checks. The uGT must receive exactly the same objects that the CaloLayer2 sends. Mismatches are indicated in the ratio plot.", 'draw': { 'withref': "no" }}
  ])

errorlayout(dqmitems, "12 - HCAL LED Misfires",
  [{'path': "Hcal/DigiTask/LED_CUCountvsLSmod60/Subdet/HE", 'description': "Number of LED misfire events (HE) vs FED vs LSmod60", 'draw': { 'withref': "no" }},
  {'path': "Hcal/DigiTask/LED_CUCountvsLSmod60/Subdet/HF", 'description': "Number of LED misfire events (HF) vs FED vs LSmod60", 'draw': { 'withref': "no" }},  
  ])

errorlayout(dqmitems, "13 - HCAL Bad CapID rotation",
  [{'path':"Hcal/DigiTask/CapID/CapID_BadvsLSmod60", 'description': "Bad CapID values vs FED vs LSmod60", 'draw': { 'withref': "no" }}
  ])

apply_dqm_items_to_new_back_end(dqmitems, __file__)
