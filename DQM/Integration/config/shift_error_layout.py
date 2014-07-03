def errorlayout(i, p, *rows): i["00 Shift/Errors/" + p] = DQMItem(layout=rows)

errorlayout(dqmitems, "00 - HBHEHF Warning Plots",
 [{ 'path': "Hcal/DeadCellMonitor_Hcal/TotalDeadCells_HBHEHF_vs_LS",
    'description': "This plot represents the number of HCAL dead Cells in HB,HE,HF as a function of the LumiSection. The shift leader must be immediately informed if this number is greater than 50.", 'draw': { 'withref': "no" }}])
errorlayout(dqmitems, "01 - HO Warning Plots",
 [{ 'path': "Hcal/DeadCellMonitor_Hcal/TotalDeadCells_HO_vs_LS",
    'description': "This plot represents the number of HO dead Cells as a function of the LumiSection.",'draw':{'withref':"no"}}])
errorlayout(dqmitems, "02 - SiStrip FED errors",
 [{ 'path': "SiStrip/ReadoutView/FedSummary/FED/nFEDErrors",
    'description': "# of FEDs in error per event - Call Tracker DOC 165503 if the mean value is above 10 - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftSiStrip>DQMShiftOnlineSiStrip</a>",'draw':{'withref':"no"}}])
errorlayout(dqmitems, "03 - Hcal DQM Error",
 [{ 'path': "Hcal/DeadCellMonitor_Hcal/ProblemsInLastNLB_HBHEHF_alarm",
    'description': "This plot shows the total number of HCAL dead Cells in HB,HE,HF in last N LS. The plot is filled if there are more than 50 DeadCells present for more than 10 consequtive LumiSections. The shift leader must be immediately informed if this number is greater than 50.", 'draw': { 'withref': "no" }}])
errorlayout(dqmitems, "04 - Hcal DQM Error in HO01",
 [{ 'path': "Hcal/DeadCellMonitor_Hcal/ProblemsInLastNLB_HO01_alarm",
    'description': "This plot shows the total number of HCAL dead Cells in HO rings 0 and 1 (|ieta|<=10) in last N LS. The plot is filled if there are more than 50 DeadCells present for more than 10 consequtive LumiSections. The shift leader must be immediately informed if this number is greater than 50.", 'draw': { 'withref': "no" }}])
errorlayout(dqmitems, "05 - Hcal DQM BcN Mismatch",
 [{ 'path': "Hcal/DigiMonitor_Hcal/bad_digis/1D_digi_plots/ProblemDigisInLastNLB_HBHEHF_alarm",
    'description': "This plot shows the total number of HCAL problematic digis in HB,HE,HF in last N LS. The plot is filled if there are more than 50 BadDigis present for more than 5 consequtive LumiSections. The shift leader must be immediately informed if this number is greater than 50.", 'draw': { 'withref': "no" }}])
