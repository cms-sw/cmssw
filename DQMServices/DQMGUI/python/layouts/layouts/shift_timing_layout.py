from .adapt_to_new_backend import *
dqmitems={}

# These timing layouts were once added to the shift workspace to do the timing
# studies during the commissioning in 2015.
# We leave this file intact here, but disable it by commenting out the actual
# method body.

def shifttiminglayout(i, p, *rows):
  pass
  ##i["00 Shift/DetectorTimingPlots/" + p] = rows

shifttiminglayout(dqmitems, "01 Pixel",
  [{ 'path': "Pixel/Barrel/ALLMODS_chargeCOMB_Barrel",
     'description': "190456 example of timing problem <a href=https://twikilink>My link to plot instructions</a>",
     'draw': { 'withref': "yes" }},
   { 'path': "Pixel/Endcap/ALLMODS_chargeCOMB_Endcap",
     'description': "190456 example of timing problem <a href=https://twikilink>My link to plot instructions</a>",
     'draw': { 'withref': "yes" }}])

shifttiminglayout(dqmitems, "02 SiStrip",
  [{ 'path': "SiStrip/MechanicalView/TIB/Summary_ClusterStoNCorr_OnTrack__TIB",
     'description': "Signal-to-Noise (corrected for the angle) for On-Track clusters in TIB  - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftSiStrip>DQMShiftOnlineSiStrip</a> ",
     'draw': { 'withref': "yes" }},
   { 'path': "SiStrip/MechanicalView/TOB/Summary_ClusterStoNCorr_OnTrack__TOB",
     'description': "Signal-to-Noise (corrected for the angle) for On-Track clusters in TOB  - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftSiStrip>DQMShiftOnlineSiStrip</a> ",
     'draw': { 'withref': "yes" }}],
  [{ 'path': "SiStrip/MechanicalView/TID/PLUS/Summary_ClusterStoNCorr_OnTrack__TID__PLUS",
     'description': "Signal-to-Noise (corrected for the angle) for On-Track clusters in TID -ve side - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftSiStrip>DQMShiftOnlineSiStrip</a> ",
     'draw': { 'withref': "yes" }},
   { 'path': "SiStrip/MechanicalView/TID/MINUS/Summary_ClusterStoNCorr_OnTrack__TID__MINUS",
     'description': "Signal-to-Noise (corrected for the angle) for On-Track clusters in TID +ve side - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftSiStrip>DQMShiftOnlineSiStrip</a> ",
     'draw': { 'withref': "yes" }}],
  [{ 'path': "SiStrip/MechanicalView/TEC/PLUS/Summary_ClusterStoNCorr_OnTrack__TEC__PLUS",
     'description': "Signal-to-Noise (corrected for the angle) for On-Track clusters in TEC -ve side - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftSiStrip>DQMShiftOnlineSiStrip</a> ",
     'draw': { 'withref': "yes" }},
   { 'path': "SiStrip/MechanicalView/TEC/MINUS/Summary_ClusterStoNCorr_OnTrack__TEC__MINUS",
     'description': "Signal-to-Noise (corrected for the angle) for On-Track clusters in TEC +ve side - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftSiStrip>DQMShiftOnlineSiStrip</a> ",
     'draw': { 'withref': "yes" }}])

shifttiminglayout(dqmitems, "03 Ecal",
  [{ 'path': 'EcalBarrel/EBSummaryClient/EBTTT Trigger Primitives Timing summary',
     'description': 'Emulator TP timing where the largest number of events had Et matches. Towers with entries less than 3 are not considered.',
     'draw': { 'withref': "no" }}],
  [{ 'path': 'EcalEndcap/EESummaryClient/EETTT EE - Trigger Primitives Timing summary',
     'description': 'Emulator TP timing where the largest number of events had Et matches. Towers with entries less than 3 are not considered.',
     'draw': { 'withref': "no" }},
   { 'path': 'EcalEndcap/EESummaryClient/EETTT EE + Trigger Primitives Timing summary',
     'description': 'Emulator TP timing where the largest number of events had Et matches. Towers with entries less than 3 are not considered.',
     'draw': { 'withref': "no" }}])

shifttiminglayout(dqmitems, "04 Hcal",
 [{ 'path': "Hcal/DetDiagTimingMonitor_Hcal/Timing Plots/HB Timing (DT Trigger)",
    'description': "ToAdd",
    'draw': { 'withref': "no" }},
  { 'path': "Hcal/DetDiagTimingMonitor_Hcal/Timing Plots/HB Timing (RPC Trigger)",
    'description': "ToAdd",
    'draw': { 'withref': "no" }}],
 [{ 'path': "Hcal/DetDiagTimingMonitor_Hcal/Timing Plots/HEM Timing (CSC Trigger)",
    'description': "ToAdd",
    'draw': { 'withref': "no" }},
  { 'path': "Hcal/DetDiagTimingMonitor_Hcal/Timing Plots/HEP Timing (CSC Trigger)",
    'description': "ToAdd",
    'draw': { 'withref': "no" }}],
 [{ 'path': "Hcal/DetDiagTimingMonitor_Hcal/Timing Plots/HO Timing (HO SelfTrigger tech bit 11)",
    'description': "ToAdd",
    'draw': { 'withref': "no" }},
  { 'path': "Hcal/DetDiagTimingMonitor_Hcal/Timing Plots/HB Timing (GCT Trigger alg bit 15 16 17 18)",
    'description': "ToAdd",
    'draw': { 'withref': "no" }}])

shifttiminglayout(dqmitems, "05 DT ",
 [{ 'path': "DT/03-LocalTrigger-DCC/Wheel0/DCC_CorrectBXPhi_W0",
    'description': "ToAdd",
    'draw': { 'withref': "no" }},
  { 'path': "DT/04-LocalTrigger-DDU/Wheel0/DDU_CorrectBXPhi_W0",
    'description': "ToAdd",
    'draw': { 'withref': "no" }}])

shifttiminglayout(dqmitems, "06 RPC",
 [{ 'path': "L1T/L1TGMT/n_RPCb_vs_DTTF",
    'description': "ToAdd",
    'draw': { 'withref': "no" }},
  { 'path': "L1T/L1TGMT/n_RPCf_vs_CSCTF",
    'description': "ToAdd",
    'draw': { 'withref': "no" }}],
 [{ 'path': "L1T/L1TGMT/bx_DT_vs_RPC",
    'description': "ToAdd",
    'draw': { 'withref': "no" }},
  { 'path': "L1T/L1TGMT/bx_CSC_vs_RPC",
    'description': "ToAdd",
    'draw': { 'withref': "no" }},
  { 'path': "L1T/L1TGMT/bx_DT_vs_CSC",
    'description': "ToAdd",
    'draw': { 'withref': "no" }},],)

shifttiminglayout(dqmitems, "07 CSC",
 [{ 'path': "L1T/L1TGMT/bx_DT_vs_CSC",
    'description': "ToAdd",
    'draw': { 'withref': "no" }},
  { 'path': "L1T/L1TGMT/bx_CSC_vs_RPC",
    'description': "ToAdd",
    'draw': { 'withref': "no" }}],
 [{ 'path': "CSC/Summary/CSC_AFEB_RawHits_Time_mean",
    'description': "ToAdd",
    'draw': { 'withref': "no" }},
  { 'path': "CSC/Summary/CSC_AFEB_RawHits_Time_rms",
    'description': "ToAdd",
    'draw': { 'withref': "no" }}],
 [{ 'path': "CSC/CSCOfflineMonitor/Segments/hSTimeCathode",
    'description': "ToAdd",
    'draw': { 'withref': "no" }},
  { 'path': "CSC/CSCOfflineMonitor/Segments/hSTimeVsTOF",
    'description': "ToAdd",
    'draw': { 'withref': "no" }}])

apply_dqm_items_to_new_back_end(dqmitems, __file__)
