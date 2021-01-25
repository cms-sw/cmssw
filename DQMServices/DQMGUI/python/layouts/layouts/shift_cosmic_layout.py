from .adapt_to_new_backend import *
dqmitems={}

def shiftcosmiclayout(i, p, *rows): i["00 Shift/P5_Cosmics/" + p] = rows

########### define varialbles for frequently used strings #############
summary = "summary map for rpc, this is NOT an efficiency measurement"
rpclink = "   >>> <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftRPC>Description</a>"
fed = "FED Fatal Errors";
rpcevents = "Events processed by the RPC DQM"
quality = "Overview of system quality. Expressed in percentage of chambers."
occupancy = "Occupancy per sector"
###### needed for RPC plots #############

#Summary plots

shiftcosmiclayout(dqmitems, "00 - Tracking ReportSummary",
 [{ 'path': "Tracking/EventInfo/reportSummaryMap",
    'description': " Quality Test results plotted for Tracking parameters : Chi2, TrackRate, #of Hits in Track - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftSiStrip>DQMShiftOnlineSiStrip</a> ", 'draw': { 'withref': "no" }}])

shiftcosmiclayout(dqmitems, "01 - SiStrip ReportSummary",
 [{ 'path': "SiStrip/EventInfo/reportSummaryMap",
    'description': "Overall Report Summary where detector fraction and S/N flags are combined together -  <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftSiStrip>DQMShiftOnlineSiStrip</a> ", 'draw': { 'withref': "no" }}])

shiftcosmiclayout(dqmitems, "02 - DT ReportSummary",
 [{ 'path': "DT/EventInfo/reportSummaryMap",
    'description': "<a href=https://twiki.cern.ch/twiki/bin/view/CMS/DTDQMPlots>Description and Instructions</a>", 'draw': { 'withref': "no" }}])

shiftcosmiclayout(dqmitems, "03 - CSC ReportSummary",
 [{ 'path': "CSC/EventInfo/reportSummaryMap",
    'description': "For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftCSC#reportSummaryMap\">here</a>.", 'draw': { 'withref': "no" }}])

shiftcosmiclayout(dqmitems, "04 - RPC ReportSummary",
               [{ 'path': "RPC/EventInfo/reportSummaryMap", 'description': summary + rpclink }])

#tracks

shiftcosmiclayout(dqmitems, "05 - Tracks (Cosmic Tracking)",
 [{ 'path': "Tracking/TrackParameters/GeneralProperties/NumberOfTracks_CKFTk",
    'description': "Number of cosmic tracks  - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftSiStrip>DQMShiftOnlineSiStrip</a> ", 'draw': { 'withref': "yes" }},
  { 'path': "Tracking/TrackParameters/HitProperties/NumberOfRecHitsPerTrack_CKFTk",
    'description': "Number of RecHits per track - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftSiStrip>DQMShiftOnlineSiStrip</a> ", 'draw': { 'withref': "yes" }},
  { 'path': "Tracking/TrackParameters/GeneralProperties/TrackPt_CKFTk",
    'description': "Pt of cosmic track - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftSiStrip>DQMShiftOnlineSiStrip</a> ", 'draw': { 'withref': "yes" }}],
 [{ 'path': "Tracking/TrackParameters/GeneralProperties/Chi2oNDF_CKFTk",
    'description': "Chi Sqare per DoF - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftSiStrip>DQMShiftOnlineSiStrip</a> ", 'draw': { 'withref': "yes" }},
  { 'path': "Tracking/TrackParameters/GeneralProperties/TrackPhi_CKFTk",
    'description': "Phi distribution of cosmic tracks -  <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftSiStrip>DQMShiftOnlineSiStrip</a> ", 'draw': { 'withref': "yes" }},
  { 'path': "Tracking/TrackParameters/GeneralProperties/TrackEta_CKFTk",
    'description': " Eta distribution of cosmic tracks - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftSiStrip>DQMShiftOnlineSiStrip</a> ", 'draw': { 'withref': "yes" }}])

#strip quantities

shiftcosmiclayout(dqmitems, "06 - # of Cluster Trend",
  [{ 'path': "SiStrip/MechanicalView/TIB/TotalNumberOfClusterProfile__TIB",
      'description': "Total # of Clusters in TIB with event time in Seconds  - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftSiStrip>DQMShiftOnlineSiStrip</a> ", 'draw': { 'withref': "no" }},
   { 'path': "SiStrip/MechanicalView/TOB/TotalNumberOfClusterProfile__TOB",
       'description': "Total # of Clusters in TOB with event time in Seconds  - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftSiStrip>DQMShiftOnlineSiStrip</a> ", 'draw': { 'withref': "no" }}],
  [{ 'path': "SiStrip/MechanicalView/TID/MINUS/TotalNumberOfClusterProfile__TID__MINUS",
      'description': "Total # of Clusters in TID -ve side with event time in Seconds  - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftSiStrip>DQMShiftOnlineSiStrip</a> ", 'draw': { 'withref': "no" }},
   { 'path': "SiStrip/MechanicalView/TID/PLUS/TotalNumberOfClusterProfile__TID__PLUS",
       'description': "Total # of Clusters in TID +ve side with event time in Seconds  - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftSiStrip>DQMShiftOnlineSiStrip</a> ", 'draw': { 'withref': "no" }}],
  [{  'path':"SiStrip/MechanicalView/TEC/MINUS/TotalNumberOfClusterProfile__TEC__MINUS",
      'description': "Total # of Clusters in TEC -ve side with event time in Seconds  - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftSiStrip>DQMShiftOnlineSiStrip</a> ", 'draw': { 'withref': "no" }},
   {  'path':"SiStrip/MechanicalView/TEC/PLUS/TotalNumberOfClusterProfile__TEC__PLUS",
       'description': "Total # of Clusters in TEC +ve side with event time in Seconds  - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftSiStrip>DQMShiftOnlineSiStrip</a> ", 'draw': { 'withref': "no" }}])

shiftcosmiclayout(dqmitems, "07 OnTrackCluster",
  [{ 'path': "SiStrip/MechanicalView/TIB/Summary_ClusterStoNCorr_OnTrack__TIB",
     'description': "Signal-to-Noise (corrected for the angle) for On-Track clusters in TIB  - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftSiStrip>DQMShiftOnlineSiStrip</a> ", 'draw': { 'withref': "yes" }},
   { 'path': "SiStrip/MechanicalView/TOB/Summary_ClusterStoNCorr_OnTrack__TOB",
     'description': "Signal-to-Noise (corrected for the angle) for On-Track clusters in TOB  - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftSiStrip>DQMShiftOnlineSiStrip</a> ", 'draw': { 'withref': "yes" } }],
  [{ 'path': "SiStrip/MechanicalView/TID/MINUS/Summary_ClusterStoNCorr_OnTrack__TID__MINUS",
     'description': "Signal-to-Noise (corrected for the angle) for On-Track clusters in TID -ve side - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftSiStrip>DQMShiftOnlineSiStrip</a> ", 'draw': { 'withref': "yes" }},
   { 'path': "SiStrip/MechanicalView/TID/PLUS/Summary_ClusterStoNCorr_OnTrack__TID__PLUS",
     'description': "Signal-to-Noise (corrected for the angle) for On-Track clusters in TID +ve side - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftSiStrip>DQMShiftOnlineSiStrip</a> ", 'draw': { 'withref': "yes" } }],
  [{ 'path': "SiStrip/MechanicalView/TEC/MINUS/Summary_ClusterStoNCorr_OnTrack__TEC__MINUS",
     'description': "Signal-to-Noise (corrected for the angle) for On-Track clusters in TEC -ve side - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftSiStrip>DQMShiftOnlineSiStrip</a> ", 'draw': { 'withref': "yes" }},
   { 'path': "SiStrip/MechanicalView/TEC/PLUS/Summary_ClusterStoNCorr_OnTrack__TEC__PLUS",
     'description': "Signal-to-Noise (corrected for the angle) for On-Track clusters in TEC +ve side - <a href=https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftSiStrip>DQMShiftOnlineSiStrip</a> ", 'draw': { 'withref': "yes" } }])

#CSC

shiftcosmiclayout(dqmitems,"08 CSC Physics Efficiency - RecHits Minus",
    [{'path': "CSC/CSCOfflineMonitor/recHits/hRHGlobalm1", 'description': "Histogram shows 2D RecHits distribution in ME-1. Any unusual inhomogeneity should be reported. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/viewauth/CMS/CSCDPGDataMonitorShiftInstructions\">here</a>."},
     {'path': "CSC/CSCOfflineMonitor/recHits/hRHGlobalm2", 'description': "Histogram shows 2D RecHits distribution in ME-2. Any unusual inhomogeneity should be reported. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/viewauth/CMS/CSCDPGDataMonitorShiftInstructions\">here</a>."}],
    [{'path': "CSC/CSCOfflineMonitor/recHits/hRHGlobalm3", 'description': "Histogram shows 2D RecHits distribution in ME-3. Any unusual inhomogeneity should be reported. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/viewauth/CMS/CSCDPGDataMonitorShiftInstructions\">here</a>."},
     {'path': "CSC/CSCOfflineMonitor/recHits/hRHGlobalm4", 'description': "Histogram shows 2D RecHits distribution in ME-4. Any unusual inhomogeneity should be reported. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/viewauth/CMS/CSCDPGDataMonitorShiftInstructions\">here</a>."}])

shiftcosmiclayout(dqmitems,"09 CSC Physics Efficiency - RecHits Plus",
    [{'path': "CSC/CSCOfflineMonitor/recHits/hRHGlobalp1", 'description': "Histogram shows 2D RecHits distribution in ME+1. Any unusual inhomogeneity should be reported. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/viewauth/CMS/CSCDPGDataMonitorShiftInstructions\">here</a>."},
     {'path': "CSC/CSCOfflineMonitor/recHits/hRHGlobalp2", 'description': "Histogram shows 2D RecHits distribution in ME+2. Any unusual inhomogeneity should be reported. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/viewauth/CMS/CSCDPGDataMonitorShiftInstructions\">here</a>."}],
    [{'path': "CSC/CSCOfflineMonitor/recHits/hRHGlobalp3", 'description': "Histogram shows 2D RecHits distribution in ME+3. Any unusual inhomogeneity should be reported. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/viewauth/CMS/CSCDPGDataMonitorShiftInstructions\">here</a>."},
     {'path': "CSC/CSCOfflineMonitor/recHits/hRHGlobalp4", 'description': "Histogram shows 2D RecHits distribution in ME+4. Any unusual inhomogeneity should be reported. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/viewauth/CMS/CSCDPGDataMonitorShiftInstructions\">here</a>."}])

#RPC

shiftcosmiclayout(dqmitems, "10 - RPC_Occupancy",
               [{ 'path': "RPC/AllHits/SummaryHistograms/Occupancy_for_Barrel", 'description': occupancy + rpclink  }],
               [{ 'path': "RPC/AllHits/SummaryHistograms/Occupancy_for_Endcap", 'description': occupancy + rpclink }])

shiftcosmiclayout(dqmitems, "11 - uGMT MUON P_{T}",
  [{
    'path': "L1T/L1TStage2uGMT/ugmtMuonPt",
    'description': "This should show normal pT spectrum (spikes at 140 GeV, 200 GeV and 255 GeV expected from max TF pT assigned)",
    'draw': { 'withref': "no" }
  }])
shiftcosmiclayout(dqmitems, "12 - uGMT MUON ETA",
  [{
    'path': "L1T/L1TStage2uGMT/ugmtMuonEta",
    'description': "This should have no spikes.",
    'draw': { 'withref': "no" }
  }])
shiftcosmiclayout(dqmitems, "13 - uGMT MUON PHI",
  [{
    'path': "L1T/L1TStage2uGMT/ugmtMuonPhi",
    'description': "This should have no spikes or dips.",
    'draw': { 'withref': "no" }
  }])

apply_dqm_items_to_new_back_end(dqmitems, __file__)
