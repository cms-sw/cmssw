from .adapt_to_new_backend import *
dqmitems={}

def csclayout(i, p, *rows): i["00 Shift/CSC/" + p] = rows

csclayout(dqmitems,"00 Chamber Status (Statistically Significant)",
    [{'path': "CSC/EventInfo/reportSummaryMap", 'description': "For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftCSC#reportSummaryMap\">here</a>."}])

csclayout(dqmitems,"01 Chamber Occupancy Exceptions (Statistically Significant)",
    [{'path': "CSC/Summary/CSC_STATS_occupancy", 'description': "For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftCSC#CSC_STATS_occupancy\">here</a>."}])

csclayout(dqmitems,"02 Chamber Errors and Warnings (Statistically Significant)",
    [{'path': "CSC/Summary/CSC_STATS_format_err", 'description': "For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftCSC#CSC_STATS_format_err\">here</a>."},
     {'path': "CSC/Summary/CSC_STATS_l1sync_err", 'description': "For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftCSC#CSC_STATS_l1sync_err\">here</a>."}],
    [{'path': "CSC/Summary/CSC_STATS_fifofull_err", 'description': "For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftCSC#CSC_STATS_fifofull_err\">here</a>."},
     {'path': "CSC/Summary/CSC_STATS_inputto_err", 'description': "For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftCSC#CSC_STATS_inputto_err\">here</a>."}])

csclayout(dqmitems,"03 Chambers without Data (Statistically Significant)",
    [{'path': "CSC/Summary/CSC_STATS_wo_alct", 'description': "For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftCSC#CSC_STATS_wo_alct\">here</a>."},
     {'path': "CSC/Summary/CSC_STATS_wo_clct", 'description': "For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftCSC#CSC_STATS_wo_clct\">here</a>."}],
    [{'path': "CSC/Summary/CSC_STATS_wo_cfeb", 'description': "For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftCSC#CSC_STATS_wo_cfeb\">here</a>."},
     {'path': "CSC/Summary/CSC_STATS_cfeb_bwords", 'description': "For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/view/CMS/DQMShiftCSC#CSC_STATS_cfeb_bwords\">here</a>."}])

csclayout(dqmitems,"04 Physics Efficiency - RecHits Minus",
    [{'path': "CSC/CSCOfflineMonitor/recHits/hRHGlobalm1", 'description': "Histogram shows 2D RecHits distribution in ME-1. Any unusual inhomogeneity should be reported. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/viewauth/CMS/CSCDPGDataMonitorShiftInstructions\">here</a>."},
     {'path': "CSC/CSCOfflineMonitor/recHits/hRHGlobalm2", 'description': "Histogram shows 2D RecHits distribution in ME-2. Any unusual inhomogeneity should be reported. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/viewauth/CMS/CSCDPGDataMonitorShiftInstructions\">here</a>."}],
    [{'path': "CSC/CSCOfflineMonitor/recHits/hRHGlobalm3", 'description': "Histogram shows 2D RecHits distribution in ME-3. Any unusual inhomogeneity should be reported. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/viewauth/CMS/CSCDPGDataMonitorShiftInstructions\">here</a>."},
     {'path': "CSC/CSCOfflineMonitor/recHits/hRHGlobalm4", 'description': "Histogram shows 2D RecHits distribution in ME-4. Any unusual inhomogeneity should be reported. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/viewauth/CMS/CSCDPGDataMonitorShiftInstructions\">here</a>."}])

csclayout(dqmitems,"05 Physics Efficiency - RecHits Plus",
    [{'path': "CSC/CSCOfflineMonitor/recHits/hRHGlobalp1", 'description': "Histogram shows 2D RecHits distribution in ME+1. Any unusual inhomogeneity should be reported. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/viewauth/CMS/CSCDPGDataMonitorShiftInstructions\">here</a>."},
     {'path': "CSC/CSCOfflineMonitor/recHits/hRHGlobalp2", 'description': "Histogram shows 2D RecHits distribution in ME+2. Any unusual inhomogeneity should be reported. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/viewauth/CMS/CSCDPGDataMonitorShiftInstructions\">here</a>."}],
    [{'path': "CSC/CSCOfflineMonitor/recHits/hRHGlobalp3", 'description': "Histogram shows 2D RecHits distribution in ME+3. Any unusual inhomogeneity should be reported. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/viewauth/CMS/CSCDPGDataMonitorShiftInstructions\">here</a>."},
     {'path': "CSC/CSCOfflineMonitor/recHits/hRHGlobalp4", 'description': "Histogram shows 2D RecHits distribution in ME+4. Any unusual inhomogeneity should be reported. For more information please click <a href=\"https://twiki.cern.ch/twiki/bin/viewauth/CMS/CSCDPGDataMonitorShiftInstructions\">here</a>."}])


apply_dqm_items_to_new_back_end(dqmitems, __file__)
