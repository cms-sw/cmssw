from .adapt_to_new_backend import *
dqmitems={}

def gemlayout(i, p, *rows): i["GEM/Layouts/" + p] = rows

_GEM_OFF_LINK = '<a href="https://twiki.cern.ch/twiki/bin/view/CMS/GEMPPDOfflineDQM">Link to TWiki</a>'

# Occupancy
gemlayout(dqmitems, "01 DIGI Occupancy",
    [{ "path": "GEM/GEMOfflineMonitor/Digi/digi_det_ge-11",
       "description": _GEM_OFF_LINK },
     { "path": "GEM/GEMOfflineMonitor/Digi/digi_det_ge+11",    
       "description": _GEM_OFF_LINK }])

gemlayout(dqmitems, "02 RecHit Occupancy",
    [{ "path": "GEM/GEMOfflineMonitor/RecHit/hit_det_ge-11",
       "description": _GEM_OFF_LINK },
     { "path": "GEM/GEMOfflineMonitor/RecHit/hit_det_ge+11",
       "description": _GEM_OFF_LINK }])

# Detector
gemlayout(dqmitems, "03 Efficiency - Tight GLB Muon",
    [{ "path": "GEM/GEMEfficiency/TightGlobalMuon/Efficiency/eff_detector_ge-11",
       "description": _GEM_OFF_LINK },
     { "path": "GEM/GEMEfficiency/TightGlobalMuon/Efficiency/eff_detector_ge+11",
       "description": _GEM_OFF_LINK }])

gemlayout(dqmitems, "04 Efficiency - STA Muon",
    [{ "path": "GEM/GEMEfficiency/StandaloneMuon/Efficiency/eff_detector_ge-11",
       "description": _GEM_OFF_LINK },
     { "path": "GEM/GEMEfficiency/StandaloneMuon/Efficiency/eff_detector_ge+11",
       "description": _GEM_OFF_LINK }])

# Efficiency vs. PT
gemlayout(dqmitems, "05 Efficiency vs Tight GLB Muon PT",
    [{ "path": "GEM/GEMEfficiency/TightGlobalMuon/Efficiency/eff_muon_pt_ge-11_odd",
       "description": _GEM_OFF_LINK },
     { "path": "GEM/GEMEfficiency/TightGlobalMuon/Efficiency/eff_muon_pt_ge+11_odd",
       "description": _GEM_OFF_LINK }],
    [{ "path": "GEM/GEMEfficiency/TightGlobalMuon/Efficiency/eff_muon_pt_ge-11_even",
       "description": _GEM_OFF_LINK },
     { "path": "GEM/GEMEfficiency/TightGlobalMuon/Efficiency/eff_muon_pt_ge+11_even",
       "description": _GEM_OFF_LINK }])

gemlayout(dqmitems, "06 Efficiency vs STA Muon PT",
    [{ "path": "GEM/GEMEfficiency/StandaloneMuon/Efficiency/eff_muon_pt_ge-11_odd",
       "description": _GEM_OFF_LINK },
     { "path": "GEM/GEMEfficiency/StandaloneMuon/Efficiency/eff_muon_pt_ge+11_odd",
       "description": _GEM_OFF_LINK }],
    [{ "path": "GEM/GEMEfficiency/StandaloneMuon/Efficiency/eff_muon_pt_ge-11_even",
       "description": _GEM_OFF_LINK },
     { "path": "GEM/GEMEfficiency/StandaloneMuon/Efficiency/eff_muon_pt_ge+11_even",
       "description": _GEM_OFF_LINK }])

# Efficiency vs. Eta
gemlayout(dqmitems, "07 Efficiency vs Tight GLB Muon Eta",
    [{ "path": "GEM/GEMEfficiency/TightGlobalMuon/Efficiency/eff_muon_eta_ge-11_odd",
       "description": _GEM_OFF_LINK },
     { "path": "GEM/GEMEfficiency/TightGlobalMuon/Efficiency/eff_muon_eta_ge+11_odd",
       "description": _GEM_OFF_LINK }],
    [{ "path": "GEM/GEMEfficiency/TightGlobalMuon/Efficiency/eff_muon_eta_ge-11_even",
       "description": _GEM_OFF_LINK },
     { "path": "GEM/GEMEfficiency/TightGlobalMuon/Efficiency/eff_muon_eta_ge+11_even",
       "description": _GEM_OFF_LINK }])


gemlayout(dqmitems, "08 Efficiency vs STA Muon Eta",
    [{ "path": "GEM/GEMEfficiency/StandaloneMuon/Efficiency/eff_muon_eta_ge-11_odd",
       "description": _GEM_OFF_LINK },
     { "path": "GEM/GEMEfficiency/StandaloneMuon/Efficiency/eff_muon_eta_ge+11_odd",
       "description": _GEM_OFF_LINK }],
    [{ "path": "GEM/GEMEfficiency/StandaloneMuon/Efficiency/eff_muon_eta_ge-11_even",
       "description": _GEM_OFF_LINK },
     { "path": "GEM/GEMEfficiency/StandaloneMuon/Efficiency/eff_muon_eta_ge+11_even",
       "description": _GEM_OFF_LINK }])

# Phi Residual Histogram Parammeters
gemlayout(dqmitems, "09 Phi Residual - Tight GLB Muon",
    [{ "path": "GEM/GEMEfficiency/TightGlobalMuon/Resolution/residual_phi_ge-11_odd",
       "description": _GEM_OFF_LINK },
     { "path": "GEM/GEMEfficiency/TightGlobalMuon/Resolution/residual_phi_ge+11_odd",
       "description": _GEM_OFF_LINK }],
    [ { "path": "GEM/GEMEfficiency/TightGlobalMuon/Resolution/residual_phi_ge-11_even",
        "description": _GEM_OFF_LINK },
      { "path": "GEM/GEMEfficiency/TightGlobalMuon/Resolution/residual_phi_ge+11_even",
        "description": _GEM_OFF_LINK }])

gemlayout(dqmitems, "10 Phi Residual - STA Muon",
    [{ "path": "GEM/GEMEfficiency/StandaloneMuon/Resolution/residual_phi_ge-11_odd",
       "description": _GEM_OFF_LINK },
     { "path": "GEM/GEMEfficiency/StandaloneMuon/Resolution/residual_phi_ge+11_odd",
       "description": _GEM_OFF_LINK }],
    [{ "path": "GEM/GEMEfficiency/StandaloneMuon/Resolution/residual_phi_ge-11_even",
       "description": _GEM_OFF_LINK },
     { "path": "GEM/GEMEfficiency/StandaloneMuon/Resolution/residual_phi_ge+11_even",
       "description": _GEM_OFF_LINK }])

apply_dqm_items_to_new_back_end(dqmitems, __file__)
