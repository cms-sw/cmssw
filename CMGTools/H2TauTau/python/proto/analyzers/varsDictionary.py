# change the branch names here

vars = {}

# Event ID
vars['run'] = {'std': 'run', 'sync': 'run'}
vars['lumi'] = {'std': 'lumi', 'sync': 'lumi'}
vars['event'] = {'std': 'event', 'sync': 'evt'}

# Generator info
vars['geninfo_tt'] = {'std': 'geninfo_tt', 'sync': 'geninfo_tt'}
vars['geninfo_mt'] = {'std': 'geninfo_mt', 'sync': 'geninfo_mt'}
vars['geninfo_et'] = {'std': 'geninfo_et', 'sync': 'geninfo_et'}
vars['geninfo_ee'] = {'std': 'geninfo_ee', 'sync': 'geninfo_ee'}
vars['geninfo_mm'] = {'std': 'geninfo_mm', 'sync': 'geninfo_mm'}
vars['geninfo_em'] = {'std': 'geninfo_em', 'sync': 'geninfo_em'}
vars['geninfo_EE'] = {'std': 'geninfo_EE', 'sync': 'geninfo_EE'}
vars['geninfo_MM'] = {'std': 'geninfo_MM', 'sync': 'geninfo_MM'}
vars['geninfo_LL'] = {'std': 'geninfo_LL', 'sync': 'geninfo_LL'}
vars['geninfo_fakeid'] = {'std': 'geninfo_fakeid', 'sync': 'geninfo_fakeid'}

# Weights
vars['weight'] = {'std': 'weight', 'sync': 'weight'}
vars['weight_vertex'] = {'std': 'weight_vertex', 'sync': 'puweight'}

# PileUp
vars['geninfo_nup'] = {'geninfo_nup': 'NUP', 'sync': 'NUP'}
vars['n_vertices'] = {'std': 'n_vertices', 'sync': 'npv'}
vars['npu'] = {'std': 'npu', 'sync': 'npu'}
vars['rho'] = {'std': 'rho', 'sync': 'rho'}

# Leg 1 (tau, mu, ele)
vars['l1_pt'] = {'std': 'l1_pt', 'sync': 'pt_1'}
vars['l1_phi'] = {'std': 'l1_phi', 'sync': 'phi_1'}
vars['l1_eta'] = {'std': 'l1_eta', 'sync': 'eta_1'}
vars['l1_mass'] = {'std': 'l1_mass', 'sync': 'm_1'}
vars['l1_charge'] = {'std': 'l1_charge', 'sync': 'q_1'}
vars['l1_dxy'] = {'std': 'l1_dxy', 'sync': 'd0_1'}
vars['l1_dz'] = {'std': 'l1_dz', 'sync': 'dZ_1'}
vars['mt_leg1'] = {'std': 'mt_leg1', 'sync': 'mt_1'}
vars['l1_reliso05'] = {'std': 'l1_reliso05', 'sync': 'iso_1'}
vars['l1_muonid_loose'] = {'std': 'l1_muonid_loose', 'sync': 'id_m_loose_1'}
vars['l1_muonid_medium'] = {'std': 'l1_muonid_medium', 'sync': 'id_m_medium_1'}
vars['l1_muonid_tight'] = {'std': 'l1_muonid_tight', 'sync': 'id_m_tight_1'}
vars['l1_muonid_tightnovtx'] = {'std': 'l1_muonid_tightnovtx', 'sync': 'id_m_tightnovtx_1'}
vars['l1_muonid_highpt'] = {'std': 'l1_muonid_highpt', 'sync': 'id_m_highpt_1'}
vars['l1_eid_nontrigmva_loose'] = {'std': 'l1_eid_nontrigmva_loose', 'sync': 'id_e_mva_nt_loose_1'}
vars['l1_eid_nontrigmva_tight'] = {'std': 'l1_eid_nontrigmva_tight', 'sync': 'id_e_mva_nt_loose_1'}
vars['l1_eid_veto'] = {'std': 'l1_eid_veto', 'sync': 'id_e_cut_veto_1'}
vars['l1_eid_loose'] = {'std': 'l1_eid_loose', 'sync': 'id_e_cut_loose_1'}
vars['l1_eid_medium'] = {'std': 'l1_eid_medium', 'sync': 'id_e_cut_medium_1'}
vars['l1_eid_tight'] = {'std': 'l1_eid_tight', 'sync': 'id_e_cut_tight_1'}
vars['l1_weight_trigger'] = {'std': 'l1_weight_trigger', 'sync': 'trigweight_1'}
vars['l1_againstElectronLooseMVA5'] = {'std': 'l1_againstElectronLooseMVA5', 'sync': 'againstElectronLooseMVA5_1'}
vars['l1_againstElectronMediumMVA5'] = {'std': 'l1_againstElectronMediumMVA5', 'sync': 'againstElectronMediumMVA5_1'}
vars['l1_againstElectronTightMVA5'] = {'std': 'l1_againstElectronTightMVA5', 'sync': 'againstElectronTightMVA5_1'}
vars['l1_againstElectronVLooseMVA5'] = {'std': 'l1_againstElectronVLooseMVA5', 'sync': 'againstElectronVLooseMVA5_1'}
vars['l1_againstElectronVTightMVA5'] = {'std': 'l1_againstElectronVTightMVA5', 'sync': 'againstElectronVTightMVA5_1'}
vars['l1_againstMuonLoose3'] = {'std': 'l1_againstMuonLoose3', 'sync': 'againstMuonLoose3_1'}
vars['l1_againstMuonTight3'] = {'std': 'l1_againstMuonTight3', 'sync': 'againstMuonTight3_1'}
vars['l1_byCombinedIsolationDeltaBetaCorrRaw3Hits'] = {'std': 'l1_byCombinedIsolationDeltaBetaCorrRaw3Hits', 'sync': 'byCombinedIsolationDeltaBetaCorrRaw3Hits_1'}
vars['l1_byIsolationMVA3newDMwoLTraw'] = {'std': 'l1_byIsolationMVA3newDMwoLTraw', 'sync': 'byIsolationMVA3newDMwoLTraw_1'}
vars['l1_byIsolationMVA3oldDMwoLTraw'] = {'std': 'l1_byIsolationMVA3oldDMwoLTraw', 'sync': 'byIsolationMVA3oldDMwoLTraw_1'}
vars['l1_byIsolationMVA3newDMwLTraw'] = {'std': 'l1_byIsolationMVA3newDMwLTraw', 'sync': 'byIsolationMVA3newDMwLTraw_1'}
vars['l1_byIsolationMVA3oldDMwLTraw'] = {'std': 'l1_byIsolationMVA3oldDMwLTraw', 'sync': 'byIsolationMVA3oldDMwLTraw_1'}
vars['l1_chargedIsoPtSum'] = {'std': 'l1_chargedIsoPtSum', 'sync': 'chargedIsoPtSum_1'}
vars['l1_decayModeFinding'] = {'std': 'l1_decayModeFinding', 'sync': 'decayModeFinding_1'}
vars['l1_decayModeFindingNewDMs'] = {'std': 'l1_decayModeFindingNewDMs', 'sync': 'decayModeFindingNewDMs_1'}
vars['l1_neutralIsoPtSum'] = {'std': 'l1_neutralIsoPtSum', 'sync': 'neutralIsoPtSum_1'}
vars['l1_puCorrPtSum'] = {'std': 'l1_puCorrPtSum', 'sync': 'puCorrPtSum_1'}

# Leg 2 (tau, mu, ele)
vars['l2_pt'] = {'std': 'l2_pt', 'sync': 'pt_2'}
vars['l2_phi'] = {'std': 'l2_phi', 'sync': 'phi_2'}
vars['l2_eta'] = {'std': 'l2_eta', 'sync': 'eta_2'}
vars['l2_mass'] = {'std': 'l2_mass', 'sync': 'm_2'}
vars['l2_charge'] = {'std': 'l2_charge', 'sync': 'q_2'}
vars['l2_dxy'] = {'std': 'l2_dxy', 'sync': 'd0_2'}
vars['l2_dz'] = {'std': 'l2_dz', 'sync': 'dZ_2'}
vars['mt_leg2'] = {'std': 'mt_leg2', 'sync': 'mt_2'}
vars['l2_reliso05'] = {'std': 'l2_reliso05', 'sync': 'iso_2'}
vars['l2_muonid_loose'] = {'std': 'l2_muonid_loose', 'sync': 'id_m_loose_2'}
vars['l2_muonid_medium'] = {'std': 'l2_muonid_medium', 'sync': 'id_m_medium_2'}
vars['l2_muonid_tight'] = {'std': 'l2_muonid_tight', 'sync': 'id_m_tight_2'}
vars['l2_muonid_tightnovtx'] = {'std': 'l2_muonid_tightnovtx', 'sync': 'id_m_tightnovtx_2'}
vars['l2_muonid_highpt'] = {'std': 'l2_muonid_highpt', 'sync': 'id_m_highpt_2'}
vars['l2_eid_nontrigmva_loose'] = {'std': 'l2_eid_nontrigmva_loose', 'sync': 'id_e_mva_nt_loose_2'}
vars['l2_eid_nontrigmva_tight'] = {'std': 'l2_eid_nontrigmva_tight', 'sync': 'id_e_mva_nt_loose_2'}
vars['l2_eid_veto'] = {'std': 'l2_eid_veto', 'sync': 'id_e_cut_veto_2'}
vars['l2_eid_loose'] = {'std': 'l2_eid_loose', 'sync': 'id_e_cut_loose_2'}
vars['l2_eid_medium'] = {'std': 'l2_eid_medium', 'sync': 'id_e_cut_medium_2'}
vars['l2_eid_tight'] = {'std': 'l2_eid_tight', 'sync': 'id_e_cut_tight_2'}
vars['l2_weight_trigger'] = {'std': 'l2_weight_trigger', 'sync': 'trigweight_2'}
vars['l2_againstElectronLooseMVA5'] = {'std': 'l2_againstElectronLooseMVA5', 'sync': 'againstElectronLooseMVA5_2'}
vars['l2_againstElectronMediumMVA5'] = {'std': 'l2_againstElectronMediumMVA5', 'sync': 'againstElectronMediumMVA5_2'}
vars['l2_againstElectronTightMVA5'] = {'std': 'l2_againstElectronTightMVA5', 'sync': 'againstElectronTightMVA5_2'}
vars['l2_againstElectronVLooseMVA5'] = {'std': 'l2_againstElectronVLooseMVA5', 'sync': 'againstElectronVLooseMVA5_2'}
vars['l2_againstElectronVTightMVA5'] = {'std': 'l2_againstElectronVTightMVA5', 'sync': 'againstElectronVTightMVA5_2'}
vars['l2_againstMuonLoose3'] = {'std': 'l2_againstMuonLoose3', 'sync': 'againstMuonLoose3_2'}
vars['l2_againstMuonTight3'] = {'std': 'l2_againstMuonTight3', 'sync': 'againstMuonTight3_2'}
vars['l2_byCombinedIsolationDeltaBetaCorrRaw3Hits'] = {'std': 'l2_byCombinedIsolationDeltaBetaCorrRaw3Hits', 'sync': 'byCombinedIsolationDeltaBetaCorrRaw3Hits_2'}
vars['l2_byIsolationMVA3newDMwoLTraw'] = {'std': 'l2_byIsolationMVA3newDMwoLTraw', 'sync': 'byIsolationMVA3newDMwoLTraw_2'}
vars['l2_byIsolationMVA3oldDMwoLTraw'] = {'std': 'l2_byIsolationMVA3oldDMwoLTraw', 'sync': 'byIsolationMVA3oldDMwoLTraw_2'}
vars['l2_byIsolationMVA3newDMwLTraw'] = {'std': 'l2_byIsolationMVA3newDMwLTraw', 'sync': 'byIsolationMVA3newDMwLTraw_2'}
vars['l2_byIsolationMVA3oldDMwLTraw'] = {'std': 'l2_byIsolationMVA3oldDMwLTraw', 'sync': 'byIsolationMVA3oldDMwLTraw_2'}
vars['l2_chargedIsoPtSum'] = {'std': 'l2_chargedIsoPtSum', 'sync': 'chargedIsoPtSum_2'}
vars['l2_decayModeFinding'] = {'std': 'l2_decayModeFinding', 'sync': 'decayModeFinding_2'}
vars['l2_decayModeFindingNewDMs'] = {'std': 'l2_decayModeFindingNewDMs', 'sync': 'decayModeFindingNewDMs_2'}
vars['l2_neutralIsoPtSum'] = {'std': 'l2_neutralIsoPtSum', 'sync': 'neutralIsoPtSum_2'}
vars['l2_puCorrPtSum'] = {'std': 'l2_puCorrPtSum', 'sync': 'puCorrPtSum_2'}

# di-tau pair
vars['pthiggs'] = {'std': 'pthiggs', 'sync': 'pth'}
vars['visMass'] = {'std': 'visMass', 'sync': 'm_vis'}
vars['svfit_mass'] = {'std': 'svfit_mass', 'sync': 'm_sv'}
vars['svfit_pt'] = {'std': 'svfit_pt', 'sync': 'pt_sv'}
vars['svfit_eta'] = {'std': 'svfit_eta', 'sync': 'eta_sv'}
vars['svfit_phi'] = {'std': 'svfit_phi', 'sync': 'phi_sv'}
vars['svfit_met'] = {'std': 'svfit_met', 'sync': 'met_sv'}

# MET
vars['pfmet_pt'] = {'std': 'pfmet_pt', 'sync': 'met'}
vars['pfmet_phi'] = {'std': 'pfmet_phi', 'sync': 'metphi'}
vars['met_pt'] = {'std': 'met_pt', 'sync': 'mvamet'}
vars['met_phi'] = {'std': 'met_phi', 'sync': 'mvametphi'}
vars['pzeta_vis'] = {'std': 'pzeta_vis', 'sync': 'pzetavis'}
vars['pzeta_met'] = {'std': 'pzeta_met', 'sync': 'pzetamiss'}

vars['met_cov00'] = {'std': 'met_cov00', 'sync': 'mvacov00'}
vars['met_cov01'] = {'std': 'met_cov01', 'sync': 'mvacov01'}
vars['met_cov10'] = {'std': 'met_cov10', 'sync': 'mvacov10'}
vars['met_cov11'] = {'std': 'met_cov11', 'sync': 'mvacov11'}

# VBF
vars['ditau_mjj'] = {'std': 'ditau_mjj', 'sync': 'mjj'}
vars['ditau_deta'] = {'std': 'ditau_deta', 'sync': 'jdeta'}
vars['ditau_nCentral'] = {'std': 'ditau_nCentral', 'sync': 'njetingap'}
vars['ditau_mva'] = {'std': 'ditau_mva', 'sync': 'mva'}

vars['ditau_jdphi'] = {'std': 'ditau_jdphi', 'sync': 'jdphi'}
vars['ditau_dijetpt'] = {'std': 'ditau_dijetpt', 'sync': 'dijetpt'}
vars['ditau_dijetphi'] = {'std': 'ditau_dijetphi', 'sync': 'dijetphi'}
vars['ditau_hdijetphi'] = {'std': 'ditau_hdijetphi', 'sync': 'hdijetphi'}
vars['ditau_visjeteta'] = {'std': 'ditau_visjeteta', 'sync': 'visjeteta'}
vars['ditau_ptvis'] = {'std': 'ditau_ptvis', 'sync': 'ptvis'}

# N Jets
vars['n_jets'] = {'std': 'n_jets', 'sync': 'njets'}
vars['n_jets_20'] = {'std': 'n_jets_20', 'sync': 'njetspt20'}
vars['n_bjets'] = {'std': 'n_bjets', 'sync': 'nbtag'}

# Jet 1
vars['jet1_pt'] = {'std': 'jet1_pt', 'sync': 'jpt_1'}
vars['jet1_eta'] = {'std': 'jet1_eta', 'sync': 'jeta_1'}
vars['jet1_phi'] = {'std': 'jet1_phi', 'sync': 'jphi_1'}
vars['jet1_rawfactor'] = {'std': 'jet1_rawfactor', 'sync': 'jrawf_1'}
vars['jet1_mva_pu'] = {'std': 'jet1_mva_pu', 'sync': 'jmva_1'}
vars['jet1_id_loose'] = {'std': 'jet1_id_loose', 'sync': 'jpfid_1'}
vars['jet1_id_pu'] = {'std': 'jet1_id_pu', 'sync': 'jpuid_1'}
vars['jet1_csv'] = {'std': 'jet1_csv', 'sync': 'jcsv_1'}

# Jet 2
vars['jet2_pt'] = {'std': 'jet2_pt', 'sync': 'jpt_2'}
vars['jet2_eta'] = {'std': 'jet2_eta', 'sync': 'jeta_2'}
vars['jet2_phi'] = {'std': 'jet2_phi', 'sync': 'jphi_2'}
vars['jet2_rawfactor'] = {'std': 'jet2_rawfactor', 'sync': 'jrawf_2'}
vars['jet2_mva_pu'] = {'std': 'jet2_mva_pu', 'sync': 'jmva_2'}
vars['jet2_id_loose'] = {'std': 'jet2_id_loose', 'sync': 'jpfid_2'}
vars['jet2_id_pu'] = {'std': 'jet2_id_pu', 'sync': 'jpuid_2'}
vars['jet2_csv'] = {'std': 'jet2_csv', 'sync': 'jcsv_2'}

# bJet 1
vars['bjet1_pt'] = {'std': 'bjet1_pt', 'sync': 'bjpt_1'}
vars['bjet1_eta'] = {'std': 'bjet1_eta', 'sync': 'bjeta_1'}
vars['bjet1_phi'] = {'std': 'bjet1_phi', 'sync': 'bjphi_1'}
vars['bjet1_rawfactor'] = {'std': 'bjet1_rawfactor', 'sync': 'bjrawf_1'}
vars['bjet1_mva_pu'] = {'std': 'bjet1_mva_pu', 'sync': 'bjmva_1'}
vars['bjet1_id_loose'] = {'std': 'bjet1_id_loose', 'sync': 'bjpfid_1'}
vars['bjet1_id_pu'] = {'std': 'bjet1_id_pu', 'sync': 'bjpuid_1'}
vars['bjet1_csv'] = {'std': 'bjet1_csv', 'sync': 'bjcsv_1'}

# bJet 2
vars['bjet2_pt'] = {'std': 'bjet2_pt', 'sync': 'bjpt_2'}
vars['bjet2_eta'] = {'std': 'bjet2_eta', 'sync': 'bjeta_2'}
vars['bjet2_phi'] = {'std': 'bjet2_phi', 'sync': 'bjphi_2'}
vars['bjet2_rawfactor'] = {'std': 'bjet2_rawfactor', 'sync': 'bjrawf_2'}
vars['bjet2_mva_pu'] = {'std': 'bjet2_mva_pu', 'sync': 'bjmva_2'}
vars['bjet2_id_loose'] = {'std': 'bjet2_id_loose', 'sync': 'bjpfid_2'}
vars['bjet2_id_pu'] = {'std': 'bjet2_id_pu', 'sync': 'bjpuid_2'}
vars['bjet2_csv'] = {'std': 'bjet2_csv', 'sync': 'bjcsv_2'}
