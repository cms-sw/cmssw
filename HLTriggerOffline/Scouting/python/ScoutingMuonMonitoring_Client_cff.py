'''
Client config file for Scouting Muon DQM. Harvester to compute the efficiencies
for the Tag and Probe (hltScoutingMuonPackerVtx and hltScoutingMuonPackerNoVtx 
collections, read in ScoutingMuonTagProbeAnalyzer_cfi.py) and compute the efficiencies
of the L1 seeds.

Author: Javier Garcia de Castro, email:javigdc@bu.edu
'''

import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester
from HLTriggerOffline.Scouting.ScoutingMuonTriggerAnalyzer_cfi import *

#Harvester to measure efficiency for hltScoutingMuonPackerNoVtx collection (Tag and Probe method)
#Inputs for the efficiency vstring are (name, title, xlabel, ylabel, numerator histogram, denominator histogram)
muonEfficiencyNoVtx = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("/HLT/ScoutingOffline/Muons/NoVtx"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages                                                                                                                                          
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "effic_muonPt_JPsi       'efficiency vs pt (JPsi); Muon pt [GeV]; efficiency' resonanceJ_numerator_Probe_sctMuon_Pt       resonanceJ_denominator_Probe_sctMuon_Pt",
         "effic_muonEta_JPsi   'efficiency vs eta (JPsi); Muon eta; efficiency' resonanceJ_numerator_Probe_sctMuon_Eta       resonanceJ_denominator_Probe_sctMuon_Eta",
         "effic_muonPhi_JPsi       'efficiency vs phi (JPsi); Muon phi; efficiency' resonanceJ_numerator_Probe_sctMuon_Phi       resonanceJ_denominator_Probe_sctMuon_Phi",
         "effic_muondxy_JPsi       'efficiency vs dxy (JPsi); Muon dxy; efficiency' resonanceJ_numerator_Probe_sctMuon_dxy       resonanceJ_denominator_Probe_sctMuon_dxy",
         "effic_muonInvMass_JPsi       'efficiency vs inv mass (JPsi); Muon inv mass [GeV]; efficiency' resonanceJ_numerator_sctMuon_Invariant_Mass       resonanceJ_denominator_sctMuon_Invariant_Mass",
         "effic_muonNormChisq _JPsi     'efficiency vs normChi2 (JPsi); Muon normChi2; efficiency' resonanceJ_numerator_Probe_sctMuon_NormChisq       resonanceJ_denominator_Probe_sctMuon_NormChisq",
         "effic_muonTrkdz_JPsi      'efficiency vs Trkdz (JPsi); Muon trkdz; efficiency' resonanceJ_numerator_Probe_sctMuon_Trk_dz       resonanceJ_denominator_Probe_sctMuon_Trkdz",
         "effic_muonTrkdxy_JPsi       'efficiency vs Trkdxy (JPsi); Muon trkdxy; efficiency' resonanceJ_numerator_Probe_sctMuon_Trk_dxy       resonanceJ_denominator_Probe_sctMuon_Trkdxy",
         "effic_muonlxy_JPsi       'efficiency vs lxy (JPsi); Muon lxy; efficiency' resonanceJ_numerator_Vertex_Lxy       resonanceJ_denominator_Vertex_Lxy",
         "effic_muonVertexYerror_JPsi       'efficiency vs VertexYerror (JPsi); Muon Vertex Yerror; efficiency' resonanceJ_numerator_Vertex_Yerror       resonanceJ_denominator_Vertex_Yerror",
         "effic_muonVertexXerror_JPsi       'efficiency vs VertexXerror (JPsi); Muon Vertex Xerror; efficiency' resonanceJ_numerator_Vertex_Xerror       resonanceJ_denominator_Vertex_Xerror",
         "effic_muonVertexChi2_JPsi       'efficiency vs Vertexchi2 (JPsi); Muon Vertex chi2; efficiency' resonanceJ_numerator_Vertex_chi2       resonanceJ_denominator_Vertex_chi2",
         "effic_muonVertexYerror_JPsi       'efficiency vs z (JPsi); Muon Vertex z; efficiency' resonanceJ_numerator_Vertex_z       resonanceJ_denominator_Vertex_z",
         "effic_muontype_JPsi      'efficiency vs type (JPsi); Muon type; efficiency' resonanceJ_numerator_Probe_sctMuon_type      resonanceJ_denominator_Probe_sctMuon_type",
         "effic_muoncharge_JPsi       'efficiency vs charge (JPsi); Muon charge; efficiency' resonanceJ_numerator_Probe_sctMuon_charge       resonanceJ_denominator_Probe_sctMuon_charge",
         "effic_muonecalIso_JPsi       'efficiency vs ecalIso (JPsi); Muon ecalIso; efficiency' resonanceJ_numerator_Probe_sctMuon_ecalIso       resonanceJ_denominator_Probe_sctMuon_ecalIso",
         "effic_muonhcalIso_JPsi       'efficiency vs hcalIso (JPsi); Muon hcalIso; efficiency' resonanceJ_numerator_Probe_sctMuon_hcalIso       resonanceJ_denominator_Probe_sctMuon_hcalIso",
         "effic_muontrackIso_JPsi       'efficiency vs trackIso (JPsi); Muon trackIso; efficiency' resonanceJ_numerator_Probe_sctMuon_trackIso       resonanceJ_denominator_Probe_sctMuon_trackIso",
         "effic_nValidStandAloneMuonHits_JPsi       'efficiency vs nValidStandAloneMuonHits (JPsi); nValidStandAloneMuonHits; efficiency' resonanceJ_numerator_Probe_sctMuon_nValidStandAloneMuonHits       resonanceJ_denominator_Probe_sctMuon_nValidStandAloneMuonHits",
         "effic_nStandAloneMuonMatchedStations_JPsi       'efficiency vs nStandAloneMuonMatchedStations (JPsi); nStandAloneMuonMatchedStations; efficiency' resonanceJ_numerator_Probe_sctMuon_nStandAloneMuonMatchedStations       resonanceJ_denominator_Probe_sctMuon_nStandAloneMuonMatchedStations",
         "effic_nValidRecoMuonHits_JPsi       'efficiency vs nValidRecoMuonHits (JPsi); nValidRecoMuonHits; efficiency' resonanceJ_numerator_Probe_sctMuon_nValidRecoMuonHits       resonanceJ_denominator_Probe_sctMuon_nValidRecoMuonHits",
         "effic_nRecoMuonChambers_JPsi       'efficiency vs nRecoMuonChambers (JPsi); nRecoMuonChambers; efficiency' resonanceJ_numerator_Probe_sctMuon_nRecoMuonChambers       resonanceJ_denominator_Probe_sctMuon_nRecoMuonChambers",
         "effic_nRecoMuonChambersCSCorDT_JPsi       'efficiency vs nRecoMuonChambersCSCorDT (JPsi); nRecoMuonChambersCSCorDT; efficiency' resonanceJ_numerator_Probe_sctMuon_nRecoMuonChambersCSCorDT       resonanceJ_denominator_Probe_sctMuon_nRecoMuonChambersCSCorDT",
         "effic_nRecoMuonMatches_JPsi       'efficiency vs nRecoMuonMatches (JPsi); nRecoMuonMatches; efficiency' resonanceJ_numerator_Probe_sctMuon_nRecoMuonMatches       resonanceJ_denominator_Probe_sctMuon_nRecoMuonMatches",
         "effic_nRecoMuonMatchedStations_JPsi       'efficiency vs nRecoMuonMatchedStations (JPsi); nRecoMuonMatchedStations; efficiency' resonanceJ_numerator_Probe_sctMuon_nRecoMuonMatchedStations       resonanceJ_denominator_Probe_sctMuon_nRecoMuonMatchedStations",
         "effic_nRecoMuonExpectedMatchedStations_JPsi      'efficiency vs nRecoMuonExpectedMatchedStations (JPsi); nRecoMuonExpectedMatchedStations; efficiency' resonanceJ_numerator_Probe_sctMuon_nRecoMuonExpectedMatchedStations       resonanceJ_denominator_Probe_sctMuon_nRecoMuonExpectedMatchedStations",
         "effic_nRecoMuonMatchedRPCLayers_JPsi       'efficiency vs nRecoMuonMatchedRPCLayers (JPsi); nRecoMuonMatchedRPCLayers; efficiency' resonanceJ_numerator_Probe_sctMuon_nRecoMuonMatchedRPCLayers       resonanceJ_denominator_Probe_sctMuon_nRecoMuonMatchedRPCLayers",
         "effic_recoMuonRPClayerMask_JPsi      'efficiency vs recoMuonRPClayerMask (JPsi); recoMuonRPClayerMask; efficiency' resonanceJ_numerator_Probe_sctMuon_recoMuonRPClayerMask       resonanceJ_denominator_Probe_sctMuon_recoMuonRPClayerMask",
         "effic_nValidPixelHits_JPsi       'efficiency vs nValidPixelHits (JPsi); nValidPixelHits; efficiency' resonanceJ_numerator_Probe_sctMuon_nValidPixelHits       resonanceJ_denominator_Probe_sctMuon_nValidPixelHits",
         "effic_nValidStripHits_JPsi       'efficiency vs nValidStripHits (JPsi); nValidStripHits; efficiency' resonanceJ_numerator_Probe_sctMuon_nValidStripHits       resonanceJ_denominator_Probe_sctMuon_nValidStripHits",
         "effic_nPixelLayersWithMeasurement_JPsi       'efficiency vs nPixelLayersWithMeasurement (JPsi); nPixelLayersWithMeasurement; efficiency' resonanceJ_numerator_Probe_sctMuon_nPixelLayersWithMeasurement       resonanceJ_denominator_Probe_sctMuon_nPixelLayersWithMeasurement",
         "effic_nTrackerLayersWithMeasurement_JPsi       'efficiency vs nTrackerLayersWithMeasurement (JPsi); nTrackerLayersWithMeasurement; efficiency' resonanceJ_numerator_Probe_sctMuon_nTrackerLayersWithMeasurement       resonanceJ_denominator_Probe_sctMuon_nTrackerLayersWithMeasurement",
         "effic_trk_chi2_JPsi       'efficiency vs trk_chi2 (JPsi); trk_chi2; efficiency' resonanceJ_numerator_Probe_sctMuon_trk_chi2      resonanceJ_denominator_Probe_sctMuon_trk_chi2",
         "effic_trk_ndof_JPsi       'efficiency vs trk_ndof (JPsi); trk_ndof; efficiency' resonanceJ_numerator_Probe_sctMuon_trk_ndof      resonanceJ_denominator_Probe_sctMuon_trk_ndof",
         "effic_trk_lambda_JPsi       'efficiency vs trk_lambda (JPsi); trk_lambda; efficiency' resonanceJ_numerator_Probe_sctMuon_trk_lambda       resonanceJ_denominator_Probe_sctMuon_trk_lambda",
         "effic_trk_pt_JPsi       'efficiency vs trk_pt (JPsi); trk_pt; efficiency' resonanceJ_numerator_Probe_sctMuon_trk_pt       resonanceJ_denominator_Probe_sctMuon_trk_pt",
         "effic_trk_eta_JPsi      'efficiency vs trk_eta (JPsi); trk_eta; efficiency' resonanceJ_numerator_Probe_sctMuon_trk_eta    resonanceJ_denominator_Probe_sctMuon_trk_eta",
         "effic_trk_dxyError_JPsi       'efficiency vs trk_dxyError (JPsi); trk_dxyError; efficiency' resonanceJ_numerator_Probe_sctMuon_trk_dxyError       resonanceJ_denominator_Probe_sctMuon_trk_dxyError",
         "effic_trk_dzError_JPsi       'efficiency vs trk_dzError (JPsi); trk_dzError; efficiency' resonanceJ_numerator_Probe_sctMuon_trk_dzError       resonanceJ_denominator_Probe_sctMuon_trk_dzError",
         "effic_trk_qoverpError_JPsi       'efficiency vs trk_qoverpError (JPsi); trk_qoverpError; efficiency' resonanceJ_numerator_Probe_sctMuon_trk_qoverpError       resonanceJ_denominator_Probe_sctMuon_trk_qoverpError",
         "effic_trk_lambdaError_JPsi      'efficiency vs trk_lambdaError (JPsi); trk_lambdaError; efficiency' resonanceJ_numerator_Probe_sctMuon_trk_lambdaError       resonanceJ_denominator_Probe_sctMuon_trk_lambdaError",
         "effic_trk_phiError_JPsi       'efficiency vs trk_phiError (JPsi); trk_phiError; efficiency' resonanceJ_numerator_Probe_sctMuon_trk_phiError       resonanceJ_denominator_Probe_sctMuon_trk_phiError",
         "effic_trk_dsz_JPsi      'efficiency vs trk_dsz (JPsi); trk_dsz; efficiency' resonanceJ_numerator_Probe_sctMuon_trk_dsz       resonanceJ_denominator_Probe_sctMuon_trk_dsz",
         "effic_trk_dszError_JPsi       'efficiency vs trk_dszError (JPsi); trk_dszError; efficiency' resonanceJ_numerator_Probe_sctMuon_trk_dszError       resonanceJ_denominator_Probe_sctMuon_trk_dszError",
         "effic_ndof_JPsi       'efficiency vs ndof (JPsi); ndof; efficiency' resonanceJ_numerator_Probe_sctMuon_ndof      resonanceJ_denominator_Probe_sctMuon_ndof",
         "effic_trk_vx_JPsi       'efficiency vs trk_vx (JPsi); trk_vx; efficiency' resonanceJ_numerator_Probe_sctMuon_trk_vx       resonanceJ_denominator_Probe_sctMuon_trk_vx",
         "effic_trk_vy_JPsi       'efficiency vs trk_vy (JPsi); trk_vy; efficiency' resonanceJ_numerator_Probe_sctMuon_trk_vy       resonanceJ_denominator_Probe_sctMuon_trk_vy",
         "effic_trk_vz_JPsi       'efficiency vs trk_vz (JPsi); trk_vz; efficiency' resonanceJ_numerator_Probe_sctMuon_trk_vz      resonanceJ_denominator_Probe_sctMuon_trk_vz",
         "effic_vertex_x_JPsi       'efficiency vs vertex_x (JPsi); vertex x; efficiency' resonanceJ_numerator_Probe_sctMuon_x       resonanceJ_denominator_Probe_sctMuon_x",
         "effic_vertex_y_JPsi       'efficiency vs vertex_y (JPsi); vertex y; efficiency' resonanceJ_numerator_Probe_sctMuon_y       resonanceJ_denominator_Probe_sctMuon_y",
         "effic_vertex_Zerror_JPsi       'efficiency vs Zerror (JPsi); vertex Zerror; efficiency' resonanceJ_numerator_Probe_sctMuon_Zerror       resonanceJ_denominator_Probe_sctMuon_Zerror",
         "effic_tracksSize_JPsi       'efficiency vs tracksSize (JPsi); tracksSize; efficiency' resonanceJ_numerator_Probe_sctMuon_tracksSize       resonanceJ_denominator_Probe_sctMuon_tracksSize",
         "effic_muonPt_Z       'efficiency vs pt (Z); Muon pt [GeV]; efficiency' resonanceZ_numerator_Probe_sctMuon_Pt       resonanceZ_denominator_Probe_sctMuon_Pt",
         "effic_muonEta_Z   'efficiency vs eta (Z); Muon eta; efficiency' resonanceZ_numerator_Probe_sctMuon_Eta       resonanceZ_denominator_Probe_sctMuon_Eta",
         "effic_muonPhi_Z       'efficiency vs phi (Z); Muon phi; efficiency' resonanceZ_numerator_Probe_sctMuon_Phi       resonanceZ_denominator_Probe_sctMuon_Phi",
         "effic_muondxy_Z       'efficiency vs dxy (Z); Muon dxy; efficiency' resonanceZ_numerator_Probe_sctMuon_dxy       resonanceZ_denominator_Probe_sctMuon_dxy",
         "effic_muonInvMass_Z       'efficiency vs inv mass (Z); Muon inv mass [GeV]; efficiency' resonanceZ_numerator_sctMuon_Invariant_Mass       resonanceZ_denominator_sctMuon_Invariant_Mass",
         "effic_muonNormChisq _Z     'efficiency vs normChi2 (Z); Muon normChi2; efficiency' resonanceZ_numerator_Probe_sctMuon_NormChisq       resonanceZ_denominator_Probe_sctMuon_NormChisq",
         "effic_muonTrkdz_Z      'efficiency vs Trkdz (Z); Muon trkdz; efficiency' resonanceZ_numerator_Probe_sctMuon_Trk_dz       resonanceZ_denominator_Probe_sctMuon_Trkdz",
         "effic_muonTrkdxy_Z       'efficiency vs Trkdxy (Z); Muon trkdxy; efficiency' resonanceZ_numerator_Probe_sctMuon_Trk_dxy       resonanceZ_denominator_Probe_sctMuon_Trkdxy",
         "effic_muonlxy_Z       'efficiency vs lxy (Z); Muon lxy; efficiency' resonanceZ_numerator_Vertex_Lxy       resonanceZ_denominator_Vertex_Lxy",
         "effic_muonVertexYerror_Z       'efficiency vs VertexYerror (Z); Muon Vertex Yerror; efficiency' resonanceZ_numerator_Vertex_Yerror       resonanceZ_denominator_Vertex_Yerror",
         "effic_muonVertexXerror_Z       'efficiency vs VertexXerror (Z); Muon Vertex Xerror; efficiency' resonanceZ_numerator_Vertex_Xerror       resonanceZ_denominator_Vertex_Xerror",
         "effic_muonVertexChi2_Z       'efficiency vs Vertexchi2 (Z); Muon Vertex chi2; efficiency' resonanceZ_numerator_Vertex_chi2       resonanceZ_denominator_Vertex_chi2",
         "effic_muonVertexYerror_Z       'efficiency vs z (Z); Muon Vertex z; efficiency' resonanceZ_numerator_Vertex_z       resonanceZ_denominator_Vertex_z",
         "effic_muontype_Z      'efficiency vs type (Z); Muon type; efficiency' resonanceZ_numerator_Probe_sctMuon_type      resonanceZ_denominator_Probe_sctMuon_type",
         "effic_muoncharge_Z       'efficiency vs charge (Z); Muon charge; efficiency' resonanceZ_numerator_Probe_sctMuon_charge       resonanceZ_denominator_Probe_sctMuon_charge",
         "effic_muonecalIso_Z       'efficiency vs ecalIso (Z); Muon ecalIso; efficiency' resonanceZ_numerator_Probe_sctMuon_ecalIso       resonanceZ_denominator_Probe_sctMuon_ecalIso",
         "effic_muonhcalIso_Z       'efficiency vs hcalIso (Z); Muon hcalIso; efficiency' resonanceZ_numerator_Probe_sctMuon_hcalIso       resonanceZ_denominator_Probe_sctMuon_hcalIso",
         "effic_muontrackIso_Z       'efficiency vs trackIso (Z); Muon trackIso; efficiency' resonanceZ_numerator_Probe_sctMuon_trackIso       resonanceZ_denominator_Probe_sctMuon_trackIso",
         "effic_nValidStandAloneMuonHits_Z       'efficiency vs nValidStandAloneMuonHits (Z); nValidStandAloneMuonHits; efficiency' resonanceZ_numerator_Probe_sctMuon_nValidStandAloneMuonHits       resonanceZ_denominator_Probe_sctMuon_nValidStandAloneMuonHits",
         "effic_nStandAloneMuonMatchedStations_Z       'efficiency vs nStandAloneMuonMatchedStations (Z); nStandAloneMuonMatchedStations; efficiency' resonanceZ_numerator_Probe_sctMuon_nStandAloneMuonMatchedStations       resonanceZ_denominator_Probe_sctMuon_nStandAloneMuonMatchedStations",
         "effic_nValidRecoMuonHits_Z       'efficiency vs nValidRecoMuonHits (Z); nValidRecoMuonHits; efficiency' resonanceZ_numerator_Probe_sctMuon_nValidRecoMuonHits       resonanceZ_denominator_Probe_sctMuon_nValidRecoMuonHits",
         "effic_nRecoMuonChambers_Z       'efficiency vs nRecoMuonChambers (Z); nRecoMuonChambers; efficiency' resonanceZ_numerator_Probe_sctMuon_nRecoMuonChambers       resonanceZ_denominator_Probe_sctMuon_nRecoMuonChambers",
         "effic_nRecoMuonChambersCSCorDT_Z       'efficiency vs nRecoMuonChambersCSCorDT (Z); nRecoMuonChambersCSCorDT; efficiency' resonanceZ_numerator_Probe_sctMuon_nRecoMuonChambersCSCorDT       resonanceZ_denominator_Probe_sctMuon_nRecoMuonChambersCSCorDT",
         "effic_nRecoMuonMatches_Z       'efficiency vs nRecoMuonMatches (Z); nRecoMuonMatches; efficiency' resonanceZ_numerator_Probe_sctMuon_nRecoMuonMatches       resonanceZ_denominator_Probe_sctMuon_nRecoMuonMatches",
         "effic_nRecoMuonMatchedStations_Z       'efficiency vs nRecoMuonMatchedStations (Z); nRecoMuonMatchedStations; efficiency' resonanceZ_numerator_Probe_sctMuon_nRecoMuonMatchedStations       resonanceZ_denominator_Probe_sctMuon_nRecoMuonMatchedStations",
         "effic_nRecoMuonExpectedMatchedStations_Z      'efficiency vs nRecoMuonExpectedMatchedStations (Z); nRecoMuonExpectedMatchedStations; efficiency' resonanceZ_numerator_Probe_sctMuon_nRecoMuonExpectedMatchedStations       resonanceZ_denominator_Probe_sctMuon_nRecoMuonExpectedMatchedStations",
         "effic_nRecoMuonMatchedRPCLayers_Z       'efficiency vs nRecoMuonMatchedRPCLayers (Z); nRecoMuonMatchedRPCLayers; efficiency' resonanceZ_numerator_Probe_sctMuon_nRecoMuonMatchedRPCLayers       resonanceZ_denominator_Probe_sctMuon_nRecoMuonMatchedRPCLayers",
         "effic_recoMuonRPClayerMask_Z      'efficiency vs recoMuonRPClayerMask (Z); recoMuonRPClayerMask; efficiency' resonanceZ_numerator_Probe_sctMuon_recoMuonRPClayerMask       resonanceZ_denominator_Probe_sctMuon_recoMuonRPClayerMask",
         "effic_nValidPixelHits_Z       'efficiency vs nValidPixelHits (Z); nValidPixelHits; efficiency' resonanceZ_numerator_Probe_sctMuon_nValidPixelHits       resonanceZ_denominator_Probe_sctMuon_nValidPixelHits",
         "effic_nValidStripHits_Z       'efficiency vs nValidStripHits (Z); nValidStripHits; efficiency' resonanceZ_numerator_Probe_sctMuon_nValidStripHits       resonanceZ_denominator_Probe_sctMuon_nValidStripHits",
         "effic_nPixelLayersWithMeasurement_Z       'efficiency vs nPixelLayersWithMeasurement (Z); nPixelLayersWithMeasurement; efficiency' resonanceZ_numerator_Probe_sctMuon_nPixelLayersWithMeasurement       resonanceZ_denominator_Probe_sctMuon_nPixelLayersWithMeasurement",
         "effic_nTrackerLayersWithMeasurement_Z       'efficiency vs nTrackerLayersWithMeasurement (Z); nTrackerLayersWithMeasurement; efficiency' resonanceZ_numerator_Probe_sctMuon_nTrackerLayersWithMeasurement       resonanceZ_denominator_Probe_sctMuon_nTrackerLayersWithMeasurement",
         "effic_trk_chi2_Z       'efficiency vs trk_chi2 (Z); trk_chi2; efficiency' resonanceZ_numerator_Probe_sctMuon_trk_chi2      resonanceZ_denominator_Probe_sctMuon_trk_chi2",
         "effic_trk_ndof_Z       'efficiency vs trk_ndof (Z); trk_ndof; efficiency' resonanceZ_numerator_Probe_sctMuon_trk_ndof      resonanceZ_denominator_Probe_sctMuon_trk_ndof",
         "effic_trk_lambda_Z       'efficiency vs trk_lambda (Z); trk_lambda; efficiency' resonanceZ_numerator_Probe_sctMuon_trk_lambda       resonanceZ_denominator_Probe_sctMuon_trk_lambda",
         "effic_trk_pt_Z       'efficiency vs trk_pt (Z); trk_pt; efficiency' resonanceZ_numerator_Probe_sctMuon_trk_pt       resonanceZ_denominator_Probe_sctMuon_trk_pt",
         "effic_trk_eta_Z      'efficiency vs trk_eta (Z); trk_eta; efficiency' resonanceZ_numerator_Probe_sctMuon_trk_eta    resonanceZ_denominator_Probe_sctMuon_trk_eta",
         "effic_trk_dxyError_Z       'efficiency vs trk_dxyError (Z); trk_dxyError; efficiency' resonanceZ_numerator_Probe_sctMuon_trk_dxyError       resonanceZ_denominator_Probe_sctMuon_trk_dxyError",
         "effic_trk_dzError_Z       'efficiency vs trk_dzError (Z); trk_dzError; efficiency' resonanceZ_numerator_Probe_sctMuon_trk_dzError       resonanceZ_denominator_Probe_sctMuon_trk_dzError",
         "effic_trk_qoverpError_Z       'efficiency vs trk_qoverpError (Z); trk_qoverpError; efficiency' resonanceZ_numerator_Probe_sctMuon_trk_qoverpError       resonanceZ_denominator_Probe_sctMuon_trk_qoverpError",
         "effic_trk_lambdaError_Z      'efficiency vs trk_lambdaError (Z); trk_lambdaError; efficiency' resonanceZ_numerator_Probe_sctMuon_trk_lambdaError       resonanceZ_denominator_Probe_sctMuon_trk_lambdaError",
         "effic_trk_phiError_Z       'efficiency vs trk_phiError (Z); trk_phiError; efficiency' resonanceZ_numerator_Probe_sctMuon_trk_phiError       resonanceZ_denominator_Probe_sctMuon_trk_phiError",
         "effic_trk_dsz_Z      'efficiency vs trk_dsz (Z); trk_dsz; efficiency' resonanceZ_numerator_Probe_sctMuon_trk_dsz       resonanceZ_denominator_Probe_sctMuon_trk_dsz",
         "effic_trk_dszError_Z       'efficiency vs trk_dszError (Z); trk_dszError; efficiency' resonanceZ_numerator_Probe_sctMuon_trk_dszError       resonanceZ_denominator_Probe_sctMuon_trk_dszError",
         "effic_ndof_Z       'efficiency vs ndof (Z); ndof; efficiency' resonanceZ_numerator_Probe_sctMuon_ndof      resonanceZ_denominator_Probe_sctMuon_ndof",
         "effic_trk_vx_Z       'efficiency vs trk_vx (Z); trk_vx; efficiency' resonanceZ_numerator_Probe_sctMuon_trk_vx       resonanceZ_denominator_Probe_sctMuon_trk_vx",
         "effic_trk_vy_Z       'efficiency vs trk_vy (Z); trk_vy; efficiency' resonanceZ_numerator_Probe_sctMuon_trk_vy       resonanceZ_denominator_Probe_sctMuon_trk_vy",
         "effic_trk_vz_Z       'efficiency vs trk_vz (Z); trk_vz; efficiency' resonanceZ_numerator_Probe_sctMuon_trk_vz      resonanceZ_denominator_Probe_sctMuon_trk_vz",
         "effic_vertex_x_Z       'efficiency vs vertex_x (Z); vertex x; efficiency' resonanceZ_numerator_Probe_sctMuon_x       resonanceZ_denominator_Probe_sctMuon_x",
         "effic_vertex_y_Z       'efficiency vs vertex_y (Z); vertex y; efficiency' resonanceZ_numerator_Probe_sctMuon_y       resonanceZ_denominator_Probe_sctMuon_y",
         "effic_vertex_Zerror_Z       'efficiency vs Zerror (Z); vertex Zerror; efficiency' resonanceZ_numerator_Probe_sctMuon_Zerror       resonanceZ_denominator_Probe_sctMuon_Zerror",
         "effic_tracksSize_Z       'efficiency vs tracksSize (Z); tracksSize; efficiency' resonanceZ_numerator_Probe_sctMuon_tracksSize       resonanceZ_denominator_Probe_sctMuon_tracksSize",
    ),
)

#To declare muonEfficiencyVtx, clone muonEfficiencyNoVtx and change only the output subDir
muonEfficiencyVtx = muonEfficiencyNoVtx.clone()
muonEfficiencyVtx.subDirs = cms.untracked.vstring("/HLT/ScoutingOffline/Muons/Vtx")

#L1 seeds efficiency measurement
efficiencyList_DoubleMu = ["effic_pt1_%s       '%s; Leading muon pt [GeV]; L1 efficiency' h_pt1_numerator_%s h_pt1_denominator"%(seed,seed, seed) for seed in DoubleMuL1]+\
 ["effic_eta1_%s       '%s; Leading muon eta; L1 efficiency' h_eta1_numerator_%s h_eta1_denominator"%(seed,seed, seed) for seed in DoubleMuL1]+\
 ["effic_phi1_%s       '%s; Leading muon phi; L1 efficiency' h_phi1_numerator_%s h_phi1_denominator"%(seed,seed, seed) for seed in DoubleMuL1]+\
 ["effic_dxy1_%s       '%s; Leading muon dxy; L1 efficiency' h_dxy1_numerator_%s h_dxy1_denominator"%(seed,seed, seed) for seed in DoubleMuL1]+\
 ["effic_pt2_%s       '%s; Subleading muon pt [GeV]; L1 efficiency' h_pt2_numerator_%s h_pt2_denominator"%(seed,seed, seed) for seed in DoubleMuL1]+\
 ["effic_eta2_%s       '%s; Subleading muon eta; L1 efficiency' h_eta2_numerator_%s h_eta2_denominator"%(seed,seed, seed) for seed in DoubleMuL1]+\
 ["effic_phi2_%s       '%s; Subleading muon phi; L1 efficiency' h_phi2_numerator_%s h_phi2_denominator"%(seed,seed, seed) for seed in DoubleMuL1]+\
 ["effic_dxy2_%s       '%s; Subleading muon dxy; L1 efficiency' h_dxy2_numerator_%s h_dxy2_denominator"%(seed,seed, seed) for seed in DoubleMuL1]+\
 ["effic_invMass_%s       '%s; Invariant Mass; L1 efficiency' h_invMass_numerator_%s h_invMass_denominator"%(seed,seed, seed) for seed in DoubleMuL1]
 
efficiencyList_SingleMu = ["effic_pt1_%s       '%s; Leading muon pt [GeV]; L1 efficiency' h_pt1_numerator_%s h_pt1_denominator"%(seed,seed, seed) for seed in SingleMuL1]+\
 ["effic_eta1_%s       '%s; Leading muon eta; L1 efficiency' h_eta1_numerator_%s h_eta1_denominator"%(seed,seed, seed) for seed in SingleMuL1]+\
 ["effic_phi1_%s       '%s; Leading muon phi; L1 efficiency' h_phi1_numerator_%s h_phi1_denominator"%(seed,seed, seed) for seed in SingleMuL1]+\
 ["effic_dxy1_%s       '%s; Leading muon dxy; L1 efficiency' h_dxy1_numerator_%s h_dxy1_denominator"%(seed,seed, seed) for seed in SingleMuL1]+\
 ["effic_pt2_%s       '%s; Subleading muon pt [GeV]; L1 efficiency' h_pt2_numerator_%s h_pt2_denominator"%(seed,seed, seed) for seed in SingleMuL1]+\
 ["effic_eta2_%s       '%s; Subleading muon eta; L1 efficiency' h_eta2_numerator_%s h_eta2_denominator"%(seed,seed, seed) for seed in SingleMuL1]+\
 ["effic_phi2_%s       '%s; Subleading muon phi; L1 efficiency' h_phi2_numerator_%s h_phi2_denominator"%(seed,seed, seed) for seed in SingleMuL1]+\
 ["effic_dxy2_%s       '%s; Subleading muon dxy; L1 efficiency' h_dxy2_numerator_%s h_dxy2_denominator"%(seed,seed, seed) for seed in SingleMuL1]+\
 ["effic_invMass_%s       '%s; Invariant Mass; L1 efficiency' h_invMass_numerator_%s h_invMass_denominator"%(seed,seed, seed) for seed in SingleMuL1]
 
muonTriggerEfficiency_DoubleMu = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("/HLT/ScoutingOffline/Muons/L1Efficiency/DoubleMu"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages                                                                                                                                        
    resolution     = cms.vstring(),
    efficiency     = cms.vstring( efficiencyList_DoubleMu ),
)
muonTriggerEfficiency_SingleMu = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("/HLT/ScoutingOffline/Muons/L1Efficiency/SingleMu"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages                                                                                                                                          
    resolution     = cms.vstring(),
    efficiency     = cms.vstring( efficiencyList_SingleMu ),
)
