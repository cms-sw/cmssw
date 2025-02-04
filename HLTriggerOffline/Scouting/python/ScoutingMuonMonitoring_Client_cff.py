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
        "effic_muonPt       'efficiency vs pt; Muon pt [GeV]; efficiency' resonanceJ_numerator_Probe_sctMuon_Pt       resonanceJ_denominator_Probe_sctMuon_Pt",
        "effic_muonEta       'efficiency vs eta; Muon eta; efficiency' resonanceJ_numerator_Probe_sctMuon_Eta       resonanceJ_denominator_Probe_sctMuon_Eta",
        "effic_muonPhi       'efficiency vs phi; Muon phi; efficiency' resonanceJ_numerator_Probe_sctMuon_Phi       resonanceJ_denominator_Probe_sctMuon_Phi",
        "effic_muondxy       'efficiency vs dxy; Muon dxy; efficiency' resonanceJ_numerator_Probe_sctMuon_dxy       resonanceJ_denominator_Probe_sctMuon_dxy",
        "effic_muonInvMass       'efficiency vs inv mass; Muon inv mass [GeV]; efficiency' resonanceJ_numerator_sctMuon_Invariant_Mass       resonanceJ_denominator_sctMuon_Invariant_Mass",
        "effic_muonNormChisq      'efficiency vs normChi2; Muon normChi2; efficiency' resonanceJ_numerator_Probe_sctMuon_NormChisq       resonanceJ_denominator_Probe_sctMuon_NormChisq",
        "effic_muonTrkdz       'efficiency vs Trkdz; Muon trkdz; efficiency' resonanceJ_numerator_Probe_sctMuon_Trk_dz       resonanceJ_denominator_Probe_sctMuon_Trkdz",
        "effic_muonTrkdxy       'efficiency vs Trkdxy; Muon trkdxy; efficiency' resonanceJ_numerator_Probe_sctMuon_Trk_dxy       resonanceJ_denominator_Probe_sctMuon_Trkdxy",
        "effic_muonlxy       'efficiency vs lxy; Muon lxy; efficiency' resonanceJ_numerator_Vertex_Lxy       resonanceJ_denominator_Vertex_Lxy",
        "effic_muonVertexYerror       'efficiency vs VertexYerror; Muon Vertex Yerror; efficiency' resonanceJ_numerator_Vertex_Yerror       resonanceJ_denominator_Vertex_Yerror",
        "effic_muonVertexXerror       'efficiency vs VertexXerror; Muon Vertex Xerror; efficiency' resonanceJ_numerator_Vertex_Xerror       resonanceJ_denominator_Vertex_Xerror",
        "effic_muonVertexChi2       'efficiency vs Vertexchi2; Muon Vertex chi2; efficiency' resonanceJ_numerator_Vertex_chi2       resonanceJ_denominator_Vertex_chi2",
        "effic_muonVertexYerror       'efficiency vs z; Muon Vertex z; efficiency' resonanceJ_numerator_Vertex_z       resonanceJ_denominator_Vertex_z",
        "effic_muontype      'efficiency vs type; Muon type; efficiency' resonanceJ_numerator_Probe_sctMuon_type      resonanceJ_denominator_Probe_sctMuon_type",
        "effic_muoncharge       'efficiency vs charge; Muon charge; efficiency' resonanceJ_numerator_Probe_sctMuon_charge       resonanceJ_denominator_Probe_sctMuon_charge",
        "effic_muonecalIso       'efficiency vs ecalIso; Muon ecalIso; efficiency' resonanceJ_numerator_Probe_sctMuon_ecalIso       resonanceJ_denominator_Probe_sctMuon_ecalIso",
        "effic_muonhcalIso       'efficiency vs hcalIso; Muon hcalIso; efficiency' resonanceJ_numerator_Probe_sctMuon_hcalIso       resonanceJ_denominator_Probe_sctMuon_hcalIso",
        "effic_muontrackIso       'efficiency vs trackIso; Muon trackIso; efficiency' resonanceJ_numerator_Probe_sctMuon_trackIso       resonanceJ_denominator_Probe_sctMuon_trackIso",
        "effic_nValidStandAloneMuonHits       'efficiency vs nValidStandAloneMuonHits; nValidStandAloneMuonHits; efficiency' resonanceJ_numerator_Probe_sctMuon_nValidStandAloneMuonHits       resonanceJ_denominator_Probe_sctMuon_nValidStandAloneMuonHits",
        "effic_nStandAloneMuonMatchedStations       'efficiency vs nStandAloneMuonMatchedStations; nStandAloneMuonMatchedStations; efficiency' resonanceJ_numerator_Probe_sctMuon_nStandAloneMuonMatchedStations       resonanceJ_denominator_Probe_sctMuon_nStandAloneMuonMatchedStations",
        "effic_nValidRecoMuonHits       'efficiency vs nValidRecoMuonHits; nValidRecoMuonHits; efficiency' resonanceJ_numerator_Probe_sctMuon_nValidRecoMuonHits       resonanceJ_denominator_Probe_sctMuon_nValidRecoMuonHits",
        "effic_nRecoMuonChambers       'efficiency vs nRecoMuonChambers; nRecoMuonChambers; efficiency' resonanceJ_numerator_Probe_sctMuon_nRecoMuonChambers       resonanceJ_denominator_Probe_sctMuon_nRecoMuonChambers",
        "effic_nRecoMuonChambersCSCorDT       'efficiency vs nRecoMuonChambersCSCorDT; nRecoMuonChambersCSCorDT; efficiency' resonanceJ_numerator_Probe_sctMuon_nRecoMuonChambersCSCorDT       resonanceJ_denominator_Probe_sctMuon_nRecoMuonChambersCSCorDT",
        "effic_nRecoMuonMatches       'efficiency vs nRecoMuonMatches; nRecoMuonMatches; efficiency' resonanceJ_numerator_Probe_sctMuon_nRecoMuonMatches       resonanceJ_denominator_Probe_sctMuon_nRecoMuonMatches",
        "effic_nRecoMuonMatchedStations       'efficiency vs nRecoMuonMatchedStations; nRecoMuonMatchedStations; efficiency' resonanceJ_numerator_Probe_sctMuon_nRecoMuonMatchedStations       resonanceJ_denominator_Probe_sctMuon_nRecoMuonMatchedStations",
        "effic_nRecoMuonExpectedMatchedStations      'efficiency vs nRecoMuonExpectedMatchedStations; nRecoMuonExpectedMatchedStations; efficiency' resonanceJ_numerator_Probe_sctMuon_nRecoMuonExpectedMatchedStations       resonanceJ_denominator_Probe_sctMuon_nRecoMuonExpectedMatchedStations",
        "effic_nRecoMuonMatchedRPCLayers       'efficiency vs nRecoMuonMatchedRPCLayers; nRecoMuonMatchedRPCLayers; efficiency' resonanceJ_numerator_Probe_sctMuon_nRecoMuonMatchedRPCLayers       resonanceJ_denominator_Probe_sctMuon_nRecoMuonMatchedRPCLayers",
        "effic_recoMuonRPClayerMask      'efficiency vs recoMuonRPClayerMask; recoMuonRPClayerMask; efficiency' resonanceJ_numerator_Probe_sctMuon_recoMuonRPClayerMask       resonanceJ_denominator_Probe_sctMuon_recoMuonRPClayerMask",
        "effic_nValidPixelHits       'efficiency vs nValidPixelHits; nValidPixelHits; efficiency' resonanceJ_numerator_Probe_sctMuon_nValidPixelHits       resonanceJ_denominator_Probe_sctMuon_nValidPixelHits",
        "effic_nValidStripHits       'efficiency vs nValidStripHits; nValidStripHits; efficiency' resonanceJ_numerator_Probe_sctMuon_nValidStripHits       resonanceJ_denominator_Probe_sctMuon_nValidStripHits",
        "effic_nPixelLayersWithMeasurement       'efficiency vs nPixelLayersWithMeasurement; nPixelLayersWithMeasurement; efficiency' resonanceJ_numerator_Probe_sctMuon_nPixelLayersWithMeasurement       resonanceJ_denominator_Probe_sctMuon_nPixelLayersWithMeasurement",
        "effic_nTrackerLayersWithMeasurement       'efficiency vs nTrackerLayersWithMeasurement; nTrackerLayersWithMeasurement; efficiency' resonanceJ_numerator_Probe_sctMuon_nTrackerLayersWithMeasurement       resonanceJ_denominator_Probe_sctMuon_nTrackerLayersWithMeasurement",
        "effic_trk_chi2       'efficiency vs trk_chi2; trk_chi2; efficiency' resonanceJ_numerator_Probe_sctMuon_trk_chi2      resonanceJ_denominator_Probe_sctMuon_trk_chi2",
        "effic_trk_ndof       'efficiency vs trk_ndof; trk_ndof; efficiency' resonanceJ_numerator_Probe_sctMuon_trk_ndof      resonanceJ_denominator_Probe_sctMuon_trk_ndof",
        "effic_trk_lambda       'efficiency vs trk_lambda; trk_lambda; efficiency' resonanceJ_numerator_Probe_sctMuon_trk_lambda       resonanceJ_denominator_Probe_sctMuon_trk_lambda",
        "effic_trk_pt       'efficiency vs trk_pt; trk_pt; efficiency' resonanceJ_numerator_Probe_sctMuon_trk_pt       resonanceJ_denominator_Probe_sctMuon_trk_pt",
        "effic_trk_eta      'efficiency vs trk_eta; trk_eta; efficiency' resonanceJ_numerator_Probe_sctMuon_trk_eta    resonanceJ_denominator_Probe_sctMuon_trk_eta",
        "effic_trk_dxyError       'efficiency vs trk_dxyError; trk_dxyError; efficiency' resonanceJ_numerator_Probe_sctMuon_trk_dxyError       resonanceJ_denominator_Probe_sctMuon_trk_dxyError",
        "effic_trk_dzError       'efficiency vs trk_dzError; trk_dzError; efficiency' resonanceJ_numerator_Probe_sctMuon_trk_dzError       resonanceJ_denominator_Probe_sctMuon_trk_dzError",
        "effic_trk_qoverpError       'efficiency vs trk_qoverpError; trk_qoverpError; efficiency' resonanceJ_numerator_Probe_sctMuon_trk_qoverpError       resonanceJ_denominator_Probe_sctMuon_trk_qoverpError",
        "effic_trk_lambdaError      'efficiency vs trk_lambdaError; trk_lambdaError; efficiency' resonanceJ_numerator_Probe_sctMuon_trk_lambdaError       resonanceJ_denominator_Probe_sctMuon_trk_lambdaError",
        "effic_trk_phiError       'efficiency vs trk_phiError; trk_phiError; efficiency' resonanceJ_numerator_Probe_sctMuon_trk_phiError       resonanceJ_denominator_Probe_sctMuon_trk_phiError",
        "effic_trk_dsz      'efficiency vs trk_dsz; trk_dsz; efficiency' resonanceJ_numerator_Probe_sctMuon_trk_dsz       resonanceJ_denominator_Probe_sctMuon_trk_dsz",
        "effic_trk_dszError       'efficiency vs trk_dszError; trk_dszError; efficiency' resonanceJ_numerator_Probe_sctMuon_trk_dszError       resonanceJ_denominator_Probe_sctMuon_trk_dszError",
        "effic_ndof       'efficiency vs ndof; ndof; efficiency' resonanceJ_numerator_Probe_sctMuon_ndof      resonanceJ_denominator_Probe_sctMuon_ndof",
        "effic_trk_vx       'efficiency vs trk_vx; trk_vx; efficiency' resonanceJ_numerator_Probe_sctMuon_trk_vx       resonanceJ_denominator_Probe_sctMuon_trk_vx",
        "effic_trk_vy       'efficiency vs trk_vy; trk_vy; efficiency' resonanceJ_numerator_Probe_sctMuon_trk_vy       resonanceJ_denominator_Probe_sctMuon_trk_vy",
        "effic_trk_vz       'efficiency vs trk_vz; trk_vz; efficiency' resonanceJ_numerator_Probe_sctMuon_trk_vz      resonanceJ_denominator_Probe_sctMuon_trk_vz",
        "effic_vertex_x       'efficiency vs vertex_x; vertex x; efficiency' resonanceJ_numerator_Probe_sctMuon_x       resonanceJ_denominator_Probe_sctMuon_x",
        "effic_vertex_y       'efficiency vs vertex_y; vertex y; efficiency' resonanceJ_numerator_Probe_sctMuon_y       resonanceJ_denominator_Probe_sctMuon_y",
        "effic_vertex_Zerror       'efficiency vs Zerror; vertex Zerror; efficiency' resonanceJ_numerator_Probe_sctMuon_Zerror       resonanceJ_denominator_Probe_sctMuon_Zerror",
        "effic_tracksSize       'efficiency vs tracksSize; tracksSize; efficiency' resonanceJ_numerator_Probe_sctMuon_tracksSize       resonanceJ_denominator_Probe_sctMuon_tracksSize",
    ),
)

#To declare muonEfficiencyVtx, clone muonEfficiencyNoVtx and change only the output subDir
muonEfficiencyVtx = muonEfficiencyNoVtx.clone()
muonEfficiencyVtx.subDirs = cms.untracked.vstring("/HLT/ScoutingOffline/Muons/Vtx")

#L1 seeds efficiency measurement
allSeeds = SingleMuL1 + DoubleMuL1
efficiencyList = ["effic_pt1_%s       '%s; Leading muon pt [GeV]; L1 efficiency' h_pt1_numerator_%s h_pt1_denominator"%(seed,seed, seed) for seed in allSeeds]+\
["effic_eta1_%s       '%s; Leading muon eta; L1 efficiency' h_eta1_numerator_%s h_eta1_denominator"%(seed,seed, seed) for seed in allSeeds]+\
["effic_phi1_%s       '%s; Leading muon phi; L1 efficiency' h_phi1_numerator_%s h_phi1_denominator"%(seed,seed, seed) for seed in allSeeds]+\
["effic_dxy1_%s       '%s; Leading muon dxy; L1 efficiency' h_dxy1_numerator_%s h_dxy1_denominator"%(seed,seed, seed) for seed in allSeeds]+\
["effic_pt2_%s       '%s; Subleading muon pt [GeV]; L1 efficiency' h_pt2_numerator_%s h_pt2_denominator"%(seed,seed, seed) for seed in allSeeds]+\
["effic_eta2_%s       '%s; Subleading muon eta; L1 efficiency' h_eta2_numerator_%s h_eta2_denominator"%(seed,seed, seed) for seed in allSeeds]+\
["effic_phi2_%s       '%s; Subleading muon phi; L1 efficiency' h_phi2_numerator_%s h_phi2_denominator"%(seed,seed, seed) for seed in allSeeds]+\
["effic_dxy2_%s       '%s; Subleading muon dxy; L1 efficiency' h_dxy2_numerator_%s h_dxy2_denominator"%(seed,seed, seed) for seed in allSeeds]+\
["effic_invMass_%s       '%s; Invariant Mass; L1 efficiency' h_invMass_numerator_%s h_invMass_denominator"%(seed,seed, seed) for seed in allSeeds]

muonTriggerEfficiency = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("/HLT/ScoutingOffline/Muons/L1Efficiency"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages                                                                                                                                          
    resolution     = cms.vstring(),
    efficiency     = cms.vstring( efficiencyList ),
)
