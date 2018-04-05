import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester
bphEfficiency = DQMEDHarvester("DQMGenericClient",
    subDirs        = cms.untracked.vstring("HLT/BPH/*"),
    verbose        = cms.untracked.uint32(0), # Set to 2 for all messages
    resolution     = cms.vstring(),
    efficiency     = cms.vstring(
        "effic_muPhi       'mu efficiency vs phi; mu phi [rad]; efficiency' muPhi_numerator       muPhi_denominator",
        "effic_muEta       'mu efficiency vs eta; mu eta [rad]; efficiency' muEta_numerator       muEta_denominator",
        "effic_muPt       'mu efficiency vs pt; mu pt [GeV]; efficiency' muPt_numerator       muPt_denominator",
        "effic_phPhi       'ph efficiency vs phi; ph phi [rad]; efficiency' phPhi_numerator       phPhi_denominator",
        "effic_phEta       'ph efficiency vs eta; ph eta [rad]; efficiency' phEta_numerator       phEta_denominator",
        "effic_phPt       'ph efficiency vs pt; ph pt [GeV]; efficiency' phPt_numerator       phPt_denominator",
        "effic_trPhi       'tr efficiency vs phi; tr phi [rad]; efficiency' trPhi_numerator       trPhi_denominator",
        "effic_trEta       'tr efficiency vs eta; tr eta [rad]; efficiency' trEta_numerator       trEta_denominator",
        "effic_trPt       'tr efficiency vs pt; tr pt [GeV]; efficiency' trPt_numerator       trPt_denominator",
        "effic_mu1Phi       'mu1 efficiency vs phi; mu1 phi [rad]; efficiency' mu1Phi_numerator       mu1Phi_denominator",
        "effic_mu1Eta       'mu1 efficiency vs eta; mu1 eta [rad]; efficiency' mu1Eta_numerator       mu1Eta_denominator",
        "effic_mu1Pt       'mu1 efficiency vs pt; mu1 pt [GeV]; efficiency' mu1Pt_numerator       mu1Pt_denominator",
        "effic_mu2Phi       'mu2 efficiency vs phi; mu2 phi [rad]; efficiency' mu2Phi_numerator       mu2Phi_denominator",
        "effic_mu2Eta       'mu2 efficiency vs eta; mu2 eta [rad]; efficiency' mu2Eta_numerator       mu2Eta_denominator",
        "effic_mu2Pt       'mu2 efficiency vs pt; mu2 pt [GeV]; efficiency' mu2Pt_numerator       mu2Pt_denominator",
        "effic_mu3Phi       'mu3 efficiency vs phi; mu3 phi [rad]; efficiency' mu3Phi_numerator       mu3Phi_denominator",
        "effic_mu3Eta       'mu3 efficiency vs eta; mu3 eta [rad]; efficiency' mu3Eta_numerator       mu3Eta_denominator",
        "effic_mu3Pt       'mu3 efficiency vs pt; mu3 pt [GeV]; efficiency' mu3Pt_numerator       mu3Pt_denominator",
        "effic_DiMuPhi       'DiMu efficiency vs phi; DiMu phi [rad]; efficiency' DiMuPhi_numerator       DiMuPhi_denominator",
        "effic_DiMuEta       'DiMu efficiency vs eta; DiMu eta [rad]; efficiency' DiMuEta_numerator       DiMuEta_denominator",
        "effic_DiMuPt       'DiMu efficiency vs pt; DiMu pt [GeV]; efficiency' DiMuPt_numerator       DiMuPt_denominator",
        "effic_DiMuPVcos       'DiMu efficiency vs cosPV; DiMu cosPV ; efficiency' DiMuPVcos_numerator       DiMuPVcos_denominator",
        "effic_DiMuProb       'DiMu efficiency vs prob; DiMu prob ; efficiency' DiMuProb_numerator       DiMuProb_denominator",
        "effic_DiMuDS       'DiMu efficiency vs DS; DiMu DS; efficiency' DiMuDS_numerator       DiMuDS_denominator",
        "effic_DiMuDCA       'DiMu efficiency vs DCA; DiMu DCA [cm]; efficiency' DiMuDCA_numerator       DiMuDCA_denominator",
        "effic_DiMuMass       'DiMu efficiency vs Mass; DiMu Mass[GeV]; efficiency' DiMuMass_numerator       DiMuMass_denominator",
        "effic_DiMudR       'DiMu efficiency vs dR; DiMu dR; efficiency' DiMudR_numerator       DiMudR_denominator",
        "effic_tr1Phi       'tr1 efficiency vs phi; tr1 phi [rad]; efficiency' tr1Phi_numerator       tr1Phi_denominator",
        "effic_tr1Eta       'tr1 efficiency vs eta; tr1 eta [rad]; efficiency' tr1Eta_numerator       tr1Eta_denominator",
        "effic_tr1Pt       'tr1 efficiency vs pt; tr1 pt [GeV]; efficiency' tr1Pt_numerator       tr1Pt_denominator",
        "effic_tr2Phi       'tr2 efficiency vs phi; tr2 phi [rad]; efficiency' tr2Phi_numerator       tr2Phi_denominator",
        "effic_tr2Eta       'tr2 efficiency vs eta; tr2 eta [rad]; efficiency' tr2Eta_numerator       tr2Eta_denominator",
        "effic_tr2Pt       'tr2 efficiency vs pt; tr2 pt [GeV]; efficiency' tr2Pt_numerator       tr2Pt_denominator",
        "effic_tr_d0       'tr efficiency vs d0; tr d0 [cm]; efficiency' tr_d0_numerator       tr_d0_denominator",
        "effic_tr_z0       'tr efficiency vs z0; tr z0 [cm]; efficiency' tr_z0_numerator       tr_z0_denominator",


    ),
#    efficiencyProfile = cms.untracked.vstring(
#        "effic_met_vs_LS 'MET efficiency vs LS; LS; PF MET efficiency' metVsLS_numerator metVsLS_denominator"
#    ),
  
)

bphClient = cms.Sequence(
    bphEfficiency
)

##

##
