// -*- C++ -*-
//
// Package:    RecoEgamma/Examples
// Class:      GsfElectronMCAnalyzer
//
/**\class GsfElectronMCAnalyzer RecoEgamma/Examples/src/GsfElectronMCAnalyzer.cc

 Description: GsfElectrons analyzer using MC truth

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Ursula Berthon, Claude Charlot
//         Created:  Mon Mar 27 13:22:06 CEST 2006
//
//

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

#include "CLHEP/Units/GlobalPhysicalConstants.h"

#include "TMath.h"
#include "TFile.h"
#include "TH1F.h"
#include "TH1I.h"
#include "TH2F.h"
#include "TProfile.h"
#include "TTree.h"

#include <iostream>
#include <vector>

class GsfElectronMCAnalyzer : public edm::one::EDAnalyzer<> {
public:
  explicit GsfElectronMCAnalyzer(const edm::ParameterSet &conf);

  ~GsfElectronMCAnalyzer() override;

  void beginJob() override;
  void endJob() override;
  void analyze(const edm::Event &e, const edm::EventSetup &c) override;

private:
  TrajectoryStateTransform transformer_;
  edm::ESHandle<TrackerGeometry> pDD;
  edm::ESHandle<MagneticField> theMagField;
  TFile *histfile_;
  TTree *tree_;
  float mcEnergy[10], mcEta[10], mcPhi[10], mcPt[10], mcQ[10];
  float superclusterEnergy[10], superclusterEta[10], superclusterPhi[10], superclusterEt[10];
  float seedMomentum[10], seedEta[10], seedPhi[10], seedPt[10], seedQ[10];

  TH1F *h_mcNum;
  TH1F *h_eleNum;
  TH1F *h_gamNum;

  TH1F *h_simEta;
  TH1F *h_simAbsEta;
  TH1F *h_simP;
  TH1F *h_simPt;
  TH1F *h_simPhi;
  TH1F *h_simZ;
  TH2F *h_simPtEta;

  TH1F *h_ele_simEta_matched;
  TH1F *h_ele_simAbsEta_matched;
  TH1F *h_ele_simPt_matched;
  TH1F *h_ele_simPhi_matched;
  TH1F *h_ele_simZ_matched;
  TH2F *h_ele_simPtEta_matched;

  TH1F *h_ele_simEta_matched_qmisid;
  TH1F *h_ele_simAbsEta_matched_qmisid;
  TH1F *h_ele_simPt_matched_qmisid;
  TH1F *h_ele_simPhi_matched_qmisid;
  TH1F *h_ele_simZ_matched_qmisid;

  TH1F *h_ele_EoverP_all;
  TH1F *h_ele_EoverP_all_barrel;
  TH1F *h_ele_EoverP_all_endcaps;
  TH1F *h_ele_EseedOP_all;
  TH1F *h_ele_EseedOP_all_barrel;
  TH1F *h_ele_EseedOP_all_endcaps;
  TH1F *h_ele_EoPout_all;
  TH1F *h_ele_EoPout_all_barrel;
  TH1F *h_ele_EoPout_all_endcaps;
  TH1F *h_ele_EeleOPout_all;
  TH1F *h_ele_EeleOPout_all_barrel;
  TH1F *h_ele_EeleOPout_all_endcaps;
  TH1F *h_ele_dEtaSc_propVtx_all;
  TH1F *h_ele_dEtaSc_propVtx_all_barrel;
  TH1F *h_ele_dEtaSc_propVtx_all_endcaps;
  TH1F *h_ele_dPhiSc_propVtx_all;
  TH1F *h_ele_dPhiSc_propVtx_all_barrel;
  TH1F *h_ele_dPhiSc_propVtx_all_endcaps;
  TH1F *h_ele_dEtaCl_propOut_all;
  TH1F *h_ele_dEtaCl_propOut_all_barrel;
  TH1F *h_ele_dEtaCl_propOut_all_endcaps;
  TH1F *h_ele_dPhiCl_propOut_all;
  TH1F *h_ele_dPhiCl_propOut_all_barrel;
  TH1F *h_ele_dPhiCl_propOut_all_endcaps;
  TH1F *h_ele_TIP_all;
  TH1F *h_ele_TIP_all_barrel;
  TH1F *h_ele_TIP_all_endcaps;
  TH1F *h_ele_HoE_all;
  TH1F *h_ele_HoE_all_barrel;
  TH1F *h_ele_HoE_all_endcaps;
  TH1F *h_ele_vertexEta_all;
  TH1F *h_ele_vertexPt_all;
  TH1F *h_ele_Et_all;
  TH1F *h_ele_mee_all;
  TH1F *h_ele_mee_os;
  TH1F *h_ele_mee_os_ebeb;
  TH1F *h_ele_mee_os_ebee;
  TH1F *h_ele_mee_os_eeee;
  TH1F *h_ele_mee_os_gg;
  TH1F *h_ele_mee_os_gb;
  TH1F *h_ele_mee_os_bb;

  TH2F *h_ele_E2mnE1vsMee_all;
  TH2F *h_ele_E2mnE1vsMee_egeg_all;

  TH1F *h_ele_charge;
  TH2F *h_ele_chargeVsEta;
  TH2F *h_ele_chargeVsPhi;
  TH2F *h_ele_chargeVsPt;
  TH1F *h_ele_vertexP;
  TH1F *h_ele_vertexPt;
  TH1F *h_ele_Et;
  TH2F *h_ele_vertexPtVsEta;
  TH2F *h_ele_vertexPtVsPhi;
  TH1F *h_ele_vertexPt_5100;
  TH1F *h_ele_vertexEta;
  TH2F *h_ele_vertexEtaVsPhi;
  TH1F *h_ele_vertexAbsEta;
  TH1F *h_ele_vertexPhi;
  TH1F *h_ele_vertexX;
  TH1F *h_ele_vertexY;
  TH1F *h_ele_vertexZ;
  TH1F *h_ele_vertexTIP;
  TH2F *h_ele_vertexTIPVsEta;
  TH2F *h_ele_vertexTIPVsPhi;
  TH2F *h_ele_vertexTIPVsPt;

  TH1F *histNum_;

  TH1F *histSclEn_;
  TH1F *histSclEoEtrue_barrel;
  TH1F *histSclEoEtrue_endcaps;
  TH1F *histSclEoEtrue_barrel_eg;
  TH1F *histSclEoEtrue_endcaps_eg;
  TH1F *histSclEoEtrue_barrel_etagap;
  TH1F *histSclEoEtrue_barrel_phigap;
  TH1F *histSclEoEtrue_ebeegap;
  TH1F *histSclEoEtrue_endcaps_deegap;
  TH1F *histSclEoEtrue_endcaps_ringgap;
  TH1F *histSclEoEtrue_barrel_new;
  TH1F *histSclEoEtrue_endcaps_new;
  TH1F *histSclEoEtrue_barrel_eg_new;
  TH1F *histSclEoEtrue_endcaps_eg_new;
  TH1F *histSclEoEtrue_barrel_etagap_new;
  TH1F *histSclEoEtrue_barrel_phigap_new;
  TH1F *histSclEoEtrue_ebeegap_new;
  TH1F *histSclEoEtrue_endcaps_deegap_new;
  TH1F *histSclEoEtrue_endcaps_ringgap_new;
  TH1F *histSclEt_;
  TH2F *histSclEtVsEta_;
  TH2F *histSclEtVsPhi_;
  TH2F *histSclEtaVsPhi_;
  TH1F *histSclEta_;
  TH1F *histSclPhi_;

  TH2F *histSclEoEtruePfVsEg;

  TH1F *histSclSigEtaEta_;
  TH1F *histSclSigEtaEta_barrel_;
  TH1F *histSclSigEtaEta_endcaps_;
  TH1F *histSclSigIEtaIEta_;
  TH1F *histSclSigIEtaIEta_barrel_;
  TH1F *histSclSigIEtaIEta_endcaps_;
  TH1F *histSclE1x5_;
  TH1F *histSclE1x5_barrel_;
  TH1F *histSclE1x5_endcaps_;
  TH1F *histSclE2x5max_;
  TH1F *histSclE2x5max_barrel_;
  TH1F *histSclE2x5max_endcaps_;
  TH1F *histSclE5x5_;
  TH1F *histSclE5x5_barrel_;
  TH1F *histSclE5x5_endcaps_;
  TH1F *histSclSigEtaEta_eg_;
  TH1F *histSclSigEtaEta_eg_barrel_;
  TH1F *histSclSigEtaEta_eg_endcaps_;
  TH1F *histSclSigIEtaIEta_eg_;
  TH1F *histSclSigIEtaIEta_eg_barrel_;
  TH1F *histSclSigIEtaIEta_eg_endcaps_;
  TH1F *histSclE1x5_eg_;
  TH1F *histSclE1x5_eg_barrel_;
  TH1F *histSclE1x5_eg_endcaps_;
  TH1F *histSclE2x5max_eg_;
  TH1F *histSclE2x5max_eg_barrel_;
  TH1F *histSclE2x5max_eg_endcaps_;
  TH1F *histSclE5x5_eg_;
  TH1F *histSclE5x5_eg_barrel_;
  TH1F *histSclE5x5_eg_endcaps_;

  TH1F *h_ele_ambiguousTracks;
  TH2F *h_ele_ambiguousTracksVsEta;
  TH2F *h_ele_ambiguousTracksVsPhi;
  TH2F *h_ele_ambiguousTracksVsPt;
  TH1F *h_ele_foundHits;
  TH1F *h_ele_foundHits_barrel;
  TH1F *h_ele_foundHits_endcaps;
  TH2F *h_ele_foundHitsVsEta;
  TH2F *h_ele_foundHitsVsPhi;
  TH2F *h_ele_foundHitsVsPt;
  TH1F *h_ele_lostHits;
  TH1F *h_ele_lostHits_barrel;
  TH1F *h_ele_lostHits_endcaps;
  TH2F *h_ele_lostHitsVsEta;
  TH2F *h_ele_lostHitsVsPhi;
  TH2F *h_ele_lostHitsVsPt;
  TH1F *h_ele_chi2;
  TH1F *h_ele_chi2_barrel;
  TH1F *h_ele_chi2_endcaps;
  TH2F *h_ele_chi2VsEta;
  TH2F *h_ele_chi2VsPhi;
  TH2F *h_ele_chi2VsPt;

  TH1F *h_ele_PoPtrue;
  TH1F *h_ele_PtoPttrue;
  TH2F *h_ele_PoPtrueVsEta;
  TH2F *h_ele_PoPtrueVsPhi;
  TH2F *h_ele_PoPtrueVsPt;
  TH1F *h_ele_PoPtrue_barrel;
  TH1F *h_ele_PoPtrue_endcaps;
  TH1F *h_ele_PoPtrue_golden_barrel;
  TH1F *h_ele_PoPtrue_golden_endcaps;
  TH1F *h_ele_PoPtrue_showering_barrel;
  TH1F *h_ele_PoPtrue_showering_endcaps;
  TH1F *h_ele_PtoPttrue_barrel;
  TH1F *h_ele_PtoPttrue_endcaps;
  TH1F *h_ele_ChargeMnChargeTrue;
  TH1F *h_ele_EtaMnEtaTrue;
  TH1F *h_ele_EtaMnEtaTrue_barrel;
  TH1F *h_ele_EtaMnEtaTrue_endcaps;
  TH2F *h_ele_EtaMnEtaTrueVsEta;
  TH2F *h_ele_EtaMnEtaTrueVsPhi;
  TH2F *h_ele_EtaMnEtaTrueVsPt;
  TH1F *h_ele_PhiMnPhiTrue;
  TH1F *h_ele_PhiMnPhiTrue_barrel;
  TH1F *h_ele_PhiMnPhiTrue_endcaps;
  TH1F *h_ele_PhiMnPhiTrue2;
  TH2F *h_ele_PhiMnPhiTrueVsEta;
  TH2F *h_ele_PhiMnPhiTrueVsPhi;
  TH2F *h_ele_PhiMnPhiTrueVsPt;
  TH1F *h_ele_PinMnPout;
  TH1F *h_ele_PinMnPout_mode;
  TH2F *h_ele_PinMnPoutVsEta_mode;
  TH2F *h_ele_PinMnPoutVsPhi_mode;
  TH2F *h_ele_PinMnPoutVsPt_mode;
  TH2F *h_ele_PinMnPoutVsE_mode;
  TH2F *h_ele_PinMnPoutVsChi2_mode;

  TH1F *h_ele_outerP;
  TH1F *h_ele_outerP_mode;
  TH2F *h_ele_outerPVsEta_mode;
  TH1F *h_ele_outerPt;
  TH1F *h_ele_outerPt_mode;
  TH2F *h_ele_outerPtVsEta_mode;
  TH2F *h_ele_outerPtVsPhi_mode;
  TH2F *h_ele_outerPtVsPt_mode;
  TH1F *h_ele_EoP;
  TH1F *h_ele_EoP_barrel;
  TH1F *h_ele_EoP_endcaps;
  TH1F *h_ele_EoP_eg;
  TH1F *h_ele_EoP_eg_barrel;
  TH1F *h_ele_EoP_eg_endcaps;
  TH2F *h_ele_EoPVsEta;
  TH2F *h_ele_EoPVsPhi;
  TH2F *h_ele_EoPVsE;
  TH1F *h_ele_EseedOP;
  TH1F *h_ele_EseedOP_barrel;
  TH1F *h_ele_EseedOP_endcaps;
  TH1F *h_ele_EseedOP_eg;
  TH1F *h_ele_EseedOP_eg_barrel;
  TH1F *h_ele_EseedOP_eg_endcaps;
  TH2F *h_ele_EseedOPVsEta;
  TH2F *h_ele_EseedOPVsPhi;
  TH2F *h_ele_EseedOPVsE;
  TH1F *h_ele_EoPout;
  TH1F *h_ele_EoPout_barrel;
  TH1F *h_ele_EoPout_endcaps;
  TH1F *h_ele_EoPout_eg;
  TH1F *h_ele_EoPout_eg_barrel;
  TH1F *h_ele_EoPout_eg_endcaps;
  TH2F *h_ele_EoPoutVsEta;
  TH2F *h_ele_EoPoutVsPhi;
  TH2F *h_ele_EoPoutVsE;
  TH1F *h_ele_EeleOPout;
  TH1F *h_ele_EeleOPout_barrel;
  TH1F *h_ele_EeleOPout_endcaps;
  TH1F *h_ele_EeleOPout_eg;
  TH1F *h_ele_EeleOPout_eg_barrel;
  TH1F *h_ele_EeleOPout_eg_endcaps;
  TH2F *h_ele_EeleOPoutVsEta;
  TH2F *h_ele_EeleOPoutVsPhi;
  TH2F *h_ele_EeleOPoutVsE;

  TH1F *h_ele_dEtaSc_propVtx;
  TH1F *h_ele_dEtaSc_propVtx_barrel;
  TH1F *h_ele_dEtaSc_propVtx_endcaps;
  TH1F *h_ele_dEtaSc_propVtx_eg;
  TH1F *h_ele_dEtaSc_propVtx_eg_barrel;
  TH1F *h_ele_dEtaSc_propVtx_eg_endcaps;
  TH2F *h_ele_dEtaScVsEta_propVtx;
  TH2F *h_ele_dEtaScVsPhi_propVtx;
  TH2F *h_ele_dEtaScVsPt_propVtx;
  TH1F *h_ele_dPhiSc_propVtx;
  TH1F *h_ele_dPhiSc_propVtx_barrel;
  TH1F *h_ele_dPhiSc_propVtx_endcaps;
  TH1F *h_ele_dPhiSc_propVtx_eg;
  TH1F *h_ele_dPhiSc_propVtx_eg_barrel;
  TH1F *h_ele_dPhiSc_propVtx_eg_endcaps;
  TH2F *h_ele_dPhiScVsEta_propVtx;
  TH2F *h_ele_dPhiScVsPhi_propVtx;
  TH2F *h_ele_dPhiScVsPt_propVtx;
  TH1F *h_ele_dEtaCl_propOut;
  TH1F *h_ele_dEtaCl_propOut_barrel;
  TH1F *h_ele_dEtaCl_propOut_endcaps;
  TH1F *h_ele_dEtaCl_propOut_eg;
  TH1F *h_ele_dEtaCl_propOut_eg_barrel;
  TH1F *h_ele_dEtaCl_propOut_eg_endcaps;
  TH2F *h_ele_dEtaClVsEta_propOut;
  TH2F *h_ele_dEtaClVsPhi_propOut;
  TH2F *h_ele_dEtaClVsPt_propOut;
  TH1F *h_ele_dPhiCl_propOut;
  TH1F *h_ele_dPhiCl_propOut_barrel;
  TH1F *h_ele_dPhiCl_propOut_endcaps;
  TH1F *h_ele_dPhiCl_propOut_eg;
  TH1F *h_ele_dPhiCl_propOut_eg_barrel;
  TH1F *h_ele_dPhiCl_propOut_eg_endcaps;
  TH2F *h_ele_dPhiClVsEta_propOut;
  TH2F *h_ele_dPhiClVsPhi_propOut;
  TH2F *h_ele_dPhiClVsPt_propOut;
  TH1F *h_ele_dEtaEleCl_propOut;
  TH1F *h_ele_dEtaEleCl_propOut_barrel;
  TH1F *h_ele_dEtaEleCl_propOut_endcaps;
  TH1F *h_ele_dEtaEleCl_propOut_eg;
  TH1F *h_ele_dEtaEleCl_propOut_eg_barrel;
  TH1F *h_ele_dEtaEleCl_propOut_eg_endcaps;
  TH2F *h_ele_dEtaEleClVsEta_propOut;
  TH2F *h_ele_dEtaEleClVsPhi_propOut;
  TH2F *h_ele_dEtaEleClVsPt_propOut;
  TH1F *h_ele_dPhiEleCl_propOut;
  TH1F *h_ele_dPhiEleCl_propOut_barrel;
  TH1F *h_ele_dPhiEleCl_propOut_endcaps;
  TH1F *h_ele_dPhiEleCl_propOut_eg;
  TH1F *h_ele_dPhiEleCl_propOut_eg_barrel;
  TH1F *h_ele_dPhiEleCl_propOut_eg_endcaps;
  TH2F *h_ele_dPhiEleClVsEta_propOut;
  TH2F *h_ele_dPhiEleClVsPhi_propOut;
  TH2F *h_ele_dPhiEleClVsPt_propOut;

  TH1F *h_ele_seed_dphi2_;
  TH2F *h_ele_seed_dphi2VsEta_;
  TH2F *h_ele_seed_dphi2VsPt_;
  TH1F *h_ele_seed_drz2_;
  TH2F *h_ele_seed_drz2VsEta_;
  TH2F *h_ele_seed_drz2VsPt_;
  TH1F *h_ele_seed_subdet2_;

  TH1F *h_ele_classes;
  TH1F *h_ele_eta;
  TH1F *h_ele_eta_golden;
  TH1F *h_ele_eta_bbrem;
  TH1F *h_ele_eta_narrow;
  TH1F *h_ele_eta_shower;

  TH1F *h_ele_HoE;
  TH1F *h_ele_HoE_barrel;
  TH1F *h_ele_HoE_endcaps;
  TH1F *h_ele_HoE_eg;
  TH1F *h_ele_HoE_eg_barrel;
  TH1F *h_ele_HoE_eg_endcaps;
  TH1F *h_ele_HoE_fiducial;
  TH2F *h_ele_HoEVsEta;
  TH2F *h_ele_HoEVsPhi;
  TH2F *h_ele_HoEVsE;

  TH1F *h_ele_fbrem;
  TH1F *h_ele_fbrem_eg;
  TProfile *h_ele_fbremVsEta_mode;
  TProfile *h_ele_fbremVsEta_mean;

  TH2F *h_ele_PinVsPoutGolden_mode;
  TH2F *h_ele_PinVsPoutShowering_mode;
  TH2F *h_ele_PinVsPoutGolden_mean;
  TH2F *h_ele_PinVsPoutShowering_mean;
  TH2F *h_ele_PtinVsPtoutGolden_mode;
  TH2F *h_ele_PtinVsPtoutShowering_mode;
  TH2F *h_ele_PtinVsPtoutGolden_mean;
  TH2F *h_ele_PtinVsPtoutShowering_mean;
  TH1F *histSclEoEtrueGolden_barrel;
  TH1F *histSclEoEtrueGolden_endcaps;
  TH1F *histSclEoEtrueShowering_barrel;
  TH1F *histSclEoEtrueShowering_endcaps;

  TH1F *h_ele_mva;
  TH1F *h_ele_mva_eg;
  TH1F *h_ele_provenance;

  TH1F *h_ele_tkSumPt_dr03;
  TH1F *h_ele_ecalRecHitSumEt_dr03;
  TH1F *h_ele_hcalDepth1TowerSumEt_dr03;
  TH1F *h_ele_hcalDepth2TowerSumEt_dr03;
  TH1F *h_ele_tkSumPt_dr04;
  TH1F *h_ele_ecalRecHitSumEt_dr04;
  TH1F *h_ele_hcalDepth1TowerSumEt_dr04;
  TH1F *h_ele_hcalDepth2TowerSumEt_dr04;

  std::string outputFile_;
  edm::InputTag electronCollection_;
  edm::InputTag mcTruthCollection_;
  bool readAOD_;

  double maxPt_;
  double maxAbsEta_;
  double deltaR_;
  std::vector<int> matchingIDs_;
  std::vector<int> matchingMotherIDs_;

  // histos limits and binning
  double etamin;
  double etamax;
  double phimin;
  double phimax;
  double ptmax;
  double pmax;
  double eopmax;
  double eopmaxsht;
  double detamin;
  double detamax;
  double dphimin;
  double dphimax;
  double detamatchmin;
  double detamatchmax;
  double dphimatchmin;
  double dphimatchmax;
  double fhitsmax;
  double lhitsmax;
  double poptruemin;
  double poptruemax;
  double meemin;
  double meemax;
  double hoemin;
  double hoemax;
  int nbineta;
  int nbinp;
  int nbinpt;
  int nbinpteff;
  int nbinphi;
  int nbinp2D;
  int nbinpt2D;
  int nbineta2D;
  int nbinphi2D;
  int nbineop;
  int nbineop2D;
  int nbinfhits;
  int nbinlhits;
  int nbinxyz;
  int nbindeta;
  int nbindphi;
  int nbindetamatch;
  int nbindphimatch;
  int nbindetamatch2D;
  int nbindphimatch2D;
  int nbinpoptrue;
  int nbinmee;
  int nbinhoe;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(GsfElectronMCAnalyzer);

using namespace reco;

GsfElectronMCAnalyzer::GsfElectronMCAnalyzer(const edm::ParameterSet &conf) {
  outputFile_ = conf.getParameter<std::string>("outputFile");
  histfile_ = new TFile(outputFile_.c_str(), "RECREATE");
  electronCollection_ = conf.getParameter<edm::InputTag>("electronCollection");
  mcTruthCollection_ = conf.getParameter<edm::InputTag>("mcTruthCollection");
  readAOD_ = conf.getParameter<bool>("readAOD");
  maxPt_ = conf.getParameter<double>("MaxPt");
  maxAbsEta_ = conf.getParameter<double>("MaxAbsEta");
  deltaR_ = conf.getParameter<double>("DeltaR");
  matchingIDs_ = conf.getParameter<std::vector<int> >("MatchingID");
  matchingMotherIDs_ = conf.getParameter<std::vector<int> >("MatchingMotherID");
  edm::ParameterSet pset = conf.getParameter<edm::ParameterSet>("HistosConfigurationMC");

  etamin = pset.getParameter<double>("Etamin");
  etamax = pset.getParameter<double>("Etamax");
  phimin = pset.getParameter<double>("Phimin");
  phimax = pset.getParameter<double>("Phimax");
  ptmax = pset.getParameter<double>("Ptmax");
  pmax = pset.getParameter<double>("Pmax");
  eopmax = pset.getParameter<double>("Eopmax");
  eopmaxsht = pset.getParameter<double>("Eopmaxsht");
  detamin = pset.getParameter<double>("Detamin");
  detamax = pset.getParameter<double>("Detamax");
  dphimin = pset.getParameter<double>("Dphimin");
  dphimax = pset.getParameter<double>("Dphimax");
  detamatchmin = pset.getParameter<double>("Detamatchmin");
  detamatchmax = pset.getParameter<double>("Detamatchmax");
  dphimatchmin = pset.getParameter<double>("Dphimatchmin");
  dphimatchmax = pset.getParameter<double>("Dphimatchmax");
  fhitsmax = pset.getParameter<double>("Fhitsmax");
  lhitsmax = pset.getParameter<double>("Lhitsmax");
  nbineta = pset.getParameter<int>("Nbineta");
  nbineta2D = pset.getParameter<int>("Nbineta2D");
  nbinp = pset.getParameter<int>("Nbinp");
  nbinpt = pset.getParameter<int>("Nbinpt");
  nbinp2D = pset.getParameter<int>("Nbinp2D");
  nbinpt2D = pset.getParameter<int>("Nbinpt2D");
  nbinpteff = pset.getParameter<int>("Nbinpteff");
  nbinphi = pset.getParameter<int>("Nbinphi");
  nbinphi2D = pset.getParameter<int>("Nbinphi2D");
  nbineop = pset.getParameter<int>("Nbineop");
  nbineop2D = pset.getParameter<int>("Nbineop2D");
  nbinfhits = pset.getParameter<int>("Nbinfhits");
  nbinlhits = pset.getParameter<int>("Nbinlhits");
  nbinxyz = pset.getParameter<int>("Nbinxyz");
  nbindeta = pset.getParameter<int>("Nbindeta");
  nbindphi = pset.getParameter<int>("Nbindphi");
  nbindetamatch = pset.getParameter<int>("Nbindetamatch");
  nbindphimatch = pset.getParameter<int>("Nbindphimatch");
  nbindetamatch2D = pset.getParameter<int>("Nbindetamatch2D");
  nbindphimatch2D = pset.getParameter<int>("Nbindphimatch2D");
  nbinpoptrue = pset.getParameter<int>("Nbinpoptrue");
  poptruemin = pset.getParameter<double>("Poptruemin");
  poptruemax = pset.getParameter<double>("Poptruemax");
  nbinmee = pset.getParameter<int>("Nbinmee");
  meemin = pset.getParameter<double>("Meemin");
  meemax = pset.getParameter<double>("Meemax");
  nbinhoe = pset.getParameter<int>("Nbinhoe");
  hoemin = pset.getParameter<double>("Hoemin");
  hoemax = pset.getParameter<double>("Hoemax");
}

void GsfElectronMCAnalyzer::beginJob() {
  histfile_->cd();

  // mc truth
  h_mcNum = new TH1F("h_mcNum", "# mc particles", nbinfhits, 0., fhitsmax);
  h_mcNum->Sumw2();
  h_eleNum = new TH1F("h_mcNum_ele", "# mc electrons", nbinfhits, 0., fhitsmax);
  h_eleNum->Sumw2();
  h_gamNum = new TH1F("h_mcNum_gam", "# mc gammas", nbinfhits, 0., fhitsmax);
  h_gamNum->Sumw2();

  // rec event
  histNum_ = new TH1F("h_recEleNum", "# rec electrons", 20, 0., 20.);

  // mc
  h_simEta = new TH1F("h_mc_eta", "gen #eta", nbineta, etamin, etamax);
  h_simEta->Sumw2();
  h_simAbsEta = new TH1F("h_mc_abseta", "gen |#eta|", nbineta / 2, 0., etamax);
  h_simAbsEta->Sumw2();
  h_simP = new TH1F("h_mc_P", "gen p", nbinp, 0., pmax);
  h_simP->Sumw2();
  h_simPt = new TH1F("h_mc_Pt", "gen pt", nbinpteff, 5., ptmax);
  h_simPt->Sumw2();
  h_simPhi = new TH1F("h_mc_phi", "gen phi", nbinphi, phimin, phimax);
  h_simPhi->Sumw2();
  h_simZ = new TH1F("h_mc_z", "gen z ", nbinxyz, -25, 25);
  h_simZ->Sumw2();
  h_simPtEta = new TH2F("h_mc_pteta", "gen pt vs #eta", nbineta2D, etamin, etamax, nbinpt2D, 5., ptmax);
  h_simPtEta->Sumw2();

  // all electrons
  h_ele_EoverP_all = new TH1F("h_ele_EoverP_all", "ele E/P_{vertex}, all reco electrons", nbineop, 0., eopmax);
  h_ele_EoverP_all->Sumw2();
  h_ele_EoverP_all_barrel =
      new TH1F("h_ele_EoverP_all_barrel", "ele E/P_{vertex}, all reco electrons, barrel", nbineop, 0., eopmax);
  h_ele_EoverP_all_barrel->Sumw2();
  h_ele_EoverP_all_endcaps =
      new TH1F("h_ele_EoverP_all_endcaps", "ele E/P_{vertex}, all reco electrons, endcaps", nbineop, 0., eopmax);
  h_ele_EoverP_all_endcaps->Sumw2();
  h_ele_EseedOP_all = new TH1F("h_ele_EseedOP_all", "ele E_{seed}/P_{vertex}, all reco electrons", nbineop, 0., eopmax);
  h_ele_EseedOP_all->Sumw2();
  h_ele_EseedOP_all_barrel =
      new TH1F("h_ele_EseedOP_all_barrel", "ele E_{seed}/P_{vertex}, all reco electrons, barrel", nbineop, 0., eopmax);
  h_ele_EseedOP_all_barrel->Sumw2();
  h_ele_EseedOP_all_endcaps = new TH1F(
      "h_ele_EseedOP_all_endcaps", "ele E_{seed}/P_{vertex}, all reco electrons, endcaps", nbineop, 0., eopmax);
  h_ele_EseedOP_all_endcaps->Sumw2();
  h_ele_EoPout_all = new TH1F("h_ele_EoPout_all", "ele E_{seed}/P_{out}, all reco electrons", nbineop, 0., eopmax);
  h_ele_EoPout_all->Sumw2();
  h_ele_EoPout_all_barrel =
      new TH1F("h_ele_EoPout_all_barrel", "ele E_{seed}/P_{out}, all reco electrons barrel", nbineop, 0., eopmax);
  h_ele_EoPout_all_barrel->Sumw2();
  h_ele_EoPout_all_endcaps =
      new TH1F("h_ele_EoPout_all_endcaps", "ele E_{seed}/P_{out}, all reco electrons endcaps", nbineop, 0., eopmax);
  h_ele_EoPout_all_endcaps->Sumw2();
  h_ele_EeleOPout_all = new TH1F("h_ele_EeleOPout_all", "ele E_{ele}/P_{out}, all reco electrons", nbineop, 0., eopmax);
  h_ele_EeleOPout_all->Sumw2();
  h_ele_EeleOPout_all_barrel =
      new TH1F("h_ele_EeleOPout_all_barrel", "ele E_{ele}/P_{out}, all reco electrons barrel", nbineop, 0., eopmax);
  h_ele_EeleOPout_all_barrel->Sumw2();
  h_ele_EeleOPout_all_endcaps =
      new TH1F("h_ele_EeleOPout_all_endcaps", "ele E_{ele}/P_{out}, all reco electrons endcaps", nbineop, 0., eopmax);
  h_ele_EeleOPout_all_endcaps->Sumw2();
  h_ele_dEtaSc_propVtx_all = new TH1F("h_ele_dEtaSc_propVtx_all",
                                      "ele #eta_{sc} - #eta_{tr}, prop from vertex, all reco electrons",
                                      nbindetamatch,
                                      detamatchmin,
                                      detamatchmax);
  h_ele_dEtaSc_propVtx_all->Sumw2();
  h_ele_dEtaSc_propVtx_all_barrel = new TH1F("h_ele_dEtaSc_propVtx_all_barrel",
                                             "ele #eta_{sc} - #eta_{tr}, prop from vertex, all reco electrons barrel",
                                             nbindetamatch,
                                             detamatchmin,
                                             detamatchmax);
  h_ele_dEtaSc_propVtx_all_barrel->Sumw2();
  h_ele_dEtaSc_propVtx_all_endcaps = new TH1F("h_ele_dEtaSc_propVtx_all_endcaps",
                                              "ele #eta_{sc} - #eta_{tr}, prop from vertex, all reco electrons endcaps",
                                              nbindetamatch,
                                              detamatchmin,
                                              detamatchmax);
  h_ele_dEtaSc_propVtx_all_endcaps->Sumw2();
  h_ele_dPhiSc_propVtx_all = new TH1F("h_ele_dPhiSc_propVtx_all",
                                      "ele #phi_{sc} - #phi_{tr}, prop from vertex, all reco electrons",
                                      nbindphimatch,
                                      dphimatchmin,
                                      dphimatchmax);
  h_ele_dPhiSc_propVtx_all->Sumw2();
  h_ele_dPhiSc_propVtx_all_barrel = new TH1F("h_ele_dPhiSc_propVtx_all_barrel",
                                             "ele #phi_{sc} - #phi_{tr}, prop from vertex, all reco electrons barrel",
                                             nbindphimatch,
                                             dphimatchmin,
                                             dphimatchmax);
  h_ele_dPhiSc_propVtx_all_barrel->Sumw2();
  h_ele_dPhiSc_propVtx_all_endcaps = new TH1F("h_ele_dPhiSc_propVtx_all_endcaps",
                                              "ele #phi_{sc} - #phi_{tr}, prop from vertex, all reco electrons endcaps",
                                              nbindphimatch,
                                              dphimatchmin,
                                              dphimatchmax);
  h_ele_dPhiSc_propVtx_all_endcaps->Sumw2();
  h_ele_dEtaCl_propOut_all = new TH1F("h_ele_dEtaCl_propOut_all",
                                      "ele #eta_{cl} - #eta_{tr}, prop from outermost, all reco electrons",
                                      nbindetamatch,
                                      detamatchmin,
                                      detamatchmax);
  h_ele_dEtaCl_propOut_all->Sumw2();
  h_ele_dEtaCl_propOut_all_barrel =
      new TH1F("h_ele_dEtaCl_propOut_all_barrel",
               "ele #eta_{cl} - #eta_{tr}, prop from outermost, all reco electrons barrel",
               nbindetamatch,
               detamatchmin,
               detamatchmax);
  h_ele_dEtaCl_propOut_all_barrel->Sumw2();
  h_ele_dEtaCl_propOut_all_endcaps =
      new TH1F("h_ele_dEtaCl_propOut_all_endcaps",
               "ele #eta_{cl} - #eta_{tr}, prop from outermost, all reco electrons endcaps",
               nbindetamatch,
               detamatchmin,
               detamatchmax);
  h_ele_dEtaCl_propOut_all_endcaps->Sumw2();
  h_ele_dPhiCl_propOut_all = new TH1F("h_ele_dPhiCl_propOut_all",
                                      "ele #phi_{cl} - #phi_{tr}, prop from outermost, all reco electrons",
                                      nbindphimatch,
                                      dphimatchmin,
                                      dphimatchmax);
  h_ele_dPhiCl_propOut_all->Sumw2();
  h_ele_dPhiCl_propOut_all_barrel =
      new TH1F("h_ele_dPhiCl_propOut_all_barrel",
               "ele #phi_{cl} - #phi_{tr}, prop from outermost, all reco electrons barrel",
               nbindphimatch,
               dphimatchmin,
               dphimatchmax);
  h_ele_dPhiCl_propOut_all_endcaps =
      new TH1F("h_ele_dPhiCl_propOut_all_endcaps",
               "ele #phi_{cl} - #phi_{tr}, prop from outermost, all reco electrons endcaps",
               nbindphimatch,
               dphimatchmin,
               dphimatchmax);
  h_ele_dPhiCl_propOut_all_barrel->Sumw2();
  h_ele_dPhiCl_propOut_all_endcaps->Sumw2();
  h_ele_HoE_all =
      new TH1F("h_ele_HoE_all", "ele hadronic energy / em energy, all reco electrons", nbinhoe, hoemin, hoemax);
  h_ele_HoE_all->Sumw2();
  h_ele_HoE_all_barrel = new TH1F(
      "h_ele_HoE_all_barrel", "ele hadronic energy / em energy, all reco electrons barrel", nbinhoe, hoemin, hoemax);
  h_ele_HoE_all_barrel->Sumw2();
  h_ele_HoE_all_endcaps = new TH1F(
      "h_ele_HoE_all_endcaps", "ele hadronic energy / em energy, all reco electrons endcaps", nbinhoe, hoemin, hoemax);
  h_ele_HoE_all_endcaps->Sumw2();
  h_ele_vertexPt_all = new TH1F("h_ele_vertexPt_all", "ele p_{T}, all reco electrons", nbinpteff, 5., ptmax);
  h_ele_vertexPt_all->Sumw2();
  h_ele_Et_all = new TH1F("h_ele_Et_all", "ele SC E_{T}, all reco electrons", nbinpteff, 5., ptmax);
  h_ele_Et_all->Sumw2();
  h_ele_vertexEta_all = new TH1F("h_ele_vertexEta_all", "ele eta, all reco electrons", nbineta, etamin, etamax);
  h_ele_vertexEta_all->Sumw2();
  h_ele_TIP_all = new TH1F("h_ele_TIP_all", "ele vertex transverse radius, all reco electrons", 100, 0., 0.2);
  h_ele_TIP_all->Sumw2();
  h_ele_TIP_all_barrel =
      new TH1F("h_ele_TIP_all_barrel", "ele vertex transverse radius, all reco electrons barrel", 100, 0., 0.2);
  h_ele_TIP_all_barrel->Sumw2();
  h_ele_TIP_all_endcaps =
      new TH1F("h_ele_TIP_all_endcaps", "ele vertex transverse radius, all reco electrons endcaps", 100, 0., 0.2);
  h_ele_TIP_all_endcaps->Sumw2();
  h_ele_mee_all = new TH1F("h_ele_mee_all", "ele pairs invariant mass, all reco electrons", nbinmee, meemin, meemax);
  h_ele_mee_all->Sumw2();
  h_ele_mee_os = new TH1F("h_ele_mee_os", "ele pairs invariant mass, opp. sign", nbinmee, meemin, meemax);
  h_ele_mee_os->Sumw2();
  h_ele_mee_os_ebeb =
      new TH1F("h_ele_mee_os_ebeb", "ele pairs invariant mass, opp. sign, EB-EB", nbinmee, meemin, meemax);
  h_ele_mee_os_ebeb->Sumw2();
  h_ele_mee_os_ebee =
      new TH1F("h_ele_mee_os_ebee", "ele pairs invariant mass, opp. sign, EB-EE", nbinmee, meemin, meemax);
  h_ele_mee_os_ebee->Sumw2();
  h_ele_mee_os_eeee =
      new TH1F("h_ele_mee_os_eeee", "ele pairs invariant mass, opp. sign, EE-EE", nbinmee, meemin, meemax);
  h_ele_mee_os_eeee->Sumw2();
  h_ele_mee_os_gg =
      new TH1F("h_ele_mee_os_gg", "ele pairs invariant mass, opp. sign, good-good", nbinmee, meemin, meemax);
  h_ele_mee_os_gg->Sumw2();
  h_ele_mee_os_gb =
      new TH1F("h_ele_mee_os_gb", "ele pairs invariant mass, opp. sign, good-bad", nbinmee, meemin, meemax);
  h_ele_mee_os_gb->Sumw2();
  h_ele_mee_os_bb =
      new TH1F("h_ele_mee_os_bb", "ele pairs invariant mass, opp. sign, bad-bad", nbinmee, meemin, meemax);
  h_ele_mee_os_bb->Sumw2();

  // duplicates
  h_ele_E2mnE1vsMee_all = new TH2F("h_ele_E2mnE1vsMee_all",
                                   "E2 - E1 vs ele pairs invariant mass, all electrons",
                                   nbinmee,
                                   meemin,
                                   meemax,
                                   100,
                                   -50.,
                                   50.);
  h_ele_E2mnE1vsMee_egeg_all = new TH2F("h_ele_E2mnE1vsMee_egeg_all",
                                        "E2 - E1 vs ele pairs invariant mass, ecal driven pairs, all electrons",
                                        nbinmee,
                                        meemin,
                                        meemax,
                                        100,
                                        -50.,
                                        50.);

  // charge ID
  h_ele_ChargeMnChargeTrue = new TH1F("h_ele_ChargeMnChargeTrue", "ele charge - gen charge ", 5, -1., 4.);
  h_ele_ChargeMnChargeTrue->Sumw2();
  h_ele_simEta_matched_qmisid =
      new TH1F("h_ele_eta_matched_qmisid", "charge misid vs gen eta", nbineta, etamin, etamax);
  h_ele_simEta_matched_qmisid->Sumw2();
  h_ele_simAbsEta_matched_qmisid =
      new TH1F("h_ele_abseta_matched_qmisid", "charge misid vs gen |eta|", nbineta / 2, 0., etamax);
  h_ele_simAbsEta_matched_qmisid->Sumw2();
  h_ele_simPt_matched_qmisid =
      new TH1F("h_ele_Pt_matched_qmisid", "charge misid vs gen transverse momentum", nbinpteff, 5., ptmax);
  h_ele_simPt_matched_qmisid->Sumw2();
  h_ele_simPhi_matched_qmisid =
      new TH1F("h_ele_phi_matched_qmisid", "charge misid vs gen phi", nbinphi, phimin, phimax);
  h_ele_simPhi_matched_qmisid->Sumw2();
  h_ele_simZ_matched_qmisid = new TH1F("h_ele_z_matched_qmisid", "charge misid vs gen z", nbinxyz, -25, 25);
  h_ele_simZ_matched_qmisid->Sumw2();

  // matched electrons
  h_ele_charge = new TH1F("h_ele_charge", "ele charge", 5, -2., 2.);
  h_ele_charge->Sumw2();
  h_ele_chargeVsEta = new TH2F("h_ele_chargeVsEta", "ele charge vs eta", nbineta2D, etamin, etamax, 5, -2., 2.);
  h_ele_chargeVsPhi = new TH2F("h_ele_chargeVsPhi", "ele charge vs phi", nbinphi2D, phimin, phimax, 5, -2., 2.);
  h_ele_chargeVsPt = new TH2F("h_ele_chargeVsPt", "ele charge vs pt", nbinpt, 0., 100., 5, -2., 2.);
  h_ele_vertexP = new TH1F("h_ele_vertexP", "ele momentum", nbinp, 0., pmax);
  h_ele_vertexP->Sumw2();
  h_ele_vertexPt = new TH1F("h_ele_vertexPt", "ele transverse momentum", nbinpt, 0., ptmax);
  h_ele_vertexPt->Sumw2();
  h_ele_Et = new TH1F("h_ele_Et", "ele transverse energy", nbinpt, 0., ptmax);
  h_ele_Et->Sumw2();
  h_ele_vertexPtVsEta =
      new TH2F("h_ele_vertexPtVsEta", "ele transverse momentum vs eta", nbineta2D, etamin, etamax, nbinpt2D, 0., ptmax);
  h_ele_vertexPtVsPhi =
      new TH2F("h_ele_vertexPtVsPhi", "ele transverse momentum vs phi", nbinphi2D, phimin, phimax, nbinpt2D, 0., ptmax);
  h_ele_simPt_matched = new TH1F("h_ele_simPt_matched", "Efficiency vs gen transverse momentum", nbinpteff, 5., ptmax);
  h_ele_vertexEta = new TH1F("h_ele_vertexEta", "ele momentum eta", nbineta, etamin, etamax);
  h_ele_vertexEta->Sumw2();
  h_ele_vertexEtaVsPhi =
      new TH2F("h_ele_vertexEtaVsPhi", "ele momentum eta vs phi", nbineta2D, etamin, etamax, nbinphi2D, phimin, phimax);
  h_ele_simAbsEta_matched = new TH1F("h_ele_simAbsEta_matched", "Efficiency vs gen |eta|", nbineta / 2, 0., 2.5);
  h_ele_simAbsEta_matched->Sumw2();
  h_ele_simEta_matched = new TH1F("h_ele_simEta_matched", "Efficiency vs gen eta", nbineta, etamin, etamax);
  h_ele_simEta_matched->Sumw2();
  h_ele_simPtEta_matched =
      new TH2F("h_ele_simPtEta_matched", "Efficiency vs pt #eta", nbineta2D, etamin, etamax, nbinpt2D, 5., ptmax);
  h_ele_simPtEta_matched->Sumw2();
  h_ele_simPhi_matched = new TH1F("h_ele_simPhi_matched", "Efficiency vs gen phi", nbinphi, phimin, phimax);
  h_ele_simPhi_matched->Sumw2();
  h_ele_vertexPhi = new TH1F("h_ele_vertexPhi", "ele  momentum #phi", nbinphi, phimin, phimax);
  h_ele_vertexPhi->Sumw2();
  h_ele_vertexX = new TH1F("h_ele_vertexX", "ele vertex x", nbinxyz, -0.1, 0.1);
  h_ele_vertexX->Sumw2();
  h_ele_vertexY = new TH1F("h_ele_vertexY", "ele vertex y", nbinxyz, -0.1, 0.1);
  h_ele_vertexY->Sumw2();
  h_ele_vertexZ = new TH1F("h_ele_vertexZ", "ele vertex z", nbinxyz, -25, 25);
  h_ele_vertexZ->Sumw2();
  h_ele_simZ_matched = new TH1F("h_ele_simZ_matched", "Efficiency vs gen vertex z", nbinxyz, -25, 25);
  h_ele_simZ_matched->Sumw2();
  h_ele_vertexTIP = new TH1F("h_ele_vertexTIP", "ele transverse impact parameter (wrt gen vtx)", 90, 0., 0.15);
  h_ele_vertexTIP->Sumw2();
  h_ele_vertexTIPVsEta = new TH2F("h_ele_vertexTIPVsEta",
                                  "ele transverse impact parameter (wrt gen vtx) vs eta",
                                  nbineta2D,
                                  etamin,
                                  etamax,
                                  45,
                                  0.,
                                  0.15);
  h_ele_vertexTIPVsPhi = new TH2F("h_ele_vertexTIPVsPhi",
                                  "ele transverse impact parameter (wrt gen vtx) vs phi",
                                  nbinphi2D,
                                  phimin,
                                  phimax,
                                  45,
                                  0.,
                                  0.15);
  h_ele_vertexTIPVsPt = new TH2F("h_ele_vertexTIPVsPt",
                                 "ele transverse impact parameter (wrt gen vtx) vs transverse momentum",
                                 nbinpt2D,
                                 0.,
                                 ptmax,
                                 45,
                                 0.,
                                 0.15);
  h_ele_PoPtrue = new TH1F("h_ele_PoPtrue", "ele momentum / gen momentum", nbinpoptrue, poptruemin, poptruemax);
  h_ele_PoPtrue->Sumw2();
  h_ele_PtoPttrue = new TH1F(
      "h_ele_PtoPttrue", "ele transverse momentum / gen transverse momentum", nbinpoptrue, poptruemin, poptruemax);
  h_ele_PtoPttrue->Sumw2();
  h_ele_PoPtrueVsEta = new TH2F(
      "h_ele_PoPtrueVsEta", "ele momentum / gen momentum vs eta", nbineta2D, etamin, etamax, 50, poptruemin, poptruemax);
  h_ele_PoPtrueVsPhi = new TH2F(
      "h_ele_PoPtrueVsPhi", "ele momentum / gen momentum vs phi", nbinphi2D, phimin, phimax, 50, poptruemin, poptruemax);
  h_ele_PoPtrueVsPt = new TH2F(
      "h_ele_PoPtrueVsPt", "ele momentum / gen momentum vs eta", nbinpt2D, 0., ptmax, 50, poptruemin, poptruemax);
  h_ele_PoPtrue_barrel =
      new TH1F("h_ele_PoPtrue_barrel", "ele momentum / gen momentum, barrel", nbinpoptrue, poptruemin, poptruemax);
  h_ele_PoPtrue_barrel->Sumw2();
  h_ele_PoPtrue_endcaps =
      new TH1F("h_ele_PoPtrue_endcaps", "ele momentum / gen momentum, endcaps", nbinpoptrue, poptruemin, poptruemax);
  h_ele_PoPtrue_endcaps->Sumw2();
  h_ele_PoPtrue_golden_barrel = new TH1F(
      "h_ele_PoPtrue_golden_barrel", "ele momentum / gen momentum, golden, barrel", nbinpoptrue, poptruemin, poptruemax);
  h_ele_PoPtrue_golden_barrel->Sumw2();
  h_ele_PoPtrue_golden_endcaps = new TH1F("h_ele_PoPtrue_golden_endcaps",
                                          "ele momentum / gen momentum, golden, endcaps",
                                          nbinpoptrue,
                                          poptruemin,
                                          poptruemax);
  h_ele_PoPtrue_golden_endcaps->Sumw2();
  h_ele_PoPtrue_showering_barrel = new TH1F("h_ele_PoPtrue_showering_barrel",
                                            "ele momentum / gen momentum, showering, barrel",
                                            nbinpoptrue,
                                            poptruemin,
                                            poptruemax);
  h_ele_PoPtrue_showering_barrel->Sumw2();
  h_ele_PoPtrue_showering_endcaps = new TH1F("h_ele_PoPtrue_showering_endcaps",
                                             "ele momentum / gen momentum, showering, endcaps",
                                             nbinpoptrue,
                                             poptruemin,
                                             poptruemax);
  h_ele_PoPtrue_showering_endcaps->Sumw2();
  h_ele_PtoPttrue_barrel = new TH1F("h_ele_PtoPttrue_barrel",
                                    "ele transverse momentum / gen transverse momentum, barrel",
                                    nbinpoptrue,
                                    poptruemin,
                                    poptruemax);
  h_ele_PtoPttrue_barrel->Sumw2();
  h_ele_PtoPttrue_endcaps = new TH1F("h_ele_PtoPttrue_endcaps",
                                     "ele transverse momentum / gen transverse momentum, endcaps",
                                     nbinpoptrue,
                                     poptruemin,
                                     poptruemax);
  h_ele_PtoPttrue_endcaps->Sumw2();
  h_ele_EtaMnEtaTrue = new TH1F("h_ele_EtaMnEtaTrue", "ele momentum  eta - gen  eta", nbindeta, detamin, detamax);
  h_ele_EtaMnEtaTrue->Sumw2();
  h_ele_EtaMnEtaTrue_barrel =
      new TH1F("h_ele_EtaMnEtaTrue_barrel", "ele momentum  eta - gen  eta barrel", nbindeta, detamin, detamax);
  h_ele_EtaMnEtaTrue_barrel->Sumw2();
  h_ele_EtaMnEtaTrue_endcaps =
      new TH1F("h_ele_EtaMnEtaTrue_endcaps", "ele momentum  eta - gen  eta endcaps", nbindeta, detamin, detamax);
  h_ele_EtaMnEtaTrue_endcaps->Sumw2();
  h_ele_EtaMnEtaTrueVsEta = new TH2F("h_ele_EtaMnEtaTrueVsEta",
                                     "ele momentum  eta - gen  eta vs eta",
                                     nbineta2D,
                                     etamin,
                                     etamax,
                                     nbindeta / 2,
                                     detamin,
                                     detamax);
  h_ele_EtaMnEtaTrueVsPhi = new TH2F("h_ele_EtaMnEtaTrueVsPhi",
                                     "ele momentum  eta - gen  eta vs phi",
                                     nbinphi2D,
                                     phimin,
                                     phimax,
                                     nbindeta / 2,
                                     detamin,
                                     detamax);
  h_ele_EtaMnEtaTrueVsPt = new TH2F(
      "h_ele_EtaMnEtaTrueVsPt", "ele momentum  eta - gen  eta vs pt", nbinpt, 0., ptmax, nbindeta / 2, detamin, detamax);
  h_ele_PhiMnPhiTrue = new TH1F("h_ele_PhiMnPhiTrue", "ele momentum  phi - gen  phi", nbindphi, dphimin, dphimax);
  h_ele_PhiMnPhiTrue->Sumw2();
  h_ele_PhiMnPhiTrue_barrel =
      new TH1F("h_ele_PhiMnPhiTrue_barrel", "ele momentum  phi - gen  phi barrel", nbindphi, dphimin, dphimax);
  h_ele_PhiMnPhiTrue_barrel->Sumw2();
  h_ele_PhiMnPhiTrue_endcaps =
      new TH1F("h_ele_PhiMnPhiTrue_endcaps", "ele momentum  phi - gen  phi endcaps", nbindphi, dphimin, dphimax);
  h_ele_PhiMnPhiTrue_endcaps->Sumw2();
  h_ele_PhiMnPhiTrue2 =
      new TH1F("h_ele_PhiMnPhiTrue2", "ele momentum  phi - gen  phi", nbindphimatch2D, dphimatchmin, dphimatchmax);
  h_ele_PhiMnPhiTrueVsEta = new TH2F("h_ele_PhiMnPhiTrueVsEta",
                                     "ele momentum  phi - gen  phi vs eta",
                                     nbineta2D,
                                     etamin,
                                     etamax,
                                     nbindphi / 2,
                                     dphimin,
                                     dphimax);
  h_ele_PhiMnPhiTrueVsPhi = new TH2F("h_ele_PhiMnPhiTrueVsPhi",
                                     "ele momentum  phi - gen  phi vs phi",
                                     nbinphi2D,
                                     phimin,
                                     phimax,
                                     nbindphi / 2,
                                     dphimin,
                                     dphimax);
  h_ele_PhiMnPhiTrueVsPt = new TH2F("h_ele_PhiMnPhiTrueVsPt",
                                    "ele momentum  phi - gen  phi vs pt",
                                    nbinpt2D,
                                    0.,
                                    ptmax,
                                    nbindphi / 2,
                                    dphimin,
                                    dphimax);

  // matched electron, superclusters
  histSclEn_ = new TH1F("h_scl_energy", "ele supercluster energy", nbinp, 0., pmax);
  histSclEn_->Sumw2();
  histSclEoEtrue_barrel =
      new TH1F("h_scl_EoEtrue_barrel", "ele supercluster energy / gen energy, barrel", 50, 0.2, 1.2);
  histSclEoEtrue_barrel->Sumw2();
  histSclEoEtrue_barrel_eg =
      new TH1F("h_scl_EoEtrue_barrel_eg", "ele supercluster energy / gen energy, barrel, ecal driven", 50, 0.2, 1.2);
  histSclEoEtrue_barrel_eg->Sumw2();
  histSclEoEtrue_barrel_etagap =
      new TH1F("h_scl_EoEtrue_barrel_etagap", "ele supercluster energy / gen energy, barrel, etagap", 50, 0.2, 1.2);
  histSclEoEtrue_barrel_etagap->Sumw2();
  histSclEoEtrue_barrel_phigap =
      new TH1F("h_scl_EoEtrue_barrel_phigap", "ele supercluster energy / gen energy, barrel, phigap", 50, 0.2, 1.2);
  histSclEoEtrue_barrel_phigap->Sumw2();
  histSclEoEtrue_ebeegap =
      new TH1F("h_scl_EoEtrue_ebeegap", "ele supercluster energy / gen energy, ebeegap", 50, 0.2, 1.2);
  histSclEoEtrue_ebeegap->Sumw2();
  histSclEoEtrue_endcaps =
      new TH1F("h_scl_EoEtrue_endcaps", "ele supercluster energy / gen energy, endcaps", 50, 0.2, 1.2);
  histSclEoEtrue_endcaps->Sumw2();
  histSclEoEtrue_endcaps_eg =
      new TH1F("h_scl_EoEtrue_endcaps_eg", "ele supercluster energy / gen energy, endcaps, ecal driven", 50, 0.2, 1.2);
  histSclEoEtrue_endcaps_eg->Sumw2();
  histSclEoEtrue_endcaps_deegap =
      new TH1F("h_scl_EoEtrue_endcaps_deegap", "ele supercluster energy / gen energy, endcaps, deegap", 50, 0.2, 1.2);
  histSclEoEtrue_endcaps_deegap->Sumw2();
  histSclEoEtrue_endcaps_ringgap =
      new TH1F("h_scl_EoEtrue_endcaps_ringgap", "ele supercluster energy / gen energy, endcaps, ringgap", 50, 0.2, 1.2);
  histSclEoEtrue_endcaps_ringgap->Sumw2();
  histSclEoEtrue_barrel_new = new TH1F(
      "h_scl_EoEtrue_barrel_new", "ele supercluster energy / gen energy, barrel", nbinpoptrue, poptruemin, poptruemax);
  histSclEoEtrue_barrel_new->Sumw2();
  histSclEoEtrue_barrel_eg_new = new TH1F("h_scl_EoEtrue_barrel_eg_new",
                                          "ele supercluster energy / gen energy, barrel, ecal driven",
                                          nbinpoptrue,
                                          poptruemin,
                                          poptruemax);
  histSclEoEtrue_barrel_eg_new->Sumw2();
  histSclEoEtrue_barrel_etagap_new = new TH1F("h_scl_EoEtrue_barrel_etagap_new",
                                              "ele supercluster energy / gen energy, barrel, etagap",
                                              nbinpoptrue,
                                              poptruemin,
                                              poptruemax);
  histSclEoEtrue_barrel_etagap_new->Sumw2();
  histSclEoEtrue_barrel_phigap_new = new TH1F("h_scl_EoEtrue_barrel_phigap_new",
                                              "ele supercluster energy / gen energy, barrel, phigap",
                                              nbinpoptrue,
                                              poptruemin,
                                              poptruemax);
  histSclEoEtrue_barrel_phigap_new->Sumw2();
  histSclEoEtrue_ebeegap_new = new TH1F(
      "h_scl_EoEtrue_ebeegap_new", "ele supercluster energy / gen energy, ebeegap", nbinpoptrue, poptruemin, poptruemax);
  histSclEoEtrue_ebeegap_new->Sumw2();
  histSclEoEtrue_endcaps_new = new TH1F(
      "h_scl_EoEtrue_endcaps_new", "ele supercluster energy / gen energy, endcaps", nbinpoptrue, poptruemin, poptruemax);
  histSclEoEtrue_endcaps_new->Sumw2();
  histSclEoEtrue_endcaps_eg_new = new TH1F("h_scl_EoEtrue_endcaps_eg_new",
                                           "ele supercluster energy / gen energy, endcaps, ecal driven",
                                           nbinpoptrue,
                                           poptruemin,
                                           poptruemax);
  histSclEoEtrue_endcaps_eg_new->Sumw2();
  histSclEoEtrue_endcaps_deegap_new = new TH1F("h_scl_EoEtrue_endcaps_deegap_new",
                                               "ele supercluster energy / gen energy, endcaps, deegap",
                                               nbinpoptrue,
                                               poptruemin,
                                               poptruemax);
  histSclEoEtrue_endcaps_deegap_new->Sumw2();
  histSclEoEtrue_endcaps_ringgap_new = new TH1F("h_scl_EoEtrue_endcaps_ringgap_new",
                                                "ele supercluster energy / gen energy, endcaps, ringgap",
                                                nbinpoptrue,
                                                poptruemin,
                                                poptruemax);
  histSclEoEtrue_endcaps_ringgap_new->Sumw2();
  histSclEt_ = new TH1F("h_scl_et", "ele supercluster transverse energy", nbinpt, 0., ptmax);
  histSclEt_->Sumw2();
  histSclEtVsEta_ = new TH2F(
      "h_scl_etVsEta", "ele supercluster transverse energy vs eta", nbineta2D, etamin, etamax, nbinpt, 0., ptmax);
  histSclEtVsPhi_ = new TH2F(
      "h_scl_etVsPhi", "ele supercluster transverse energy vs phi", nbinphi2D, phimin, phimax, nbinpt, 0., ptmax);
  histSclEtaVsPhi_ =
      new TH2F("h_scl_etaVsPhi", "ele supercluster eta vs phi", nbinphi2D, phimin, phimax, nbineta2D, etamin, etamax);
  histSclEta_ = new TH1F("h_scl_eta", "ele supercluster eta", nbineta, etamin, etamax);
  histSclEta_->Sumw2();
  histSclPhi_ = new TH1F("h_scl_phi", "ele supercluster phi", nbinphi, phimin, phimax);
  histSclPhi_->Sumw2();

  histSclSigEtaEta_ = new TH1F("h_scl_sigetaeta", "ele supercluster sigma eta eta", 100, 0., 0.05);
  histSclSigEtaEta_->Sumw2();
  histSclSigEtaEta_barrel_ = new TH1F("h_scl_sigetaeta_barrel", "ele supercluster sigma eta eta barrel", 100, 0., 0.05);
  histSclSigEtaEta_barrel_->Sumw2();
  histSclSigEtaEta_endcaps_ =
      new TH1F("h_scl_sigetaeta_endcaps", "ele supercluster sigma eta eta endcaps", 100, 0., 0.05);
  histSclSigEtaEta_endcaps_->Sumw2();
  histSclSigIEtaIEta_ = new TH1F("h_scl_sigietaieta", "ele supercluster sigma ieta ieta", 100, 0., 0.05);
  histSclSigIEtaIEta_->Sumw2();
  histSclSigIEtaIEta_barrel_ =
      new TH1F("h_scl_sigietaieta_barrel", "ele supercluster sigma ieta ieta, barrel", 100, 0., 0.05);
  histSclSigIEtaIEta_barrel_->Sumw2();
  histSclSigIEtaIEta_endcaps_ =
      new TH1F("h_scl_sigietaieta_endcaps", "ele supercluster sigma ieta ieta, endcaps", 100, 0., 0.05);
  histSclSigIEtaIEta_endcaps_->Sumw2();
  histSclE1x5_ = new TH1F("h_scl_E1x5", "ele supercluster energy in 1x5", nbinp, 0., pmax);
  histSclE1x5_->Sumw2();
  histSclE1x5_barrel_ = new TH1F("h_scl_E1x5_barrel", "ele supercluster energy in 1x5 barrel", nbinp, 0., pmax);
  histSclE1x5_barrel_->Sumw2();
  histSclE1x5_endcaps_ = new TH1F("h_scl_E1x5_endcaps", "ele supercluster energy in 1x5 endcaps", nbinp, 0., pmax);
  histSclE1x5_endcaps_->Sumw2();
  histSclE2x5max_ = new TH1F("h_scl_E2x5max", "ele supercluster energy in 2x5 max", nbinp, 0., pmax);
  histSclE2x5max_->Sumw2();
  histSclE2x5max_barrel_ =
      new TH1F("h_scl_E2x5max_barrel", "ele supercluster energy in 2x5 max barrel", nbinp, 0., pmax);
  histSclE2x5max_barrel_->Sumw2();
  histSclE2x5max_endcaps_ =
      new TH1F("h_scl_E2x5max_endcaps", "ele supercluster energy in 2x5 max endcaps", nbinp, 0., pmax);
  histSclE2x5max_endcaps_->Sumw2();
  histSclE5x5_ = new TH1F("h_scl_E5x5", "ele supercluster energy in 5x5", nbinp, 0., pmax);
  histSclE5x5_->Sumw2();
  histSclE5x5_barrel_ = new TH1F("h_scl_E5x5_barrel", "ele supercluster energy in 5x5 barrel", nbinp, 0., pmax);
  histSclE5x5_barrel_->Sumw2();
  histSclE5x5_endcaps_ = new TH1F("h_scl_E5x5_endcaps", "ele supercluster energy in 5x5 endcaps", nbinp, 0., pmax);
  histSclE5x5_endcaps_->Sumw2();
  histSclSigEtaEta_eg_ = new TH1F("h_scl_sigetaeta_eg", "ele supercluster sigma eta eta, ecal driven", 100, 0., 0.05);
  histSclSigEtaEta_eg_->Sumw2();
  histSclSigEtaEta_eg_barrel_ =
      new TH1F("h_scl_sigetaeta_eg_barrel", "ele supercluster sigma eta eta, ecal driven barrel", 100, 0., 0.05);
  histSclSigEtaEta_eg_barrel_->Sumw2();
  histSclSigEtaEta_eg_endcaps_ =
      new TH1F("h_scl_sigetaeta_eg_endcaps", "ele supercluster sigma eta eta, ecal driven endcaps", 100, 0., 0.05);
  histSclSigEtaEta_eg_endcaps_->Sumw2();
  histSclSigIEtaIEta_eg_ =
      new TH1F("h_scl_sigietaieta_eg", "ele supercluster sigma ieta ieta, ecal driven", 100, 0., 0.05);
  histSclSigIEtaIEta_eg_->Sumw2();
  histSclSigIEtaIEta_eg_barrel_ =
      new TH1F("h_scl_sigietaieta_barrel_eg", "ele supercluster sigma ieta ieta, barrel, ecal driven", 100, 0., 0.05);
  histSclSigIEtaIEta_eg_barrel_->Sumw2();
  histSclSigIEtaIEta_eg_endcaps_ =
      new TH1F("h_scl_sigietaieta_endcaps_eg", "ele supercluster sigma ieta ieta, endcaps, ecal driven", 100, 0., 0.05);
  histSclSigIEtaIEta_eg_endcaps_->Sumw2();
  histSclE1x5_eg_ = new TH1F("h_scl_E1x5_eg", "ele supercluster energy in 1x5, ecal driven", nbinp, 0., pmax);
  histSclE1x5_eg_->Sumw2();
  histSclE1x5_eg_barrel_ =
      new TH1F("h_scl_E1x5_eg_barrel", "ele supercluster energy in 1x5, ecal driven barrel", nbinp, 0., pmax);
  histSclE1x5_eg_barrel_->Sumw2();
  histSclE1x5_eg_endcaps_ =
      new TH1F("h_scl_E1x5_eg_endcaps", "ele supercluster energy in 1x5, ecal driven endcaps", nbinp, 0., pmax);
  histSclE1x5_eg_endcaps_->Sumw2();
  histSclE2x5max_eg_ = new TH1F("h_scl_E2x5max_eg", "ele supercluster energy in 2x5 max, ecal driven", nbinp, 0., pmax);
  histSclE2x5max_eg_->Sumw2();
  histSclE2x5max_eg_barrel_ =
      new TH1F("h_scl_E2x5max_eg_barrel", "ele supercluster energy in 2x5 max, ecal driven barrel", nbinp, 0., pmax);
  histSclE2x5max_eg_barrel_->Sumw2();
  histSclE2x5max_eg_endcaps_ =
      new TH1F("h_scl_E2x5max_eg_endcaps", "ele supercluster energy in 2x5 max, ecal driven endcaps", nbinp, 0., pmax);
  histSclE2x5max_eg_endcaps_->Sumw2();
  histSclE5x5_eg_ = new TH1F("h_scl_E5x5_eg", "ele supercluster energy in 5x5, ecal driven", nbinp, 0., pmax);
  histSclE5x5_eg_->Sumw2();
  histSclE5x5_eg_barrel_ =
      new TH1F("h_scl_E5x5_eg_barrel", "ele supercluster energy in 5x5, ecal driven barrel", nbinp, 0., pmax);
  histSclE5x5_eg_barrel_->Sumw2();
  histSclE5x5_eg_endcaps_ =
      new TH1F("h_scl_E5x5_eg_endcaps", "ele supercluster energy in 5x5, ecal driven endcaps", nbinp, 0., pmax);
  histSclE5x5_eg_endcaps_->Sumw2();

  histSclEoEtruePfVsEg =
      new TH2F("h_scl_EoEtruePfVsEg", "ele supercluster energy / gen energy pflow vs eg", 75, -0.1, 1.4, 75, -0.1, 1.4);

  // matched electron, gsf tracks
  h_ele_ambiguousTracks = new TH1F("h_ele_ambiguousTracks", "ele # ambiguous tracks", 5, 0., 5.);
  h_ele_ambiguousTracks->Sumw2();
  h_ele_ambiguousTracksVsEta =
      new TH2F("h_ele_ambiguousTracksVsEta", "ele # ambiguous tracks  vs eta", nbineta2D, etamin, etamax, 5, 0., 5.);
  h_ele_ambiguousTracksVsPhi =
      new TH2F("h_ele_ambiguousTracksVsPhi", "ele # ambiguous tracks  vs phi", nbinphi2D, phimin, phimax, 5, 0., 5.);
  h_ele_ambiguousTracksVsPt =
      new TH2F("h_ele_ambiguousTracksVsPt", "ele # ambiguous tracks vs pt", nbinpt2D, 0., ptmax, 5, 0., 5.);
  h_ele_foundHits = new TH1F("h_ele_foundHits", "ele track # found hits", nbinfhits, 0., fhitsmax);
  h_ele_foundHits->Sumw2();
  h_ele_foundHits_barrel =
      new TH1F("h_ele_foundHits_barrel", "ele track # found hits, barrel", nbinfhits, 0., fhitsmax);
  h_ele_foundHits_barrel->Sumw2();
  h_ele_foundHits_endcaps =
      new TH1F("h_ele_foundHits_endcaps", "ele track # found hits, endcaps", nbinfhits, 0., fhitsmax);
  h_ele_foundHits_endcaps->Sumw2();
  h_ele_foundHitsVsEta = new TH2F(
      "h_ele_foundHitsVsEta", "ele track # found hits vs eta", nbineta2D, etamin, etamax, nbinfhits, 0., fhitsmax);
  h_ele_foundHitsVsPhi = new TH2F(
      "h_ele_foundHitsVsPhi", "ele track # found hits vs phi", nbinphi2D, phimin, phimax, nbinfhits, 0., fhitsmax);
  h_ele_foundHitsVsPt =
      new TH2F("h_ele_foundHitsVsPt", "ele track # found hits vs pt", nbinpt2D, 0., ptmax, nbinfhits, 0., fhitsmax);
  h_ele_lostHits = new TH1F("h_ele_lostHits", "ele track # lost hits", 5, 0., 5.);
  h_ele_lostHits->Sumw2();
  h_ele_lostHits_barrel = new TH1F("h_ele_lostHits_barrel", "ele track # lost hits, barrel", 5, 0., 5.);
  h_ele_lostHits_barrel->Sumw2();
  h_ele_lostHits_endcaps = new TH1F("h_ele_lostHits_endcaps", "ele track # lost hits, endcaps", 5, 0., 5.);
  h_ele_lostHits_endcaps->Sumw2();
  h_ele_lostHitsVsEta = new TH2F(
      "h_ele_lostHitsVsEta", "ele track # lost hits vs eta", nbineta2D, etamin, etamax, nbinlhits, 0., lhitsmax);
  h_ele_lostHitsVsPhi = new TH2F(
      "h_ele_lostHitsVsPhi", "ele track # lost hits vs eta", nbinphi2D, phimin, phimax, nbinlhits, 0., lhitsmax);
  h_ele_lostHitsVsPt =
      new TH2F("h_ele_lostHitsVsPt", "ele track # lost hits vs eta", nbinpt2D, 0., ptmax, nbinlhits, 0., lhitsmax);
  h_ele_chi2 = new TH1F("h_ele_chi2", "ele track #chi^{2}", 100, 0., 15.);
  h_ele_chi2->Sumw2();
  h_ele_chi2_barrel = new TH1F("h_ele_chi2_barrel", "ele track #chi^{2}, barrel", 100, 0., 15.);
  h_ele_chi2_barrel->Sumw2();
  h_ele_chi2_endcaps = new TH1F("h_ele_chi2_endcaps", "ele track #chi^{2}, endcaps", 100, 0., 15.);
  h_ele_chi2_endcaps->Sumw2();
  h_ele_chi2VsEta = new TH2F("h_ele_chi2VsEta", "ele track #chi^{2} vs eta", nbineta2D, etamin, etamax, 50, 0., 15.);
  h_ele_chi2VsPhi = new TH2F("h_ele_chi2VsPhi", "ele track #chi^{2} vs phi", nbinphi2D, phimin, phimax, 50, 0., 15.);
  h_ele_chi2VsPt = new TH2F("h_ele_chi2VsPt", "ele track #chi^{2} vs pt", nbinpt2D, 0., ptmax, 50, 0., 15.);
  h_ele_PinMnPout = new TH1F("h_ele_PinMnPout", "ele track inner p - outer p, mean of GSF components", nbinp, 0., 200.);
  h_ele_PinMnPout->Sumw2();
  h_ele_PinMnPout_mode =
      new TH1F("h_ele_PinMnPout_mode", "ele track inner p - outer p, mode of GSF components", nbinp, 0., 100.);
  h_ele_PinMnPout_mode->Sumw2();
  h_ele_PinMnPoutVsEta_mode = new TH2F("h_ele_PinMnPoutVsEta_mode",
                                       "ele track inner p - outer p vs eta, mode of GSF components",
                                       nbineta2D,
                                       etamin,
                                       etamax,
                                       nbinp2D,
                                       0.,
                                       100.);
  h_ele_PinMnPoutVsPhi_mode = new TH2F("h_ele_PinMnPoutVsPhi_mode",
                                       "ele track inner p - outer p vs phi, mode of GSF components",
                                       nbinphi2D,
                                       phimin,
                                       phimax,
                                       nbinp2D,
                                       0.,
                                       100.);
  h_ele_PinMnPoutVsPt_mode = new TH2F("h_ele_PinMnPoutVsPt_mode",
                                      "ele track inner p - outer p vs pt, mode of GSF components",
                                      nbinpt2D,
                                      0.,
                                      ptmax,
                                      nbinp2D,
                                      0.,
                                      100.);
  h_ele_PinMnPoutVsE_mode = new TH2F("h_ele_PinMnPoutVsE_mode",
                                     "ele track inner p - outer p vs E, mode of GSF components",
                                     nbinp2D,
                                     0.,
                                     200.,
                                     nbinp2D,
                                     0.,
                                     100.);
  h_ele_PinMnPoutVsChi2_mode = new TH2F("h_ele_PinMnPoutVsChi2_mode",
                                        "ele track inner p - outer p vs track chi2, mode of GSF components",
                                        50,
                                        0.,
                                        20.,
                                        nbinp2D,
                                        0.,
                                        100.);
  h_ele_outerP = new TH1F("h_ele_outerP", "ele track outer p, mean of GSF components", nbinp, 0., pmax);
  h_ele_outerP->Sumw2();
  h_ele_outerP_mode = new TH1F("h_ele_outerP_mode", "ele track outer p, mode of GSF components", nbinp, 0., pmax);
  h_ele_outerP_mode->Sumw2();
  h_ele_outerPVsEta_mode =
      new TH2F("h_ele_outerPVsEta_mode", "ele track outer p vs eta mode", nbineta2D, etamin, etamax, 50, 0., pmax);
  h_ele_outerPt = new TH1F("h_ele_outerPt", "ele track outer p_{T}, mean of GSF components", nbinpt, 0., ptmax);
  h_ele_outerPt->Sumw2();
  h_ele_outerPt_mode =
      new TH1F("h_ele_outerPt_mode", "ele track outer p_{T}, mode of GSF components", nbinpt, 0., ptmax);
  h_ele_outerPt_mode->Sumw2();
  h_ele_outerPtVsEta_mode = new TH2F("h_ele_outerPtVsEta_mode",
                                     "ele track outer p_{T} vs eta, mode of GSF components",
                                     nbineta2D,
                                     etamin,
                                     etamax,
                                     nbinpt2D,
                                     0.,
                                     ptmax);
  h_ele_outerPtVsPhi_mode = new TH2F("h_ele_outerPtVsPhi_mode",
                                     "ele track outer p_{T} vs phi, mode of GSF components",
                                     nbinphi2D,
                                     phimin,
                                     phimax,
                                     nbinpt2D,
                                     0.,
                                     ptmax);
  h_ele_outerPtVsPt_mode = new TH2F("h_ele_outerPtVsPt_mode",
                                    "ele track outer p_{T} vs pt, mode of GSF components",
                                    nbinpt2D,
                                    0.,
                                    100.,
                                    nbinpt2D,
                                    0.,
                                    ptmax);

  // matched electrons, matching
  h_ele_EoP = new TH1F("h_ele_EoP", "ele E/P_{vertex}", nbineop, 0., eopmax);
  h_ele_EoP->Sumw2();
  h_ele_EoP_eg = new TH1F("h_ele_EoP_eg", "ele E/P_{vertex}, ecal driven", nbineop, 0., eopmax);
  h_ele_EoP_eg->Sumw2();
  h_ele_EoP_barrel = new TH1F("h_ele_EoP_barrel", "ele E/P_{vertex} barrel", nbineop, 0., eopmax);
  h_ele_EoP_barrel->Sumw2();
  h_ele_EoP_eg_barrel = new TH1F("h_ele_EoP_eg_barrel", "ele E/P_{vertex}, ecal driven barrel", nbineop, 0., eopmax);
  h_ele_EoP_eg_barrel->Sumw2();
  h_ele_EoP_endcaps = new TH1F("h_ele_EoP_endcaps", "ele E/P_{vertex} endcaps", nbineop, 0., eopmax);
  h_ele_EoP_endcaps->Sumw2();
  h_ele_EoP_eg_endcaps = new TH1F("h_ele_EoP_eg_endcaps", "ele E/P_{vertex}, ecal driven endcaps", nbineop, 0., eopmax);
  h_ele_EoP_eg_endcaps->Sumw2();
  h_ele_EoPVsEta =
      new TH2F("h_ele_EoPVsEta", "ele E/P_{vertex} vs eta", nbineta2D, etamin, etamax, nbineop2D, 0., eopmaxsht);
  h_ele_EoPVsPhi =
      new TH2F("h_ele_EoPVsPhi", "ele E/P_{vertex} vs phi", nbinphi2D, phimin, phimax, nbineop2D, 0., eopmaxsht);
  h_ele_EoPVsE = new TH2F("h_ele_EoPVsE", "ele E/P_{vertex} vs E", 50, 0., pmax, 50, 0., 5.);
  h_ele_EseedOP = new TH1F("h_ele_EseedOP", "ele E_{seed}/P_{vertex}", nbineop, 0., eopmax);
  h_ele_EseedOP->Sumw2();
  h_ele_EseedOP_eg = new TH1F("h_ele_EseedOP_eg", "ele E_{seed}/P_{vertex}, ecal driven", nbineop, 0., eopmax);
  h_ele_EseedOP_eg->Sumw2();
  h_ele_EseedOP_barrel = new TH1F("h_ele_EseedOP_barrel", "ele E_{seed}/P_{vertex} barrel", nbineop, 0., eopmax);
  h_ele_EseedOP_barrel->Sumw2();
  h_ele_EseedOP_eg_barrel =
      new TH1F("h_ele_EseedOP_eg_barrel", "ele E_{seed}/P_{vertex}, ecal driven barrel", nbineop, 0., eopmax);
  h_ele_EseedOP_eg_barrel->Sumw2();
  h_ele_EseedOP_endcaps = new TH1F("h_ele_EseedOP_endcaps", "ele E_{seed}/P_{vertex} endcaps", nbineop, 0., eopmax);
  h_ele_EseedOP_endcaps->Sumw2();
  h_ele_EseedOP_eg_endcaps =
      new TH1F("h_ele_EseedOP_eg_endcaps", "ele E_{seed}/P_{vertex}, ecal driven, endcaps", nbineop, 0., eopmax);
  h_ele_EseedOP_eg_endcaps->Sumw2();
  h_ele_EseedOPVsEta = new TH2F(
      "h_ele_EseedOPVsEta", "ele E_{seed}/P_{vertex} vs eta", nbineta2D, etamin, etamax, nbineop2D, 0., eopmaxsht);
  h_ele_EseedOPVsPhi = new TH2F(
      "h_ele_EseedOPVsPhi", "ele E_{seed}/P_{vertex} vs phi", nbinphi2D, phimin, phimax, nbineop2D, 0., eopmaxsht);
  h_ele_EseedOPVsE = new TH2F("h_ele_EseedOPVsE", "ele E_{seed}/P_{vertex} vs E", 50, 0., pmax, 50, 0., 5.);
  h_ele_EoPout = new TH1F("h_ele_EoPout", "ele E_{seed}/P_{out}", nbineop, 0., eopmax);
  h_ele_EoPout->Sumw2();
  h_ele_EoPout_eg = new TH1F("h_ele_EoPout_eg", "ele E_{seed}/P_{out}, ecal driven", nbineop, 0., eopmax);
  h_ele_EoPout_eg->Sumw2();
  h_ele_EoPout_barrel = new TH1F("h_ele_EoPout_barrel", "ele E_{seed}/P_{out} barrel", nbineop, 0., eopmax);
  h_ele_EoPout_barrel->Sumw2();
  h_ele_EoPout_eg_barrel =
      new TH1F("h_ele_EoPout_eg_barrel", "ele E_{seed}/P_{out}, ecal driven, barrel", nbineop, 0., eopmax);
  h_ele_EoPout_eg_barrel->Sumw2();
  h_ele_EoPout_endcaps = new TH1F("h_ele_EoPout_endcaps", "ele E_{seed}/P_{out} endcaps", nbineop, 0., eopmax);
  h_ele_EoPout_endcaps->Sumw2();
  h_ele_EoPout_eg_endcaps =
      new TH1F("h_ele_EoPout_eg_endcaps", "ele E_{seed}/P_{out}, ecal driven, endcaps", nbineop, 0., eopmax);
  h_ele_EoPout_eg_endcaps->Sumw2();
  h_ele_EoPoutVsEta =
      new TH2F("h_ele_EoPoutVsEta", "ele E_{seed}/P_{out} vs eta", nbineta2D, etamin, etamax, nbineop2D, 0., eopmaxsht);
  h_ele_EoPoutVsPhi =
      new TH2F("h_ele_EoPoutVsPhi", "ele E_{seed}/P_{out} vs phi", nbinphi2D, phimin, phimax, nbineop2D, 0., eopmaxsht);
  h_ele_EoPoutVsE =
      new TH2F("h_ele_EoPoutVsE", "ele E_{seed}/P_{out} vs E", nbinp2D, 0., pmax, nbineop2D, 0., eopmaxsht);
  h_ele_EeleOPout = new TH1F("h_ele_EeleOPout", "ele E_{ele}/P_{out}", nbineop, 0., eopmax);
  h_ele_EeleOPout->Sumw2();
  h_ele_EeleOPout_eg = new TH1F("h_ele_EeleOPout_eg", "ele E_{ele}/P_{out}, ecal driven", nbineop, 0., eopmax);
  h_ele_EeleOPout_eg->Sumw2();
  h_ele_EeleOPout_barrel = new TH1F("h_ele_EeleOPout_barrel", "ele E_{ele}/P_{out} barrel", nbineop, 0., eopmax);
  h_ele_EeleOPout_barrel->Sumw2();
  h_ele_EeleOPout_eg_barrel =
      new TH1F("h_ele_EeleOPout_eg_barrel", "ele E_{ele}/P_{out}, ecal driven, barrel", nbineop, 0., eopmax);
  h_ele_EeleOPout_eg_barrel->Sumw2();
  h_ele_EeleOPout_endcaps = new TH1F("h_ele_EeleOPout_endcaps", "ele E_{ele}/P_{out} endcaps", nbineop, 0., eopmax);
  h_ele_EeleOPout_endcaps->Sumw2();
  h_ele_EeleOPout_eg_endcaps =
      new TH1F("h_ele_EeleOPout_eg_endcaps", "ele E_{ele}/P_{out}, ecal driven, endcaps", nbineop, 0., eopmax);
  h_ele_EeleOPout_eg_endcaps->Sumw2();
  h_ele_EeleOPoutVsEta = new TH2F(
      "h_ele_EeleOPoutVsEta", "ele E_{ele}/P_{out} vs eta", nbineta2D, etamin, etamax, nbineop2D, 0., eopmaxsht);
  h_ele_EeleOPoutVsPhi = new TH2F(
      "h_ele_EeleOPoutVsPhi", "ele E_{ele}/P_{out} vs phi", nbinphi2D, phimin, phimax, nbineop2D, 0., eopmaxsht);
  h_ele_EeleOPoutVsE =
      new TH2F("h_ele_EeleOPoutVsE", "ele E_{ele}/P_{out} vs E", nbinp2D, 0., pmax, nbineop2D, 0., eopmaxsht);
  h_ele_dEtaSc_propVtx = new TH1F(
      "h_ele_dEtaSc_propVtx", "ele #eta_{sc} - #eta_{tr}, prop from vertex", nbindetamatch, detamatchmin, detamatchmax);
  h_ele_dEtaSc_propVtx->Sumw2();
  h_ele_dEtaSc_propVtx_eg = new TH1F("h_ele_dEtaSc_propVtx_eg",
                                     "ele #eta_{sc} - #eta_{tr}, prop from vertex, ecal driven",
                                     nbindetamatch,
                                     detamatchmin,
                                     detamatchmax);
  h_ele_dEtaSc_propVtx_eg->Sumw2();
  h_ele_dEtaSc_propVtx_barrel = new TH1F("h_ele_dEtaSc_propVtx_barrel",
                                         "ele #eta_{sc} - #eta_{tr}, prop from vertex, barrel",
                                         nbindetamatch,
                                         detamatchmin,
                                         detamatchmax);
  h_ele_dEtaSc_propVtx_barrel->Sumw2();
  h_ele_dEtaSc_propVtx_eg_barrel = new TH1F("h_ele_dEtaSc_propVtx_eg_barrel",
                                            "ele #eta_{sc} - #eta_{tr}, prop from vertex, ecal driven, barrel",
                                            nbindetamatch,
                                            detamatchmin,
                                            detamatchmax);
  h_ele_dEtaSc_propVtx_eg_barrel->Sumw2();
  h_ele_dEtaSc_propVtx_endcaps = new TH1F("h_ele_dEtaSc_propVtx_endcaps",
                                          "ele #eta_{sc} - #eta_{tr}, prop from vertex, endcaps",
                                          nbindetamatch,
                                          detamatchmin,
                                          detamatchmax);
  h_ele_dEtaSc_propVtx_endcaps->Sumw2();
  h_ele_dEtaSc_propVtx_eg_endcaps = new TH1F("h_ele_dEtaSc_propVtx_eg_endcaps",
                                             "ele #eta_{sc} - #eta_{tr}, prop from vertex, ecal driven, endcaps",
                                             nbindetamatch,
                                             detamatchmin,
                                             detamatchmax);
  h_ele_dEtaSc_propVtx_eg_endcaps->Sumw2();
  h_ele_dEtaScVsEta_propVtx = new TH2F("h_ele_dEtaScVsEta_propVtx",
                                       "ele #eta_{sc} - #eta_{tr} vs eta, prop from vertex",
                                       nbineta2D,
                                       etamin,
                                       etamax,
                                       nbindetamatch2D,
                                       detamatchmin,
                                       detamatchmax);
  h_ele_dEtaScVsPhi_propVtx = new TH2F("h_ele_dEtaScVsPhi_propVtx",
                                       "ele #eta_{sc} - #eta_{tr} vs phi, prop from vertex",
                                       nbinphi2D,
                                       phimin,
                                       phimax,
                                       nbindetamatch2D,
                                       detamatchmin,
                                       detamatchmax);
  h_ele_dEtaScVsPt_propVtx = new TH2F("h_ele_dEtaScVsPt_propVtx",
                                      "ele #eta_{sc} - #eta_{tr} vs pt, prop from vertex",
                                      nbinpt2D,
                                      0.,
                                      ptmax,
                                      nbindetamatch2D,
                                      detamatchmin,
                                      detamatchmax);
  h_ele_dPhiSc_propVtx = new TH1F(
      "h_ele_dPhiSc_propVtx", "ele #phi_{sc} - #phi_{tr}, prop from vertex", nbindphimatch, dphimatchmin, dphimatchmax);
  h_ele_dPhiSc_propVtx->Sumw2();
  h_ele_dPhiSc_propVtx_eg = new TH1F("h_ele_dPhiSc_propVtx_eg",
                                     "ele #phi_{sc} - #phi_{tr}, prop from vertex, ecal driven",
                                     nbindphimatch,
                                     dphimatchmin,
                                     dphimatchmax);
  h_ele_dPhiSc_propVtx_eg->Sumw2();
  h_ele_dPhiSc_propVtx_barrel = new TH1F("h_ele_dPhiSc_propVtx_barrel",
                                         "ele #phi_{sc} - #phi_{tr}, prop from vertex, barrel",
                                         nbindphimatch,
                                         dphimatchmin,
                                         dphimatchmax);
  h_ele_dPhiSc_propVtx_barrel->Sumw2();
  h_ele_dPhiSc_propVtx_eg_barrel = new TH1F("h_ele_dPhiSc_propVtx_eg_barrel",
                                            "ele #phi_{sc} - #phi_{tr}, prop from vertex, ecal driven, barrel",
                                            nbindphimatch,
                                            dphimatchmin,
                                            dphimatchmax);
  h_ele_dPhiSc_propVtx_eg_barrel->Sumw2();
  h_ele_dPhiSc_propVtx_endcaps = new TH1F("h_ele_dPhiSc_propVtx_endcaps",
                                          "ele #phi_{sc} - #phi_{tr}, prop from vertex, endcaps",
                                          nbindphimatch,
                                          dphimatchmin,
                                          dphimatchmax);
  h_ele_dPhiSc_propVtx_endcaps->Sumw2();
  h_ele_dPhiSc_propVtx_eg_endcaps = new TH1F("h_ele_dPhiSc_propVtx_eg_endcaps",
                                             "ele #phi_{sc} - #phi_{tr}, prop from vertex, ecal driven, endcaps",
                                             nbindphimatch,
                                             dphimatchmin,
                                             dphimatchmax);
  h_ele_dPhiSc_propVtx_eg_endcaps->Sumw2();
  h_ele_dPhiScVsEta_propVtx = new TH2F("h_ele_dPhiScVsEta_propVtx",
                                       "ele #phi_{sc} - #phi_{tr} vs eta, prop from vertex",
                                       nbineta2D,
                                       etamin,
                                       etamax,
                                       nbindphimatch2D,
                                       dphimatchmin,
                                       dphimatchmax);
  h_ele_dPhiScVsPhi_propVtx = new TH2F("h_ele_dPhiScVsPhi_propVtx",
                                       "ele #phi_{sc} - #phi_{tr} vs phi, prop from vertex",
                                       nbinphi2D,
                                       phimin,
                                       phimax,
                                       nbindphimatch2D,
                                       dphimatchmin,
                                       dphimatchmax);
  h_ele_dPhiScVsPt_propVtx = new TH2F("h_ele_dPhiScVsPt_propVtx",
                                      "ele #phi_{sc} - #phi_{tr} vs pt, prop from vertex",
                                      nbinpt2D,
                                      0.,
                                      ptmax,
                                      nbindphimatch2D,
                                      dphimatchmin,
                                      dphimatchmax);
  h_ele_dEtaCl_propOut = new TH1F("h_ele_dEtaCl_propOut",
                                  "ele #eta_{cl} - #eta_{tr}, prop from outermost",
                                  nbindetamatch,
                                  detamatchmin,
                                  detamatchmax);
  h_ele_dEtaCl_propOut->Sumw2();
  h_ele_dEtaCl_propOut_eg = new TH1F("h_ele_dEtaCl_propOut_eg",
                                     "ele #eta_{cl} - #eta_{tr}, prop from outermost, ecal driven",
                                     nbindetamatch,
                                     detamatchmin,
                                     detamatchmax);
  h_ele_dEtaCl_propOut_eg->Sumw2();
  h_ele_dEtaCl_propOut_barrel = new TH1F("h_ele_dEtaCl_propOut_barrel",
                                         "ele #eta_{cl} - #eta_{tr}, prop from outermost, barrel",
                                         nbindetamatch,
                                         detamatchmin,
                                         detamatchmax);
  h_ele_dEtaCl_propOut_barrel->Sumw2();
  h_ele_dEtaCl_propOut_eg_barrel = new TH1F("h_ele_dEtaCl_propOut_eg_barrel",
                                            "ele #eta_{cl} - #eta_{tr}, prop from outermost, ecal driven, barrel",
                                            nbindetamatch,
                                            detamatchmin,
                                            detamatchmax);
  h_ele_dEtaCl_propOut_eg_barrel->Sumw2();
  h_ele_dEtaCl_propOut_endcaps = new TH1F("h_ele_dEtaCl_propOut_endcaps",
                                          "ele #eta_{cl} - #eta_{tr}, prop from outermost, endcaps",
                                          nbindetamatch,
                                          detamatchmin,
                                          detamatchmax);
  h_ele_dEtaCl_propOut_endcaps->Sumw2();
  h_ele_dEtaCl_propOut_eg_endcaps = new TH1F("h_ele_dEtaCl_propOut_eg_endcaps",
                                             "ele #eta_{cl} - #eta_{tr}, prop from outermost, ecal driven, endcaps",
                                             nbindetamatch,
                                             detamatchmin,
                                             detamatchmax);
  h_ele_dEtaCl_propOut_eg_endcaps->Sumw2();
  h_ele_dEtaClVsEta_propOut = new TH2F("h_ele_dEtaClVsEta_propOut",
                                       "ele #eta_{cl} - #eta_{tr} vs eta, prop from out",
                                       nbineta2D,
                                       etamin,
                                       etamax,
                                       nbindetamatch2D,
                                       detamatchmin,
                                       detamatchmax);
  h_ele_dEtaClVsPhi_propOut = new TH2F("h_ele_dEtaClVsPhi_propOut",
                                       "ele #eta_{cl} - #eta_{tr} vs phi, prop from out",
                                       nbinphi2D,
                                       phimin,
                                       phimax,
                                       nbindetamatch2D,
                                       detamatchmin,
                                       detamatchmax);
  h_ele_dEtaClVsPt_propOut = new TH2F("h_ele_dEtaScVsPt_propOut",
                                      "ele #eta_{cl} - #eta_{tr} vs pt, prop from out",
                                      nbinpt2D,
                                      0.,
                                      ptmax,
                                      nbindetamatch2D,
                                      detamatchmin,
                                      detamatchmax);
  h_ele_dPhiCl_propOut = new TH1F("h_ele_dPhiCl_propOut",
                                  "ele #phi_{cl} - #phi_{tr}, prop from outermost",
                                  nbindphimatch,
                                  dphimatchmin,
                                  dphimatchmax);
  h_ele_dPhiCl_propOut->Sumw2();
  h_ele_dPhiCl_propOut_eg = new TH1F("h_ele_dPhiCl_propOut_eg",
                                     "ele #phi_{cl} - #phi_{tr}, prop from outermost, ecal driven",
                                     nbindphimatch,
                                     dphimatchmin,
                                     dphimatchmax);
  h_ele_dPhiCl_propOut_eg->Sumw2();
  h_ele_dPhiCl_propOut_barrel = new TH1F("h_ele_dPhiCl_propOut_barrel",
                                         "ele #phi_{cl} - #phi_{tr}, prop from outermost, barrel",
                                         nbindphimatch,
                                         dphimatchmin,
                                         dphimatchmax);
  h_ele_dPhiCl_propOut_barrel->Sumw2();
  h_ele_dPhiCl_propOut_eg_barrel = new TH1F("h_ele_dPhiCl_propOut_eg_barrel",
                                            "ele #phi_{cl} - #phi_{tr}, prop from outermost, ecal driven, barrel",
                                            nbindphimatch,
                                            dphimatchmin,
                                            dphimatchmax);
  h_ele_dPhiCl_propOut_eg_barrel->Sumw2();
  h_ele_dPhiCl_propOut_endcaps = new TH1F("h_ele_dPhiCl_propOut_endcaps",
                                          "ele #phi_{cl} - #phi_{tr}, prop from outermost, endcaps",
                                          nbindphimatch,
                                          dphimatchmin,
                                          dphimatchmax);
  h_ele_dPhiCl_propOut_endcaps->Sumw2();
  h_ele_dPhiCl_propOut_eg_endcaps = new TH1F("h_ele_dPhiCl_propOut_eg_endcaps",
                                             "ele #phi_{cl} - #phi_{tr}, prop from outermost, ecal driven, endcaps",
                                             nbindphimatch,
                                             dphimatchmin,
                                             dphimatchmax);
  h_ele_dPhiCl_propOut_eg_endcaps->Sumw2();
  h_ele_dPhiClVsEta_propOut = new TH2F("h_ele_dPhiClVsEta_propOut",
                                       "ele #phi_{cl} - #phi_{tr} vs eta, prop from out",
                                       nbineta2D,
                                       etamin,
                                       etamax,
                                       nbindphimatch2D,
                                       dphimatchmin,
                                       dphimatchmax);
  h_ele_dPhiClVsPhi_propOut = new TH2F("h_ele_dPhiClVsPhi_propOut",
                                       "ele #phi_{cl} - #phi_{tr} vs phi, prop from out",
                                       nbinphi2D,
                                       phimin,
                                       phimax,
                                       nbindphimatch2D,
                                       dphimatchmin,
                                       dphimatchmax);
  h_ele_dPhiClVsPt_propOut = new TH2F("h_ele_dPhiSClsPt_propOut",
                                      "ele #phi_{cl} - #phi_{tr} vs pt, prop from out",
                                      nbinpt2D,
                                      0.,
                                      ptmax,
                                      nbindphimatch2D,
                                      dphimatchmin,
                                      dphimatchmax);
  h_ele_dEtaEleCl_propOut = new TH1F("h_ele_dEtaEleCl_propOut",
                                     "ele #eta_{EleCl} - #eta_{tr}, prop from outermost",
                                     nbindetamatch,
                                     detamatchmin,
                                     detamatchmax);
  h_ele_dEtaEleCl_propOut->Sumw2();
  h_ele_dEtaEleCl_propOut_eg = new TH1F("h_ele_dEtaEleCl_propOut_eg",
                                        "ele #eta_{EleCl} - #eta_{tr}, prop from outermost, ecal driven",
                                        nbindetamatch,
                                        detamatchmin,
                                        detamatchmax);
  h_ele_dEtaEleCl_propOut_eg->Sumw2();
  h_ele_dEtaEleCl_propOut_barrel = new TH1F("h_ele_dEtaEleCl_propOut_barrel",
                                            "ele #eta_{EleCl} - #eta_{tr}, prop from outermost, barrel",
                                            nbindetamatch,
                                            detamatchmin,
                                            detamatchmax);
  h_ele_dEtaEleCl_propOut_barrel->Sumw2();
  h_ele_dEtaEleCl_propOut_eg_barrel = new TH1F("h_ele_dEtaEleCl_propOut_eg_barrel",
                                               "ele #eta_{EleCl} - #eta_{tr}, prop from outermost, ecal driven, barrel",
                                               nbindetamatch,
                                               detamatchmin,
                                               detamatchmax);
  h_ele_dEtaEleCl_propOut_eg_barrel->Sumw2();
  h_ele_dEtaEleCl_propOut_endcaps = new TH1F("h_ele_dEtaEleCl_propOut_endcaps",
                                             "ele #eta_{EleCl} - #eta_{tr}, prop from outermost, endcaps",
                                             nbindetamatch,
                                             detamatchmin,
                                             detamatchmax);
  h_ele_dEtaEleCl_propOut_endcaps->Sumw2();
  h_ele_dEtaEleCl_propOut_eg_endcaps =
      new TH1F("h_ele_dEtaEleCl_propOut_eg_endcaps",
               "ele #eta_{EleCl} - #eta_{tr}, prop from outermost, ecal driven, endcaps",
               nbindetamatch,
               detamatchmin,
               detamatchmax);
  h_ele_dEtaEleCl_propOut_eg_endcaps->Sumw2();
  h_ele_dEtaEleClVsEta_propOut = new TH2F("h_ele_dEtaEleClVsEta_propOut",
                                          "ele #eta_{EleCl} - #eta_{tr} vs eta, prop from out",
                                          nbineta2D,
                                          etamin,
                                          etamax,
                                          nbindetamatch2D,
                                          detamatchmin,
                                          detamatchmax);
  h_ele_dEtaEleClVsPhi_propOut = new TH2F("h_ele_dEtaEleClVsPhi_propOut",
                                          "ele #eta_{EleCl} - #eta_{tr} vs phi, prop from out",
                                          nbinphi2D,
                                          phimin,
                                          phimax,
                                          nbindetamatch2D,
                                          detamatchmin,
                                          detamatchmax);
  h_ele_dEtaEleClVsPt_propOut = new TH2F("h_ele_dEtaScVsPt_propOut",
                                         "ele #eta_{EleCl} - #eta_{tr} vs pt, prop from out",
                                         nbinpt2D,
                                         0.,
                                         ptmax,
                                         nbindetamatch2D,
                                         detamatchmin,
                                         detamatchmax);
  h_ele_dPhiEleCl_propOut = new TH1F("h_ele_dPhiEleCl_propOut",
                                     "ele #phi_{EleCl} - #phi_{tr}, prop from outermost",
                                     nbindphimatch,
                                     dphimatchmin,
                                     dphimatchmax);
  h_ele_dPhiEleCl_propOut->Sumw2();
  h_ele_dPhiEleCl_propOut_eg = new TH1F("h_ele_dPhiEleCl_propOut_eg",
                                        "ele #phi_{EleCl} - #phi_{tr}, prop from outermost, ecal driven",
                                        nbindphimatch,
                                        dphimatchmin,
                                        dphimatchmax);
  h_ele_dPhiEleCl_propOut_eg->Sumw2();
  h_ele_dPhiEleCl_propOut_barrel = new TH1F("h_ele_dPhiEleCl_propOut_barrel",
                                            "ele #phi_{EleCl} - #phi_{tr}, prop from outermost, barrel",
                                            nbindphimatch,
                                            dphimatchmin,
                                            dphimatchmax);
  h_ele_dPhiEleCl_propOut_barrel->Sumw2();
  h_ele_dPhiEleCl_propOut_eg_barrel = new TH1F("h_ele_dPhiEleCl_propOut_eg_barrel",
                                               "ele #phi_{EleCl} - #phi_{tr}, prop from outermost, ecal driven, barrel",
                                               nbindphimatch,
                                               dphimatchmin,
                                               dphimatchmax);
  h_ele_dPhiEleCl_propOut_eg_barrel->Sumw2();
  h_ele_dPhiEleCl_propOut_endcaps = new TH1F("h_ele_dPhiEleCl_propOut_endcaps",
                                             "ele #phi_{EleCl} - #phi_{tr}, prop from outermost, endcaps",
                                             nbindphimatch,
                                             dphimatchmin,
                                             dphimatchmax);
  h_ele_dPhiEleCl_propOut_endcaps->Sumw2();
  h_ele_dPhiEleCl_propOut_eg_endcaps =
      new TH1F("h_ele_dPhiEleCl_propOut_eg_endcaps",
               "ele #phi_{EleCl} - #phi_{tr}, prop from outermost, ecal driven, endcaps",
               nbindphimatch,
               dphimatchmin,
               dphimatchmax);
  h_ele_dPhiEleCl_propOut_eg_endcaps->Sumw2();
  h_ele_dPhiEleClVsEta_propOut = new TH2F("h_ele_dPhiEleClVsEta_propOut",
                                          "ele #phi_{EleCl} - #phi_{tr} vs eta, prop from out",
                                          nbineta2D,
                                          etamin,
                                          etamax,
                                          nbindphimatch2D,
                                          dphimatchmin,
                                          dphimatchmax);
  h_ele_dPhiEleClVsPhi_propOut = new TH2F("h_ele_dPhiEleClVsPhi_propOut",
                                          "ele #phi_{EleCl} - #phi_{tr} vs phi, prop from out",
                                          nbinphi2D,
                                          phimin,
                                          phimax,
                                          nbindphimatch2D,
                                          dphimatchmin,
                                          dphimatchmax);
  h_ele_dPhiEleClVsPt_propOut = new TH2F("h_ele_dPhiSEleClsPt_propOut",
                                         "ele #phi_{EleCl} - #phi_{tr} vs pt, prop from out",
                                         nbinpt2D,
                                         0.,
                                         ptmax,
                                         nbindphimatch2D,
                                         dphimatchmin,
                                         dphimatchmax);

  h_ele_HoE = new TH1F("h_ele_HoE", "ele hadronic energy / em energy", nbinhoe, hoemin, hoemax);
  h_ele_HoE->Sumw2();
  h_ele_HoE_eg = new TH1F("h_ele_HoE_eg", "ele hadronic energy / em energy, ecal driven", nbinhoe, hoemin, hoemax);
  h_ele_HoE_eg->Sumw2();
  h_ele_HoE_barrel = new TH1F("h_ele_HoE_barrel", "ele hadronic energy / em energy, barrel", nbinhoe, hoemin, hoemax);
  h_ele_HoE_barrel->Sumw2();
  h_ele_HoE_eg_barrel =
      new TH1F("h_ele_HoE_eg_barrel", "ele hadronic energy / em energy, ecal driven, barrel", nbinhoe, hoemin, hoemax);
  h_ele_HoE_eg_barrel->Sumw2();
  h_ele_HoE_endcaps =
      new TH1F("h_ele_HoE_endcaps", "ele hadronic energy / em energy, endcaps", nbinhoe, hoemin, hoemax);
  h_ele_HoE_endcaps->Sumw2();
  h_ele_HoE_eg_endcaps = new TH1F(
      "h_ele_HoE_eg_endcaps", "ele hadronic energy / em energy, ecal driven, endcaps", nbinhoe, hoemin, hoemax);
  h_ele_HoE_eg_endcaps->Sumw2();
  h_ele_HoE_fiducial =
      new TH1F("h_ele_HoE_fiducial", "ele hadronic energy / em energy, fiducial region", nbinhoe, hoemin, hoemax);
  h_ele_HoE_fiducial->Sumw2();
  h_ele_HoEVsEta = new TH2F(
      "h_ele_HoEVsEta", "ele hadronic energy / em energy vs eta", nbineta, etamin, etamax, nbinhoe, hoemin, hoemax);
  h_ele_HoEVsPhi = new TH2F(
      "h_ele_HoEVsPhi", "ele hadronic energy / em energy vs phi", nbinphi2D, phimin, phimax, nbinhoe, hoemin, hoemax);
  h_ele_HoEVsE =
      new TH2F("h_ele_HoEVsE", "ele hadronic energy / em energy vs E", nbinp, 0., 300., nbinhoe, hoemin, hoemax);

  h_ele_seed_dphi2_ = new TH1F("h_ele_seedDphi2", "ele seed dphi 2nd layer", 50, -0.003, +0.003);
  h_ele_seed_dphi2_->Sumw2();
  h_ele_seed_dphi2VsEta_ =
      new TH2F("h_ele_seedDphi2VsEta", "ele seed dphi 2nd layer vs eta", nbineta2D, etamin, etamax, 50, -0.003, +0.003);
  h_ele_seed_dphi2VsPt_ =
      new TH2F("h_ele_seedDphi2VsPt", "ele seed dphi 2nd layer vs pt", nbinpt2D, 0., ptmax, 50, -0.003, +0.003);
  h_ele_seed_drz2_ = new TH1F("h_ele_seedDrz2", "ele seed dr (dz) 2nd layer", 50, -0.03, +0.03);
  h_ele_seed_drz2_->Sumw2();
  h_ele_seed_drz2VsEta_ =
      new TH2F("h_ele_seedDrz2VsEta", "ele seed dr/dz 2nd layer vs eta", nbineta2D, etamin, etamax, 50, -0.03, +0.03);
  h_ele_seed_drz2VsPt_ =
      new TH2F("h_ele_seedDrz2VsPt", "ele seed dr/dz 2nd layer vs pt", nbinpt2D, 0., ptmax, 50, -0.03, +0.03);
  h_ele_seed_subdet2_ = new TH1F("h_ele_seedSubdet2", "ele seed subdet 2nd layer", 10, 0., 10.);
  h_ele_seed_subdet2_->Sumw2();

  // classes
  h_ele_classes = new TH1F("h_ele_classes", "ele classes", 20, 0.0, 20.);
  h_ele_classes->Sumw2();
  h_ele_eta = new TH1F("h_ele_eta", "ele electron eta", nbineta / 2, 0.0, etamax);
  h_ele_eta->Sumw2();
  h_ele_eta_golden = new TH1F("h_ele_eta_golden", "ele electron eta golden", nbineta / 2, 0.0, etamax);
  h_ele_eta_golden->Sumw2();
  h_ele_eta_bbrem = new TH1F("h_ele_eta_bbrem", "ele electron eta bbrem", nbineta / 2, 0.0, etamax);
  h_ele_eta_bbrem->Sumw2();
  h_ele_eta_narrow = new TH1F("h_ele_eta_narrow", "ele electron eta narrow", nbineta / 2, 0.0, etamax);
  h_ele_eta_narrow->Sumw2();
  h_ele_eta_shower = new TH1F("h_ele_eta_show", "ele electron eta showering", nbineta / 2, 0.0, etamax);
  h_ele_eta_shower->Sumw2();
  h_ele_PinVsPoutGolden_mode = new TH2F("h_ele_PinVsPoutGolden_mode",
                                        "ele track inner p vs outer p vs eta, golden, mode of GSF components",
                                        nbinp2D,
                                        0.,
                                        pmax,
                                        50,
                                        0.,
                                        pmax);
  h_ele_PinVsPoutShowering_mode = new TH2F("h_ele_PinVsPoutShowering_mode",
                                           "ele track inner p vs outer p vs eta, showering, mode of GSF components",
                                           nbinp2D,
                                           0.,
                                           pmax,
                                           50,
                                           0.,
                                           pmax);
  h_ele_PinVsPoutGolden_mean = new TH2F("h_ele_PinVsPoutGolden_mean",
                                        "ele track inner p vs outer p vs eta, golden, mean of GSF components",
                                        nbinp2D,
                                        0.,
                                        pmax,
                                        50,
                                        0.,
                                        pmax);
  h_ele_PinVsPoutShowering_mean = new TH2F("h_ele_PinVsPoutShowering_mean",
                                           "ele track inner p vs outer p vs eta, showering, mean of GSF components",
                                           nbinp2D,
                                           0.,
                                           pmax,
                                           50,
                                           0.,
                                           pmax);
  h_ele_PtinVsPtoutGolden_mode = new TH2F("h_ele_PtinVsPtoutGolden_mode",
                                          "ele track inner pt vs outer pt vs eta, golden, mode of GSF components",
                                          nbinpt2D,
                                          0.,
                                          ptmax,
                                          50,
                                          0.,
                                          ptmax);
  h_ele_PtinVsPtoutShowering_mode = new TH2F("h_ele_PtinVsPtoutShowering_mode",
                                             "ele track inner pt vs outer pt vs eta, showering, mode of GSF components",
                                             nbinpt2D,
                                             0.,
                                             ptmax,
                                             50,
                                             0.,
                                             ptmax);
  h_ele_PtinVsPtoutGolden_mean = new TH2F("h_ele_PtinVsPtoutGolden_mean",
                                          "ele track inner pt vs outer pt vs eta, golden, mean of GSF components",
                                          nbinpt2D,
                                          0.,
                                          ptmax,
                                          50,
                                          0.,
                                          ptmax);
  h_ele_PtinVsPtoutShowering_mean = new TH2F("h_ele_PtinVsPtoutShowering_mean",
                                             "ele track inner pt vs outer pt vs eta, showering, mean of GSF components",
                                             nbinpt2D,
                                             0.,
                                             ptmax,
                                             50,
                                             0.,
                                             ptmax);
  histSclEoEtrueGolden_barrel = new TH1F("h_scl_EoEtrue_golden_barrel",
                                         "ele supercluster energy / gen energy, golden, barrel",
                                         nbinpoptrue,
                                         poptruemin,
                                         poptruemax);
  histSclEoEtrueGolden_barrel->Sumw2();
  histSclEoEtrueGolden_endcaps = new TH1F("h_scl_EoEtrue_golden_endcaps",
                                          "ele supercluster energy / gen energy, golden, endcaps",
                                          nbinpoptrue,
                                          poptruemin,
                                          poptruemax);
  histSclEoEtrueGolden_endcaps->Sumw2();
  histSclEoEtrueShowering_barrel = new TH1F("h_scl_EoEtrue_showering_barrel",
                                            "ele supercluster energy / gen energy, showering, barrel",
                                            nbinpoptrue,
                                            poptruemin,
                                            poptruemax);
  histSclEoEtrueShowering_barrel->Sumw2();
  histSclEoEtrueShowering_endcaps = new TH1F("h_scl_EoEtrue_showering_endcaps",
                                             "ele supercluster energy / gen energy, showering, endcaps",
                                             nbinpoptrue,
                                             poptruemin,
                                             poptruemax);
  histSclEoEtrueShowering_endcaps->Sumw2();

  // isolation
  h_ele_tkSumPt_dr03 = new TH1F("h_ele_tkSumPt_dr03", "tk isolation sum, dR=0.3", 100, 0.0, 20.);
  h_ele_tkSumPt_dr03->Sumw2();
  h_ele_ecalRecHitSumEt_dr03 = new TH1F("h_ele_ecalRecHitSumEt_dr03", "ecal isolation sum, dR=0.3", 100, 0.0, 20.);
  h_ele_ecalRecHitSumEt_dr03->Sumw2();
  h_ele_hcalDepth1TowerSumEt_dr03 =
      new TH1F("h_ele_hcalDepth1TowerSumEt_dr03", "hcal depth1 isolation sum, dR=0.3", 100, 0.0, 20.);
  h_ele_hcalDepth1TowerSumEt_dr03->Sumw2();
  h_ele_hcalDepth2TowerSumEt_dr03 =
      new TH1F("h_ele_hcalDepth2TowerSumEt_dr03", "hcal depth2 isolation sum, dR=0.3", 100, 0.0, 20.);
  h_ele_hcalDepth2TowerSumEt_dr03->Sumw2();
  h_ele_tkSumPt_dr04 = new TH1F("h_ele_tkSumPt_dr04", "tk isolation sum, dR=0.4", 100, 0.0, 20.);
  h_ele_tkSumPt_dr04->Sumw2();
  h_ele_ecalRecHitSumEt_dr04 = new TH1F("h_ele_ecalRecHitSumEt_dr04", "ecal isolation sum, dR=0.4", 100, 0.0, 20.);
  h_ele_ecalRecHitSumEt_dr04->Sumw2();
  h_ele_hcalDepth1TowerSumEt_dr04 =
      new TH1F("h_ele_hcalDepth1TowerSumEt_dr04", "hcal depth1 isolation sum, dR=0.4", 100, 0.0, 20.);
  h_ele_hcalDepth1TowerSumEt_dr04->Sumw2();
  h_ele_hcalDepth2TowerSumEt_dr04 =
      new TH1F("h_ele_hcalDepth2TowerSumEt_dr04", "hcal depth2 isolation sum, dR=0.4", 100, 0.0, 20.);
  h_ele_hcalDepth2TowerSumEt_dr04->Sumw2();

  // fbrem
  h_ele_fbrem = new TH1F("h_ele_fbrem", "ele brem fraction, mode of GSF components", 100, 0., 1.);
  h_ele_fbrem->Sumw2();
  h_ele_fbrem_eg = new TH1F("h_ele_fbrem_eg", "ele brem fraction, mode of GSF components, ecal driven", 100, 0., 1.);
  h_ele_fbrem_eg->Sumw2();
  h_ele_fbremVsEta_mode = new TProfile("h_ele_fbremvsEtamode",
                                       "mean ele brem fraction vs eta, mode of GSF components",
                                       nbineta2D,
                                       etamin,
                                       etamax,
                                       0.,
                                       1.);
  h_ele_fbremVsEta_mean = new TProfile("h_ele_fbremvsEtamean",
                                       "mean ele brem fraction vs eta, mean of GSF components",
                                       nbineta2D,
                                       etamin,
                                       etamax,
                                       0.,
                                       1.);

  // e/g et pflow electrons
  h_ele_mva = new TH1F("h_ele_mva", "ele identification mva", 100, -1., 1.);
  h_ele_mva->Sumw2();
  h_ele_mva_eg = new TH1F("h_ele_mva_eg", "ele identification mva, ecal driven", 100, -1., 1.);
  h_ele_mva_eg->Sumw2();
  h_ele_provenance = new TH1F("h_ele_provenance", "ele provenance", 5, -2., 3.);
  h_ele_provenance->Sumw2();

  // histos titles
  h_mcNum->GetXaxis()->SetTitle("N_{gen}");
  h_mcNum->GetYaxis()->SetTitle("Events");
  h_eleNum->GetXaxis()->SetTitle("# gen ele");
  h_eleNum->GetYaxis()->SetTitle("Events");
  h_gamNum->GetXaxis()->SetTitle("N_{gen #gamma}");
  h_gamNum->GetYaxis()->SetTitle("Events");
  h_simEta->GetXaxis()->SetTitle("#eta");
  h_simEta->GetYaxis()->SetTitle("Events");
  h_simP->GetXaxis()->SetTitle("p (GeV/c)");
  h_simP->GetYaxis()->SetTitle("Events");
  h_ele_foundHits->GetXaxis()->SetTitle("N_{hits}");
  h_ele_foundHits->GetYaxis()->SetTitle("Events");
  h_ele_foundHits_barrel->GetXaxis()->SetTitle("N_{hits}");
  h_ele_foundHits_barrel->GetYaxis()->SetTitle("Events");
  h_ele_foundHits_endcaps->GetXaxis()->SetTitle("N_{hits}");
  h_ele_foundHits_endcaps->GetYaxis()->SetTitle("Events");
  h_ele_ambiguousTracks->GetXaxis()->SetTitle("N_{ambiguous tracks}");
  h_ele_ambiguousTracks->GetYaxis()->SetTitle("Events");
  h_ele_lostHits->GetXaxis()->SetTitle("N_{lost hits}");
  h_ele_lostHits->GetYaxis()->SetTitle("Events");
  h_ele_lostHits_barrel->GetXaxis()->SetTitle("N_{lost hits}");
  h_ele_lostHits_barrel->GetYaxis()->SetTitle("Events");
  h_ele_lostHits_endcaps->GetXaxis()->SetTitle("N_{lost hits}");
  h_ele_lostHits_endcaps->GetYaxis()->SetTitle("Events");
  h_ele_chi2->GetXaxis()->SetTitle("#Chi^{2}");
  h_ele_chi2->GetYaxis()->SetTitle("Events");
  h_ele_chi2_barrel->GetXaxis()->SetTitle("#Chi^{2}");
  h_ele_chi2_barrel->GetYaxis()->SetTitle("Events");
  h_ele_chi2_endcaps->GetXaxis()->SetTitle("#Chi^{2}");
  h_ele_chi2_endcaps->GetYaxis()->SetTitle("Events");
  h_ele_charge->GetXaxis()->SetTitle("charge");
  h_ele_charge->GetYaxis()->SetTitle("Events");
  h_ele_vertexP->GetXaxis()->SetTitle("p_{vertex} (GeV/c)");
  h_ele_vertexP->GetYaxis()->SetTitle("Events");
  h_ele_vertexPt->GetXaxis()->SetTitle("p_{T vertex} (GeV/c)");
  h_ele_vertexPt->GetYaxis()->SetTitle("Events");
  h_ele_Et->GetXaxis()->SetTitle("E_{T} (GeV)");
  h_ele_Et->GetYaxis()->SetTitle("Events");
  h_ele_Et_all->GetXaxis()->SetTitle("E_{T} (GeV)");
  h_ele_Et_all->GetYaxis()->SetTitle("Events");
  h_ele_vertexEta->GetXaxis()->SetTitle("#eta");
  h_ele_vertexEta->GetYaxis()->SetTitle("Events");
  h_ele_vertexPhi->GetXaxis()->SetTitle("#phi (rad)");
  h_ele_vertexPhi->GetYaxis()->SetTitle("Events");
  h_ele_PoPtrue->GetXaxis()->SetTitle("P/P_{gen}");
  h_ele_PoPtrue->GetYaxis()->SetTitle("Events");
  h_ele_PoPtrue_barrel->GetXaxis()->SetTitle("P/P_{gen}");
  h_ele_PoPtrue_barrel->GetYaxis()->SetTitle("Events");
  h_ele_PoPtrue_endcaps->GetXaxis()->SetTitle("P/P_{gen}");
  h_ele_PoPtrue_endcaps->GetYaxis()->SetTitle("Events");
  h_ele_PoPtrue_golden_barrel->GetXaxis()->SetTitle("P/P_{gen}");
  h_ele_PoPtrue_golden_barrel->GetYaxis()->SetTitle("Events");
  h_ele_PoPtrue_showering_barrel->GetXaxis()->SetTitle("P/P_{gen}");
  h_ele_PoPtrue_showering_barrel->GetYaxis()->SetTitle("Events");
  h_ele_PoPtrue_golden_endcaps->GetXaxis()->SetTitle("P/P_{gen}");
  h_ele_PoPtrue_golden_endcaps->GetYaxis()->SetTitle("Events");
  h_ele_PoPtrue_showering_endcaps->GetXaxis()->SetTitle("P/P_{gen}");
  h_ele_PoPtrue_showering_endcaps->GetYaxis()->SetTitle("Events");
  h_ele_PtoPttrue->GetXaxis()->SetTitle("P_{T}/P_{T}^{gen}");
  h_ele_PtoPttrue->GetYaxis()->SetTitle("Events");
  h_ele_PtoPttrue_barrel->GetXaxis()->SetTitle("P_{T}/P_{T}^{gen}");
  h_ele_PtoPttrue_barrel->GetYaxis()->SetTitle("Events");
  h_ele_PtoPttrue_endcaps->GetXaxis()->SetTitle("P_{T}/P_{T}^{gen}");
  h_ele_PtoPttrue_endcaps->GetYaxis()->SetTitle("Events");
  histSclEoEtrue_barrel->GetXaxis()->SetTitle("E/E_{gen}");
  histSclEoEtrue_barrel->GetYaxis()->SetTitle("Events");
  histSclEoEtrue_endcaps->GetXaxis()->SetTitle("E/E_{gen}");
  histSclEoEtrue_endcaps->GetYaxis()->SetTitle("Events");
  histSclEoEtrueGolden_barrel->GetXaxis()->SetTitle("E/E_{gen}");
  histSclEoEtrueGolden_barrel->GetYaxis()->SetTitle("Events");
  histSclEoEtrueShowering_barrel->GetXaxis()->SetTitle("E/E_{gen}");
  histSclEoEtrueShowering_barrel->GetYaxis()->SetTitle("Events");
  histSclEoEtrueGolden_endcaps->GetXaxis()->SetTitle("E/E_{gen}");
  histSclEoEtrueGolden_endcaps->GetYaxis()->SetTitle("Events");
  histSclEoEtrueShowering_endcaps->GetXaxis()->SetTitle("E/E_{gen}");
  histSclEoEtrueShowering_endcaps->GetYaxis()->SetTitle("Events");
  histSclEoEtrue_barrel_etagap->GetXaxis()->SetTitle("E/E_{gen}");
  histSclEoEtrue_barrel_etagap->GetYaxis()->SetTitle("Events");
  histSclEoEtrue_barrel_phigap->GetXaxis()->SetTitle("E/E_{gen}");
  histSclEoEtrue_barrel_phigap->GetYaxis()->SetTitle("Events");
  histSclEoEtrue_ebeegap->GetXaxis()->SetTitle("E/E_{gen}");
  histSclEoEtrue_ebeegap->GetYaxis()->SetTitle("Events");
  histSclEoEtrue_endcaps_deegap->GetXaxis()->SetTitle("E/E_{gen}");
  histSclEoEtrue_endcaps_deegap->GetYaxis()->SetTitle("Events");
  histSclEoEtrue_endcaps_ringgap->GetXaxis()->SetTitle("E/E_{gen}");
  histSclEoEtrue_endcaps_ringgap->GetYaxis()->SetTitle("Events");
  histSclEoEtrue_barrel_new->GetXaxis()->SetTitle("E/E_{gen}");
  histSclEoEtrue_barrel_new->GetYaxis()->SetTitle("Events");
  histSclEoEtrue_endcaps_new->GetXaxis()->SetTitle("E/E_{gen}");
  histSclEoEtrue_endcaps_new->GetYaxis()->SetTitle("Events");
  histSclEoEtrue_barrel_etagap_new->GetXaxis()->SetTitle("E/E_{gen}");
  histSclEoEtrue_barrel_etagap_new->GetYaxis()->SetTitle("Events");
  histSclEoEtrue_barrel_phigap_new->GetXaxis()->SetTitle("E/E_{gen}");
  histSclEoEtrue_barrel_phigap_new->GetYaxis()->SetTitle("Events");
  histSclEoEtrue_ebeegap_new->GetXaxis()->SetTitle("E/E_{gen}");
  histSclEoEtrue_ebeegap_new->GetYaxis()->SetTitle("Events");
  histSclEoEtrue_endcaps_deegap_new->GetXaxis()->SetTitle("E/E_{gen}");
  histSclEoEtrue_endcaps_deegap_new->GetYaxis()->SetTitle("Events");
  histSclEoEtrue_endcaps_ringgap_new->GetXaxis()->SetTitle("E/E_{gen}");
  histSclEoEtrue_endcaps_ringgap_new->GetYaxis()->SetTitle("Events");
  histSclSigEtaEta_->GetXaxis()->SetTitle("#sigma_{#eta #eta}");
  histSclSigEtaEta_->GetYaxis()->SetTitle("Events");
  histSclSigEtaEta_barrel_->GetXaxis()->SetTitle("#sigma_{#eta #eta}");
  histSclSigEtaEta_barrel_->GetYaxis()->SetTitle("Events");
  histSclSigEtaEta_endcaps_->GetXaxis()->SetTitle("#sigma_{#eta #eta}");
  histSclSigEtaEta_endcaps_->GetYaxis()->SetTitle("Events");
  histSclSigIEtaIEta_->GetXaxis()->SetTitle("#sigma_{i#eta i#eta}");
  histSclSigIEtaIEta_->GetYaxis()->SetTitle("Events");
  histSclSigIEtaIEta_barrel_->GetXaxis()->SetTitle("#sigma_{i#eta i#eta}");
  histSclSigIEtaIEta_barrel_->GetYaxis()->SetTitle("Events");
  histSclSigIEtaIEta_endcaps_->GetXaxis()->SetTitle("#sigma_{i#eta i#eta}");
  histSclSigIEtaIEta_endcaps_->GetYaxis()->SetTitle("Events");
  histSclE1x5_->GetXaxis()->SetTitle("E1x5 (GeV)");
  histSclE1x5_->GetYaxis()->SetTitle("Events");
  histSclE1x5_barrel_->GetXaxis()->SetTitle("E1x5 (GeV)");
  histSclE1x5_barrel_->GetYaxis()->SetTitle("Events");
  histSclE1x5_endcaps_->GetXaxis()->SetTitle("E1x5 (GeV)");
  histSclE1x5_endcaps_->GetYaxis()->SetTitle("Events");
  histSclE2x5max_->GetXaxis()->SetTitle("E2x5 (GeV)");
  histSclE2x5max_->GetYaxis()->SetTitle("Events");
  histSclE2x5max_barrel_->GetXaxis()->SetTitle("E2x5 (GeV)");
  histSclE2x5max_barrel_->GetYaxis()->SetTitle("Events");
  histSclE2x5max_endcaps_->GetXaxis()->SetTitle("E2x5 (GeV)");
  histSclE2x5max_endcaps_->GetYaxis()->SetTitle("Events");
  histSclE5x5_->GetXaxis()->SetTitle("E5x5 (GeV)");
  histSclE5x5_->GetYaxis()->SetTitle("Events");
  histSclE5x5_barrel_->GetXaxis()->SetTitle("E5x5 (GeV)");
  histSclE5x5_barrel_->GetYaxis()->SetTitle("Events");
  histSclE5x5_endcaps_->GetXaxis()->SetTitle("E5x5 (GeV)");
  histSclE5x5_endcaps_->GetYaxis()->SetTitle("Events");
  histSclEoEtruePfVsEg->GetXaxis()->SetTitle("E/E_{gen} (e/g)");
  histSclEoEtruePfVsEg->GetYaxis()->SetTitle("E/E_{gen} (pflow)");
  h_ele_ChargeMnChargeTrue->GetXaxis()->SetTitle("q_{rec} - q_{gen}");
  h_ele_ChargeMnChargeTrue->GetYaxis()->SetTitle("Events");
  h_ele_EtaMnEtaTrue->GetXaxis()->SetTitle("#eta_{rec} - #eta_{gen}");
  h_ele_EtaMnEtaTrue->GetYaxis()->SetTitle("Events");
  h_ele_EtaMnEtaTrue_barrel->GetXaxis()->SetTitle("#eta_{rec} - #eta_{gen}");
  h_ele_EtaMnEtaTrue_barrel->GetYaxis()->SetTitle("Events");
  h_ele_EtaMnEtaTrue_endcaps->GetXaxis()->SetTitle("#eta_{rec} - #eta_{gen}");
  h_ele_EtaMnEtaTrue_endcaps->GetYaxis()->SetTitle("Events");
  h_ele_PhiMnPhiTrue->GetXaxis()->SetTitle("#phi_{rec} - #phi_{gen} (rad)");
  h_ele_PhiMnPhiTrue->GetYaxis()->SetTitle("Events");
  h_ele_PhiMnPhiTrue_barrel->GetXaxis()->SetTitle("#phi_{rec} - #phi_{gen} (rad)");
  h_ele_PhiMnPhiTrue_barrel->GetYaxis()->SetTitle("Events");
  h_ele_PhiMnPhiTrue_endcaps->GetXaxis()->SetTitle("#phi_{rec} - #phi_{gen} (rad)");
  h_ele_PhiMnPhiTrue_endcaps->GetYaxis()->SetTitle("Events");
  h_ele_PinMnPout->GetXaxis()->SetTitle("P_{vertex} - P_{out} (GeV/c)");
  h_ele_PinMnPout->GetYaxis()->SetTitle("Events");
  h_ele_PinMnPout_mode->GetXaxis()->SetTitle("P_{vertex} - P_{out}, mode of GSF components (GeV/c)");
  h_ele_PinMnPout_mode->GetYaxis()->SetTitle("Events");
  h_ele_outerP->GetXaxis()->SetTitle("P_{out} (GeV/c)");
  h_ele_outerP->GetYaxis()->SetTitle("Events");
  h_ele_outerP_mode->GetXaxis()->SetTitle("P_{out} (GeV/c)");
  h_ele_outerP_mode->GetYaxis()->SetTitle("Events");
  h_ele_outerPt->GetXaxis()->SetTitle("P_{T out} (GeV/c)");
  h_ele_outerPt->GetYaxis()->SetTitle("Events");
  h_ele_outerPt_mode->GetXaxis()->SetTitle("P_{T out} (GeV/c)");
  h_ele_outerPt_mode->GetYaxis()->SetTitle("Events");
  h_ele_EoP->GetXaxis()->SetTitle("E/P_{vertex}");
  h_ele_EoP->GetYaxis()->SetTitle("Events");
  h_ele_EseedOP->GetXaxis()->SetTitle("E_{seed}/P_{vertex}");
  h_ele_EseedOP->GetYaxis()->SetTitle("Events");
  h_ele_EoPout->GetXaxis()->SetTitle("E_{seed}/P_{out}");
  h_ele_EoPout->GetYaxis()->SetTitle("Events");
  h_ele_EeleOPout->GetXaxis()->SetTitle("E_{ele}/P_{out}");
  h_ele_EeleOPout->GetYaxis()->SetTitle("Events");
  h_ele_EoP_barrel->GetXaxis()->SetTitle("E/P_{vertex}");
  h_ele_EoP_barrel->GetYaxis()->SetTitle("Events");
  h_ele_EseedOP_barrel->GetXaxis()->SetTitle("E_{seed}/P_{vertex}");
  h_ele_EseedOP_barrel->GetYaxis()->SetTitle("Events");
  h_ele_EoPout_barrel->GetXaxis()->SetTitle("E_{seed}/P_{out}");
  h_ele_EoPout_barrel->GetYaxis()->SetTitle("Events");
  h_ele_EeleOPout_barrel->GetXaxis()->SetTitle("E_{ele}/P_{out}");
  h_ele_EeleOPout_barrel->GetYaxis()->SetTitle("Events");
  h_ele_EoP_endcaps->GetXaxis()->SetTitle("E/P_{vertex}");
  h_ele_EoP_endcaps->GetYaxis()->SetTitle("Events");
  h_ele_EseedOP_endcaps->GetXaxis()->SetTitle("E_{seed}/P_{vertex}");
  h_ele_EseedOP_endcaps->GetYaxis()->SetTitle("Events");
  h_ele_EoPout_endcaps->GetXaxis()->SetTitle("E_{seed}/P_{out}");
  h_ele_EoPout_endcaps->GetYaxis()->SetTitle("Events");
  h_ele_EeleOPout_endcaps->GetXaxis()->SetTitle("E_{ele}/P_{out}");
  h_ele_EeleOPout_endcaps->GetYaxis()->SetTitle("Events");
  h_ele_vertexX->GetXaxis()->SetTitle("x (cm)");
  h_ele_vertexX->GetYaxis()->SetTitle("Events");
  h_ele_vertexY->GetXaxis()->SetTitle("y (cm)");
  h_ele_vertexY->GetYaxis()->SetTitle("Events");
  h_ele_vertexZ->GetXaxis()->SetTitle("z (cm)");
  h_ele_vertexZ->GetYaxis()->SetTitle("Events");
  h_ele_vertexTIP->GetXaxis()->SetTitle("TIP (cm)");
  h_ele_vertexTIP->GetYaxis()->SetTitle("Events");
  h_ele_TIP_all->GetXaxis()->SetTitle("r_{T} (cm)");
  h_ele_TIP_all->GetYaxis()->SetTitle("Events");
  h_ele_vertexTIPVsEta->GetYaxis()->SetTitle("TIP (cm)");
  h_ele_vertexTIPVsEta->GetXaxis()->SetTitle("#eta");
  h_ele_vertexTIPVsPhi->GetYaxis()->SetTitle("TIP (cm)");
  h_ele_vertexTIPVsPhi->GetXaxis()->SetTitle("#phi (rad)");
  h_ele_vertexTIPVsPt->GetYaxis()->SetTitle("TIP (cm)");
  h_ele_vertexTIPVsPt->GetXaxis()->SetTitle("p_{T} (GeV/c)");
  h_ele_dEtaSc_propVtx->GetXaxis()->SetTitle("#eta_{sc} - #eta_{tr}");
  h_ele_dEtaSc_propVtx->GetYaxis()->SetTitle("Events");
  h_ele_dEtaCl_propOut->GetXaxis()->SetTitle("#eta_{seedcl} - #eta_{tr}");
  h_ele_dEtaCl_propOut->GetYaxis()->SetTitle("Events");
  h_ele_dEtaEleCl_propOut->GetXaxis()->SetTitle("#eta_{elecl} - #eta_{tr}");
  h_ele_dEtaEleCl_propOut->GetYaxis()->SetTitle("Events");
  h_ele_dPhiSc_propVtx->GetXaxis()->SetTitle("#phi_{sc} - #phi_{tr} (rad)");
  h_ele_dPhiSc_propVtx->GetYaxis()->SetTitle("Events");
  h_ele_dPhiCl_propOut->GetXaxis()->SetTitle("#phi_{seedcl} - #phi_{tr} (rad)");
  h_ele_dPhiCl_propOut->GetYaxis()->SetTitle("Events");
  h_ele_dPhiEleCl_propOut->GetXaxis()->SetTitle("#phi_{elecl} - #phi_{tr} (rad)");
  h_ele_dPhiEleCl_propOut->GetYaxis()->SetTitle("Events");
  h_ele_dEtaSc_propVtx_barrel->GetXaxis()->SetTitle("#eta_{sc} - #eta_{tr}");
  h_ele_dEtaSc_propVtx_barrel->GetYaxis()->SetTitle("Events");
  h_ele_dEtaCl_propOut_barrel->GetXaxis()->SetTitle("#eta_{seedcl} - #eta_{tr}");
  h_ele_dEtaCl_propOut_barrel->GetYaxis()->SetTitle("Events");
  h_ele_dEtaEleCl_propOut_barrel->GetXaxis()->SetTitle("#eta_{elecl} - #eta_{tr}");
  h_ele_dEtaEleCl_propOut_barrel->GetYaxis()->SetTitle("Events");
  h_ele_dPhiSc_propVtx_barrel->GetXaxis()->SetTitle("#phi_{sc} - #phi_{tr} (rad)");
  h_ele_dPhiSc_propVtx_barrel->GetYaxis()->SetTitle("Events");
  h_ele_dPhiCl_propOut_barrel->GetXaxis()->SetTitle("#phi_{seedcl} - #phi_{tr} (rad)");
  h_ele_dPhiCl_propOut_barrel->GetYaxis()->SetTitle("Events");
  h_ele_dPhiEleCl_propOut_barrel->GetXaxis()->SetTitle("#phi_{elecl} - #phi_{tr} (rad)");
  h_ele_dPhiEleCl_propOut_barrel->GetYaxis()->SetTitle("Events");
  h_ele_dEtaSc_propVtx_endcaps->GetXaxis()->SetTitle("#eta_{sc} - #eta_{tr}");
  h_ele_dEtaSc_propVtx_endcaps->GetYaxis()->SetTitle("Events");
  h_ele_dEtaCl_propOut_endcaps->GetXaxis()->SetTitle("#eta_{seedcl} - #eta_{tr}");
  h_ele_dEtaCl_propOut_endcaps->GetYaxis()->SetTitle("Events");
  h_ele_dEtaEleCl_propOut_endcaps->GetXaxis()->SetTitle("#eta_{elecl} - #eta_{tr}");
  h_ele_dEtaEleCl_propOut_endcaps->GetYaxis()->SetTitle("Events");
  h_ele_dPhiSc_propVtx_endcaps->GetXaxis()->SetTitle("#phi_{sc} - #phi_{tr} (rad)");
  h_ele_dPhiSc_propVtx_endcaps->GetYaxis()->SetTitle("Events");
  h_ele_dPhiCl_propOut_endcaps->GetXaxis()->SetTitle("#phi_{seedcl} - #phi_{tr} (rad)");
  h_ele_dPhiCl_propOut_endcaps->GetYaxis()->SetTitle("Events");
  h_ele_dPhiEleCl_propOut_endcaps->GetXaxis()->SetTitle("#phi_{elecl} - #phi_{tr} (rad)");
  h_ele_dPhiEleCl_propOut_endcaps->GetYaxis()->SetTitle("Events");
  h_ele_HoE->GetXaxis()->SetTitle("H/E");
  h_ele_HoE->GetYaxis()->SetTitle("Events");
  h_ele_HoE_barrel->GetXaxis()->SetTitle("H/E");
  h_ele_HoE_barrel->GetYaxis()->SetTitle("Events");
  h_ele_HoE_endcaps->GetXaxis()->SetTitle("H/E");
  h_ele_HoE_endcaps->GetYaxis()->SetTitle("Events");
  h_ele_HoE_fiducial->GetXaxis()->SetTitle("H/E");
  h_ele_HoE_fiducial->GetYaxis()->SetTitle("Events");
  h_ele_fbrem->GetXaxis()->SetTitle("P_{in} - P_{out} / P_{in}");
  h_ele_fbrem->GetYaxis()->SetTitle("Events");
  h_ele_seed_dphi2_->GetXaxis()->SetTitle("#phi_{hit}-#phi_{pred} (rad)");
  h_ele_seed_dphi2_->GetYaxis()->SetTitle("Events");
  h_ele_seed_drz2_->GetXaxis()->SetTitle("r(z)_{hit}-r(z)_{pred} (cm)");
  h_ele_seed_drz2_->GetYaxis()->SetTitle("Events");
  h_ele_seed_subdet2_->GetXaxis()->SetTitle("2nd hit subdet Id");
  h_ele_seed_subdet2_->GetYaxis()->SetTitle("Events");
  h_ele_classes->GetXaxis()->SetTitle("class Id");
  h_ele_classes->GetYaxis()->SetTitle("Events");
  h_ele_EoverP_all->GetXaxis()->SetTitle("E/P_{vertex}");
  h_ele_EoverP_all->GetYaxis()->SetTitle("Events");
  h_ele_EseedOP_all->GetXaxis()->SetTitle("E_{seed}/P_{vertex}");
  h_ele_EseedOP_all->GetYaxis()->SetTitle("Events");
  h_ele_EoPout_all->GetXaxis()->SetTitle("E_{seed}/P_{out}");
  h_ele_EoPout_all->GetYaxis()->SetTitle("Events");
  h_ele_EeleOPout_all->GetXaxis()->SetTitle("E_{ele}/P_{out}");
  h_ele_EeleOPout_all->GetYaxis()->SetTitle("Events");
  h_ele_dEtaSc_propVtx_all->GetXaxis()->SetTitle("#eta_{sc} - #eta_{tr}");
  h_ele_dEtaSc_propVtx_all->GetYaxis()->SetTitle("Events");
  h_ele_dPhiSc_propVtx_all->GetXaxis()->SetTitle("#phi_{sc} - #phi_{tr} (rad)");
  h_ele_dPhiSc_propVtx_all->GetYaxis()->SetTitle("Events");
  h_ele_dEtaCl_propOut_all->GetXaxis()->SetTitle("#eta_{sc} - #eta_{tr}");
  h_ele_dEtaCl_propOut_all->GetYaxis()->SetTitle("Events");
  h_ele_dPhiCl_propOut_all->GetXaxis()->SetTitle("#phi_{sc} - #phi_{tr} (rad)");
  h_ele_dPhiCl_propOut_all->GetYaxis()->SetTitle("Events");
  h_ele_HoE_all->GetXaxis()->SetTitle("H/E");
  h_ele_HoE_all->GetYaxis()->SetTitle("Events");
  h_ele_mee_all->GetXaxis()->SetTitle("m_{ee} (GeV/c^{2})");
  h_ele_mee_all->GetYaxis()->SetTitle("Events");
  h_ele_mee_os->GetXaxis()->SetTitle("m_{e^{+}e^{-}} (GeV/c^{2})");
  h_ele_mee_os->GetYaxis()->SetTitle("Events");
  h_ele_mee_os_ebeb->GetXaxis()->SetTitle("m_{e^{+}e^{-}} (GeV/c^{2})");
  h_ele_mee_os_ebeb->GetYaxis()->SetTitle("Events");
  h_ele_mee_os_ebee->GetXaxis()->SetTitle("m_{e^{+}e^{-}} (GeV/c^{2})");
  h_ele_mee_os_ebee->GetYaxis()->SetTitle("Events");
  h_ele_mee_os_eeee->GetXaxis()->SetTitle("m_{e^{+}e^{-}} (GeV/c^{2})");
  h_ele_mee_os_eeee->GetYaxis()->SetTitle("Events");
  h_ele_mee_os_gg->GetXaxis()->SetTitle("m_{e^{+}e^{-}} (GeV/c^{2})");
  h_ele_mee_os_gg->GetYaxis()->SetTitle("Events");
  h_ele_mee_os_gb->GetXaxis()->SetTitle("m_{e^{+}e^{-}} (GeV/c^{2})");
  h_ele_mee_os_gb->GetYaxis()->SetTitle("Events");
  h_ele_mee_os_bb->GetXaxis()->SetTitle("m_{e^{+}e^{-}} (GeV/c^{2})");
  h_ele_mee_os_bb->GetYaxis()->SetTitle("Events");
  h_ele_E2mnE1vsMee_all->GetXaxis()->SetTitle("m_{e^{+}e^{-}} (GeV/c^{2})");
  h_ele_E2mnE1vsMee_all->GetYaxis()->SetTitle("E2 - E1 (GeV)");
  h_ele_E2mnE1vsMee_egeg_all->GetXaxis()->SetTitle("m_{e^{+}e^{-}} (GeV/c^{2})");
  h_ele_E2mnE1vsMee_egeg_all->GetYaxis()->SetTitle("E2 - E1 (GeV)");
  histNum_->GetXaxis()->SetTitle("N_{ele}");
  histNum_->GetYaxis()->SetTitle("Events");
  h_ele_fbremVsEta_mode->GetXaxis()->SetTitle("#eta");
  h_ele_fbremVsEta_mean->GetXaxis()->SetTitle("#eta");
}

void GsfElectronMCAnalyzer::endJob() {
  histfile_->cd();

  std::cout << "[GsfElectronMCAnalyzer] efficiency calculation " << std::endl;
  // efficiency vs eta
  TH1F *h_ele_etaEff = (TH1F *)h_ele_simEta_matched->Clone("h_ele_etaEff");
  h_ele_etaEff->Reset();
  h_ele_etaEff->Divide(h_ele_simEta_matched, h_simEta, 1, 1, "b");
  h_ele_etaEff->Print();
  h_ele_etaEff->GetXaxis()->SetTitle("#eta");
  h_ele_etaEff->GetYaxis()->SetTitle("Efficiency");

  // efficiency vs z
  TH1F *h_ele_zEff = (TH1F *)h_ele_simZ_matched->Clone("h_ele_zEff");
  h_ele_zEff->Reset();
  h_ele_zEff->Divide(h_ele_simZ_matched, h_simZ, 1, 1, "b");
  h_ele_zEff->Print();
  h_ele_zEff->GetXaxis()->SetTitle("z (cm)");
  h_ele_zEff->GetYaxis()->SetTitle("Efficiency");

  // efficiency vs |eta|
  TH1F *h_ele_absetaEff = (TH1F *)h_ele_simAbsEta_matched->Clone("h_ele_absetaEff");
  h_ele_absetaEff->Reset();
  h_ele_absetaEff->Divide(h_ele_simAbsEta_matched, h_simAbsEta, 1, 1, "b");
  h_ele_absetaEff->GetXaxis()->SetTitle("|#eta|");
  h_ele_absetaEff->GetYaxis()->SetTitle("Efficiency");

  // efficiency vs pt
  TH1F *h_ele_ptEff = (TH1F *)h_ele_simPt_matched->Clone("h_ele_ptEff");
  h_ele_ptEff->Reset();
  h_ele_ptEff->Divide(h_ele_simPt_matched, h_simPt, 1, 1, "b");
  h_ele_ptEff->GetXaxis()->SetTitle("p_{T} (GeV/c)");
  h_ele_ptEff->GetYaxis()->SetTitle("Efficiency");

  // efficiency vs phi
  TH1F *h_ele_phiEff = (TH1F *)h_ele_simPhi_matched->Clone("h_ele_phiEff");
  h_ele_phiEff->Reset();
  h_ele_phiEff->Divide(h_ele_simPhi_matched, h_simPhi, 1, 1, "b");
  h_ele_phiEff->GetXaxis()->SetTitle("#phi (rad)");
  h_ele_phiEff->GetYaxis()->SetTitle("Efficiency");

  // efficiency vs pt eta
  TH2F *h_ele_ptEtaEff = (TH2F *)h_ele_simPtEta_matched->Clone("h_ele_ptEtaEff");
  h_ele_ptEtaEff->Reset();
  h_ele_ptEtaEff->Divide(h_ele_simPtEta_matched, h_simPtEta, 1, 1, "b");
  h_ele_ptEtaEff->GetYaxis()->SetTitle("p_{T} (GeV/c)");
  h_ele_ptEtaEff->GetXaxis()->SetTitle("#eta");

  std::cout << "[GsfElectronMCAnalyzer] q-misid calculation " << std::endl;
  // misid vs eta
  TH1F *h_ele_etaQmisid = (TH1F *)h_ele_simEta_matched_qmisid->Clone("h_ele_etaQmisid");
  h_ele_etaQmisid->Reset();
  h_ele_etaQmisid->Divide(h_ele_simEta_matched_qmisid, h_simEta, 1, 1, "b");
  h_ele_etaQmisid->Print();
  h_ele_etaQmisid->GetXaxis()->SetTitle("#eta");
  h_ele_etaQmisid->GetYaxis()->SetTitle("q misId");

  // misid vs z
  TH1F *h_ele_zQmisid = (TH1F *)h_ele_simZ_matched_qmisid->Clone("h_ele_zQmisid");
  h_ele_zQmisid->Reset();
  h_ele_zQmisid->Divide(h_ele_simZ_matched_qmisid, h_simZ, 1, 1, "b");
  h_ele_zQmisid->Print();
  h_ele_zQmisid->GetXaxis()->SetTitle("z (cm)");
  h_ele_zQmisid->GetYaxis()->SetTitle("q misId");

  // misid vs |eta|
  TH1F *h_ele_absetaQmisid = (TH1F *)h_ele_simAbsEta_matched_qmisid->Clone("h_ele_absetaQmisid");
  h_ele_absetaQmisid->Reset();
  h_ele_absetaQmisid->Divide(h_ele_simAbsEta_matched_qmisid, h_simAbsEta, 1, 1, "b");
  h_ele_absetaQmisid->GetXaxis()->SetTitle("|#eta|");
  h_ele_absetaQmisid->GetYaxis()->SetTitle("q misId");

  // misid vs pt
  TH1F *h_ele_ptQmisid = (TH1F *)h_ele_simPt_matched_qmisid->Clone("h_ele_ptQmisid");
  h_ele_ptQmisid->Reset();
  h_ele_ptQmisid->Divide(h_ele_simPt_matched_qmisid, h_simPt, 1, 1, "b");
  h_ele_ptQmisid->GetXaxis()->SetTitle("p_{T} (GeV/c)");
  h_ele_ptQmisid->GetYaxis()->SetTitle("q misId");

  std::cout << "[GsfElectronMCAnalyzer] all reco electrons " << std::endl;
  // rec/gen all electrons
  TH1F *h_ele_etaEff_all = (TH1F *)h_ele_vertexEta_all->Clone("h_ele_etaEff_all");
  h_ele_etaEff_all->Reset();
  h_ele_etaEff_all->Divide(h_ele_vertexEta_all, h_simEta, 1, 1, "b");
  h_ele_etaEff_all->Print();
  h_ele_etaEff_all->GetXaxis()->SetTitle("#eta");
  h_ele_etaEff_all->GetYaxis()->SetTitle("N_{rec}/N_{gen}");
  TH1F *h_ele_ptEff_all = (TH1F *)h_ele_vertexPt_all->Clone("h_ele_ptEff_all");
  h_ele_ptEff_all->Reset();
  h_ele_ptEff_all->Divide(h_ele_vertexPt_all, h_simPt, 1, 1, "b");
  h_ele_ptEff_all->Print();
  h_ele_ptEff_all->GetXaxis()->SetTitle("p_{T} (GeV/c)");
  h_ele_ptEff_all->GetYaxis()->SetTitle("N_{rec}/N_{gen}");

  // classes
  TH1F *h_ele_eta_goldenFrac = (TH1F *)h_ele_eta_golden->Clone("h_ele_eta_goldenFrac");
  h_ele_eta_goldenFrac->Reset();
  h_ele_eta_goldenFrac->Divide(h_ele_eta_golden, h_ele_eta, 1, 1, "b");
  h_ele_eta_goldenFrac->GetXaxis()->SetTitle("|#eta|");
  h_ele_eta_goldenFrac->GetYaxis()->SetTitle("Fraction of electrons");
  h_ele_eta_goldenFrac->SetTitle("fraction of golden electrons vs eta");
  TH1F *h_ele_eta_bbremFrac = (TH1F *)h_ele_eta_bbrem->Clone("h_ele_eta_bbremFrac");
  h_ele_eta_bbremFrac->Reset();
  h_ele_eta_bbremFrac->GetXaxis()->SetTitle("|#eta|");
  h_ele_eta_bbremFrac->GetYaxis()->SetTitle("Fraction of electrons");
  h_ele_eta_bbremFrac->Divide(h_ele_eta_bbrem, h_ele_eta, 1, 1, "b");
  h_ele_eta_bbremFrac->SetTitle("fraction of big brem electrons vs eta");
  TH1F *h_ele_eta_narrowFrac = (TH1F *)h_ele_eta_narrow->Clone("h_ele_eta_narrowFrac");
  h_ele_eta_narrowFrac->Reset();
  h_ele_eta_narrowFrac->Divide(h_ele_eta_narrow, h_ele_eta, 1, 1, "b");
  h_ele_eta_narrowFrac->GetXaxis()->SetTitle("|#eta|");
  h_ele_eta_narrowFrac->GetYaxis()->SetTitle("Fraction of electrons");
  h_ele_eta_narrowFrac->SetTitle("fraction of narrow electrons vs eta");
  TH1F *h_ele_eta_showerFrac = (TH1F *)h_ele_eta_shower->Clone("h_ele_eta_showerFrac");
  h_ele_eta_showerFrac->Reset();
  h_ele_eta_showerFrac->Divide(h_ele_eta_shower, h_ele_eta, 1, 1, "b");
  h_ele_eta_showerFrac->GetXaxis()->SetTitle("|#eta|");
  h_ele_eta_showerFrac->GetYaxis()->SetTitle("Fraction of electrons");
  h_ele_eta_showerFrac->SetTitle("fraction of showering electrons vs eta");

  // fbrem
  TH1F *h_ele_xOverX0VsEta = new TH1F("h_ele_xOverx0VsEta", "mean X/X_0 vs eta", nbineta / 2, 0.0, 2.5);
  for (int ibin = 1; ibin < h_ele_fbremVsEta_mean->GetNbinsX() + 1; ibin++) {
    double xOverX0 = 0.;
    if (h_ele_fbremVsEta_mean->GetBinContent(ibin) > 0.)
      xOverX0 = -log(h_ele_fbremVsEta_mean->GetBinContent(ibin));
    h_ele_xOverX0VsEta->SetBinContent(ibin, xOverX0);
  }

  //profiles from 2D histos
  TProfile *p_ele_PoPtrueVsEta = h_ele_PoPtrueVsEta->ProfileX();
  p_ele_PoPtrueVsEta->SetTitle("mean ele momentum / gen momentum vs eta");
  p_ele_PoPtrueVsEta->GetXaxis()->SetTitle("#eta");
  p_ele_PoPtrueVsEta->GetYaxis()->SetTitle("<P/P_{gen}>");
  p_ele_PoPtrueVsEta->Write();
  TProfile *p_ele_PoPtrueVsPhi = h_ele_PoPtrueVsPhi->ProfileX();
  p_ele_PoPtrueVsPhi->SetTitle("mean ele momentum / gen momentum vs phi");
  p_ele_PoPtrueVsPhi->GetXaxis()->SetTitle("#phi (rad)");
  p_ele_PoPtrueVsPhi->GetYaxis()->SetTitle("<P/P_{gen}>");
  p_ele_PoPtrueVsPhi->Write();
  TProfile *p_ele_EoEtruePfVsEg_x = histSclEoEtruePfVsEg->ProfileX();
  p_ele_EoEtruePfVsEg_x->SetTitle("mean pflow sc energy / true energy vs e/g sc energy");
  p_ele_EoEtruePfVsEg_x->GetXaxis()->SetTitle("E/E_{gen} (e/g)");
  p_ele_EoEtruePfVsEg_x->GetYaxis()->SetTitle("<E/E_{gen}> (pflow)");
  p_ele_EoEtruePfVsEg_x->Write();
  TProfile *p_ele_EoEtruePfVsEg_y = histSclEoEtruePfVsEg->ProfileY();
  p_ele_EoEtruePfVsEg_y->SetTitle("mean e/g sc energy / true energy vs pflow sc energy");
  p_ele_EoEtruePfVsEg_y->GetXaxis()->SetTitle("E/E_{gen} (pflow)");
  p_ele_EoEtruePfVsEg_y->GetYaxis()->SetTitle("<E/E_{gen}> (eg)");
  p_ele_EoEtruePfVsEg_y->Write();
  TProfile *p_ele_EtaMnEtaTrueVsEta = h_ele_EtaMnEtaTrueVsEta->ProfileX();
  p_ele_EtaMnEtaTrueVsEta->SetTitle("mean ele eta - gen eta vs eta");
  p_ele_EtaMnEtaTrueVsEta->GetXaxis()->SetTitle("#eta");
  p_ele_EtaMnEtaTrueVsEta->GetYaxis()->SetTitle("<#eta_{rec} - #eta_{gen}>");
  p_ele_EtaMnEtaTrueVsEta->Write();
  TProfile *p_ele_EtaMnEtaTrueVsPhi = h_ele_EtaMnEtaTrueVsPhi->ProfileX();
  p_ele_EtaMnEtaTrueVsPhi->SetTitle("mean ele eta - gen eta vs phi");
  p_ele_EtaMnEtaTrueVsPhi->GetXaxis()->SetTitle("#phi (rad)");
  p_ele_EtaMnEtaTrueVsPhi->GetYaxis()->SetTitle("<#eta_{rec} - #eta_{gen}>");
  p_ele_EtaMnEtaTrueVsPhi->Write();
  TProfile *p_ele_PhiMnPhiTrueVsEta = h_ele_PhiMnPhiTrueVsEta->ProfileX();
  p_ele_PhiMnPhiTrueVsEta->SetTitle("mean ele phi - gen phi vs eta");
  p_ele_PhiMnPhiTrueVsEta->GetXaxis()->SetTitle("#eta");
  p_ele_PhiMnPhiTrueVsEta->GetYaxis()->SetTitle("<#phi_{rec} - #phi_{gen}> (rad)");
  p_ele_PhiMnPhiTrueVsEta->Write();
  TProfile *p_ele_PhiMnPhiTrueVsPhi = h_ele_PhiMnPhiTrueVsPhi->ProfileX();
  p_ele_PhiMnPhiTrueVsPhi->SetTitle("mean ele phi - gen phi vs phi");
  p_ele_PhiMnPhiTrueVsPhi->GetXaxis()->SetTitle("#phi (rad)");
  p_ele_PhiMnPhiTrueVsPhi->Write();
  TProfile *p_ele_vertexPtVsEta = h_ele_vertexPtVsEta->ProfileX();
  p_ele_vertexPtVsEta->SetTitle("mean ele transverse momentum vs eta");
  p_ele_vertexPtVsEta->GetXaxis()->SetTitle("#eta");
  p_ele_vertexPtVsEta->GetYaxis()->SetTitle("<p_{T}> (GeV/c)");
  p_ele_vertexPtVsEta->Write();
  TProfile *p_ele_vertexPtVsPhi = h_ele_vertexPtVsPhi->ProfileX();
  p_ele_vertexPtVsPhi->SetTitle("mean ele transverse momentum vs phi");
  p_ele_vertexPtVsPhi->GetXaxis()->SetTitle("#phi (rad)");
  p_ele_vertexPtVsPhi->GetYaxis()->SetTitle("<p_{T}> (GeV/c)");
  p_ele_vertexPtVsPhi->Write();
  TProfile *p_ele_EoPVsEta = h_ele_EoPVsEta->ProfileX();
  p_ele_EoPVsEta->SetTitle("mean ele E/p vs eta");
  p_ele_EoPVsEta->GetXaxis()->SetTitle("#eta");
  p_ele_EoPVsEta->GetYaxis()->SetTitle("<E/P_{vertex}>");
  p_ele_EoPVsEta->Write();
  TProfile *p_ele_EoPVsPhi = h_ele_EoPVsPhi->ProfileX();
  p_ele_EoPVsPhi->SetTitle("mean ele E/p vs phi");
  p_ele_EoPVsPhi->GetXaxis()->SetTitle("#phi (rad)");
  p_ele_EoPVsPhi->GetYaxis()->SetTitle("<E/P_{vertex}>");
  p_ele_EoPVsPhi->Write();
  TProfile *p_ele_EoPoutVsEta = h_ele_EoPoutVsEta->ProfileX();
  p_ele_EoPoutVsEta->SetTitle("mean ele E/pout vs eta");
  p_ele_EoPoutVsEta->GetXaxis()->SetTitle("#eta");
  p_ele_EoPoutVsEta->GetYaxis()->SetTitle("<E_{seed}/P_{out}>");
  p_ele_EoPoutVsEta->Write();
  TProfile *p_ele_EoPoutVsPhi = h_ele_EoPoutVsPhi->ProfileX();
  p_ele_EoPoutVsPhi->SetTitle("mean ele E/pout vs phi");
  p_ele_EoPoutVsPhi->GetXaxis()->SetTitle("#phi (rad)");
  p_ele_EoPoutVsPhi->GetYaxis()->SetTitle("<E_{seed}/P_{out}>");
  p_ele_EoPoutVsPhi->Write();
  TProfile *p_ele_EeleOPoutVsEta = h_ele_EeleOPoutVsEta->ProfileX();
  p_ele_EeleOPoutVsEta->SetTitle("mean ele Eele/pout vs eta");
  p_ele_EeleOPoutVsEta->GetXaxis()->SetTitle("#eta");
  p_ele_EeleOPoutVsEta->GetYaxis()->SetTitle("<E_{ele}/P_{out}>");
  p_ele_EeleOPoutVsEta->Write();
  TProfile *p_ele_EeleOPoutVsPhi = h_ele_EeleOPoutVsPhi->ProfileX();
  p_ele_EeleOPoutVsPhi->SetTitle("mean ele Eele/pout vs phi");
  p_ele_EeleOPoutVsPhi->GetXaxis()->SetTitle("#phi (rad)");
  p_ele_EeleOPoutVsPhi->GetYaxis()->SetTitle("<E_{ele}/P_{out}>");
  p_ele_EeleOPoutVsPhi->Write();
  TProfile *p_ele_HoEVsEta = h_ele_HoEVsEta->ProfileX();
  p_ele_HoEVsEta->SetTitle("mean ele H/E vs eta");
  p_ele_HoEVsEta->GetXaxis()->SetTitle("#eta");
  p_ele_HoEVsEta->GetYaxis()->SetTitle("<H/E>");
  p_ele_HoEVsEta->Write();
  TProfile *p_ele_HoEVsPhi = h_ele_HoEVsPhi->ProfileX();
  p_ele_HoEVsPhi->SetTitle("mean ele H/E vs phi");
  p_ele_HoEVsPhi->GetXaxis()->SetTitle("#phi (rad)");
  p_ele_HoEVsPhi->GetYaxis()->SetTitle("<H/E>");
  p_ele_HoEVsPhi->Write();
  TProfile *p_ele_chi2VsEta = h_ele_chi2VsEta->ProfileX();
  p_ele_chi2VsEta->SetTitle("mean ele track chi2 vs eta");
  p_ele_chi2VsEta->GetXaxis()->SetTitle("#eta");
  p_ele_chi2VsEta->GetYaxis()->SetTitle("<#Chi^{2}>");
  p_ele_chi2VsEta->Write();
  TProfile *p_ele_chi2VsPhi = h_ele_chi2VsPhi->ProfileX();
  p_ele_chi2VsPhi->SetTitle("mean ele track chi2 vs phi");
  p_ele_chi2VsPhi->GetXaxis()->SetTitle("#phi (rad)");
  p_ele_chi2VsPhi->GetYaxis()->SetTitle("<#Chi^{2}>");
  p_ele_chi2VsPhi->Write();
  TProfile *p_ele_foundHitsVsEta = h_ele_foundHitsVsEta->ProfileX();
  p_ele_foundHitsVsEta->SetTitle("mean ele track # found hits vs eta");
  p_ele_foundHitsVsEta->GetXaxis()->SetTitle("#eta");
  p_ele_foundHitsVsEta->GetYaxis()->SetTitle("<N_{hits}>");
  p_ele_foundHitsVsEta->Write();
  TProfile *p_ele_foundHitsVsPhi = h_ele_foundHitsVsPhi->ProfileX();
  p_ele_foundHitsVsPhi->SetTitle("mean ele track # found hits vs phi");
  p_ele_foundHitsVsPhi->GetXaxis()->SetTitle("#phi (rad)");
  p_ele_foundHitsVsPhi->GetYaxis()->SetTitle("<N_{hits}>");
  p_ele_foundHitsVsPhi->Write();
  TProfile *p_ele_lostHitsVsEta = h_ele_lostHitsVsEta->ProfileX();
  p_ele_lostHitsVsEta->SetTitle("mean ele track # lost hits vs eta");
  p_ele_lostHitsVsEta->GetXaxis()->SetTitle("#eta");
  p_ele_lostHitsVsEta->GetYaxis()->SetTitle("<N_{hits}>");
  p_ele_lostHitsVsEta->Write();
  TProfile *p_ele_lostHitsVsPhi = h_ele_lostHitsVsPhi->ProfileX();
  p_ele_lostHitsVsPhi->SetTitle("mean ele track # lost hits vs phi");
  p_ele_lostHitsVsPhi->GetXaxis()->SetTitle("#phi (rad)");
  p_ele_lostHitsVsPhi->GetYaxis()->SetTitle("<N_{hits}>");
  p_ele_lostHitsVsPhi->Write();
  TProfile *p_ele_vertexTIPVsEta = h_ele_vertexTIPVsEta->ProfileX();
  p_ele_vertexTIPVsEta->SetTitle("mean tip (wrt gen vtx) vs eta");
  p_ele_vertexTIPVsEta->GetXaxis()->SetTitle("#eta");
  p_ele_vertexTIPVsEta->GetYaxis()->SetTitle("<TIP> (cm)");
  p_ele_vertexTIPVsEta->Write();
  TProfile *p_ele_vertexTIPVsPhi = h_ele_vertexTIPVsPhi->ProfileX();
  p_ele_vertexTIPVsPhi->SetTitle("mean tip (wrt gen vtx) vs phi");
  p_ele_vertexTIPVsPhi->GetXaxis()->SetTitle("#phi");
  p_ele_vertexTIPVsPhi->GetYaxis()->SetTitle("<TIP> (cm)");
  p_ele_vertexTIPVsPhi->Write();
  TProfile *p_ele_vertexTIPVsPt = h_ele_vertexTIPVsPt->ProfileX();
  p_ele_vertexTIPVsPt->SetTitle("mean tip (wrt gen vtx) vs phi");
  p_ele_vertexTIPVsPt->GetXaxis()->SetTitle("p_{T} (GeV/c)");
  p_ele_vertexTIPVsPt->GetYaxis()->SetTitle("<TIP> (cm)");
  p_ele_vertexTIPVsPt->Write();

  // mc truth
  h_mcNum->Write();
  h_eleNum->Write();
  h_gamNum->Write();

  // rec event
  histNum_->Write();

  // mc
  h_simEta->Write();
  h_simAbsEta->Write();
  h_simP->Write();
  h_simPt->Write();
  h_simZ->Write();
  h_simPhi->Write();
  h_simPtEta->Write();

  // all electrons
  h_ele_EoverP_all->Write();
  h_ele_EseedOP_all->Write();
  h_ele_EoPout_all->Write();
  h_ele_EeleOPout_all->Write();
  h_ele_dEtaSc_propVtx_all->Write();
  h_ele_dPhiSc_propVtx_all->Write();
  h_ele_dEtaCl_propOut_all->Write();
  h_ele_dPhiCl_propOut_all->Write();
  h_ele_HoE_all->Write();
  h_ele_TIP_all->Write();
  h_ele_vertexPt_all->Write();
  h_ele_Et_all->Write();
  h_ele_vertexEta_all->Write();
  h_ele_mee_all->Write();
  h_ele_mee_os->Write();
  h_ele_mee_os_ebeb->Write();
  h_ele_mee_os_ebee->Write();
  h_ele_mee_os_eeee->Write();
  h_ele_mee_os_gg->Write();
  h_ele_mee_os_gb->Write();
  h_ele_mee_os_bb->Write();
  h_ele_E2mnE1vsMee_all->Write();
  h_ele_E2mnE1vsMee_egeg_all->Write();

  // charge ID
  h_ele_charge->Write();
  h_ele_simEta_matched_qmisid->Write();
  h_ele_simAbsEta_matched_qmisid->Write();
  h_ele_simPt_matched_qmisid->Write();
  h_ele_simPhi_matched_qmisid->Write();
  h_ele_simZ_matched_qmisid->Write();

  // matched electrons
  h_ele_vertexP->Write();
  h_ele_vertexPt->Write();
  h_ele_Et->Write();
  h_ele_vertexPtVsEta->Write();
  h_ele_vertexPtVsPhi->Write();
  h_ele_simPt_matched->Write();
  h_ele_vertexEta->Write();
  h_ele_vertexEtaVsPhi->Write();
  h_ele_simAbsEta_matched->Write();
  h_ele_simEta_matched->Write();
  h_ele_simPhi_matched->Write();
  h_ele_simPtEta_matched->Write();
  h_ele_vertexPhi->Write();
  h_ele_vertexX->Write();
  h_ele_vertexY->Write();
  h_ele_vertexZ->Write();
  h_ele_vertexTIP->Write();
  h_ele_simZ_matched->Write();
  h_ele_vertexTIPVsEta->Write();
  h_ele_vertexTIPVsPhi->Write();
  h_ele_vertexTIPVsPt->Write();
  h_ele_PoPtrue->Write();
  h_ele_PoPtrueVsEta->Write();
  h_ele_PoPtrueVsPhi->Write();
  h_ele_PoPtrueVsPt->Write();
  h_ele_PoPtrue_barrel->Write();
  h_ele_PoPtrue_endcaps->Write();
  h_ele_PoPtrue_golden_barrel->Write();
  h_ele_PoPtrue_golden_endcaps->Write();
  h_ele_PoPtrue_showering_barrel->Write();
  h_ele_PoPtrue_showering_endcaps->Write();
  h_ele_PtoPttrue->Write();
  h_ele_PtoPttrue_barrel->Write();
  h_ele_PtoPttrue_endcaps->Write();
  h_ele_ChargeMnChargeTrue->Write();
  h_ele_EtaMnEtaTrue->Write();
  h_ele_EtaMnEtaTrue_barrel->Write();
  h_ele_EtaMnEtaTrue_endcaps->Write();
  h_ele_EtaMnEtaTrueVsEta->Write();
  h_ele_EtaMnEtaTrueVsPhi->Write();
  h_ele_EtaMnEtaTrueVsPt->Write();
  h_ele_PhiMnPhiTrue->Write();
  h_ele_PhiMnPhiTrue_barrel->Write();
  h_ele_PhiMnPhiTrue_endcaps->Write();
  h_ele_PhiMnPhiTrue2->Write();
  h_ele_PhiMnPhiTrueVsEta->Write();
  h_ele_PhiMnPhiTrueVsPhi->Write();
  h_ele_PhiMnPhiTrueVsPt->Write();

  // matched electron, superclusters
  histSclEn_->Write();
  histSclEoEtrue_barrel->Write();
  histSclEoEtrue_endcaps->Write();
  histSclEoEtrue_barrel_eg->Write();
  histSclEoEtrue_endcaps_eg->Write();
  histSclEoEtrue_barrel_etagap->Write();
  histSclEoEtrue_barrel_phigap->Write();
  histSclEoEtrue_ebeegap->Write();
  histSclEoEtrue_endcaps->Write();
  histSclEoEtrue_endcaps_deegap->Write();
  histSclEoEtrue_endcaps_ringgap->Write();
  histSclEoEtruePfVsEg->Write();
  histSclEoEtrue_barrel_new->Write();
  histSclEoEtrue_endcaps_new->Write();
  histSclEoEtrue_barrel_eg_new->Write();
  histSclEoEtrue_endcaps_eg_new->Write();
  histSclEoEtrue_barrel_etagap_new->Write();
  histSclEoEtrue_barrel_phigap_new->Write();
  histSclEoEtrue_ebeegap_new->Write();
  histSclEoEtrue_endcaps_new->Write();
  histSclEoEtrue_endcaps_deegap_new->Write();
  histSclEoEtrue_endcaps_ringgap_new->Write();
  histSclEoEtruePfVsEg->Write();
  histSclEt_->Write();
  histSclEtVsEta_->Write();
  histSclEtVsPhi_->Write();
  histSclEtaVsPhi_->Write();
  histSclEta_->Write();
  histSclPhi_->Write();
  histSclSigEtaEta_->Write();
  histSclSigEtaEta_barrel_->Write();
  histSclSigEtaEta_endcaps_->Write();
  histSclSigIEtaIEta_->Write();
  histSclSigIEtaIEta_barrel_->Write();
  histSclSigIEtaIEta_endcaps_->Write();
  histSclE1x5_->Write();
  histSclE1x5_barrel_->Write();
  histSclE1x5_endcaps_->Write();
  histSclE2x5max_->Write();
  histSclE2x5max_barrel_->Write();
  histSclE2x5max_endcaps_->Write();
  histSclE5x5_->Write();
  histSclE5x5_barrel_->Write();
  histSclE5x5_endcaps_->Write();
  histSclSigEtaEta_eg_->Write();
  histSclSigEtaEta_eg_barrel_->Write();
  histSclSigEtaEta_eg_endcaps_->Write();
  histSclSigIEtaIEta_eg_->Write();
  histSclSigIEtaIEta_eg_barrel_->Write();
  histSclSigIEtaIEta_eg_endcaps_->Write();
  histSclE1x5_eg_->Write();
  histSclE1x5_eg_barrel_->Write();
  histSclE1x5_eg_endcaps_->Write();
  histSclE2x5max_eg_->Write();
  histSclE2x5max_eg_barrel_->Write();
  histSclE2x5max_eg_endcaps_->Write();
  histSclE5x5_eg_->Write();
  histSclE5x5_eg_barrel_->Write();
  histSclE5x5_eg_endcaps_->Write();

  // matched electron, gsf tracks
  h_ele_ambiguousTracks->Write();
  h_ele_ambiguousTracksVsEta->Write();
  h_ele_ambiguousTracksVsPhi->Write();
  h_ele_ambiguousTracksVsPt->Write();
  h_ele_foundHits->Write();
  h_ele_foundHits_barrel->Write();
  h_ele_foundHits_endcaps->Write();
  h_ele_foundHitsVsEta->Write();
  h_ele_foundHitsVsPhi->Write();
  h_ele_foundHitsVsPt->Write();
  h_ele_lostHits->Write();
  h_ele_lostHits_barrel->Write();
  h_ele_lostHits_endcaps->Write();
  h_ele_lostHitsVsEta->Write();
  h_ele_lostHitsVsPhi->Write();
  h_ele_lostHitsVsPt->Write();
  h_ele_chi2->Write();
  h_ele_chi2_barrel->Write();
  h_ele_chi2_endcaps->Write();
  h_ele_chi2VsEta->Write();
  h_ele_chi2VsPhi->Write();
  h_ele_chi2VsPt->Write();
  h_ele_PinMnPout->Write();
  h_ele_PinMnPout_mode->Write();
  h_ele_PinMnPoutVsEta_mode->Write();
  h_ele_PinMnPoutVsPhi_mode->Write();
  h_ele_PinMnPoutVsPt_mode->Write();
  h_ele_PinMnPoutVsE_mode->Write();
  h_ele_PinMnPoutVsChi2_mode->Write();
  h_ele_outerP->Write();
  h_ele_outerP_mode->Write();
  h_ele_outerPVsEta_mode->Write();
  h_ele_outerPt->Write();
  h_ele_outerPt_mode->Write();
  h_ele_outerPtVsEta_mode->Write();
  h_ele_outerPtVsPhi_mode->Write();
  h_ele_outerPtVsPt_mode->Write();

  // matched electrons, matching
  h_ele_EoP->Write();
  h_ele_EoP_eg->Write();
  h_ele_EoP_barrel->Write();
  h_ele_EoP_eg_barrel->Write();
  h_ele_EoP_endcaps->Write();
  h_ele_EoP_eg_endcaps->Write();
  h_ele_EoPVsEta->Write();
  h_ele_EoPVsPhi->Write();
  h_ele_EoPVsE->Write();
  h_ele_EseedOP->Write();
  h_ele_EseedOP_eg->Write();
  h_ele_EseedOP_barrel->Write();
  h_ele_EseedOP_eg_barrel->Write();
  h_ele_EseedOP_endcaps->Write();
  h_ele_EseedOP_eg_endcaps->Write();
  h_ele_EseedOPVsEta->Write();
  h_ele_EseedOPVsPhi->Write();
  h_ele_EseedOPVsE->Write();
  h_ele_EoPout->Write();
  h_ele_EoPout_eg->Write();
  h_ele_EoPout_barrel->Write();
  h_ele_EoPout_eg_barrel->Write();
  h_ele_EoPout_endcaps->Write();
  h_ele_EoPout_eg_endcaps->Write();
  h_ele_EoPoutVsEta->Write();
  h_ele_EoPoutVsPhi->Write();
  h_ele_EoPoutVsE->Write();
  h_ele_EeleOPout->Write();
  h_ele_EeleOPout_eg->Write();
  h_ele_EeleOPout_barrel->Write();
  h_ele_EeleOPout_eg_barrel->Write();
  h_ele_EeleOPout_endcaps->Write();
  h_ele_EeleOPout_eg_endcaps->Write();
  h_ele_EeleOPoutVsEta->Write();
  h_ele_EeleOPoutVsPhi->Write();
  h_ele_EeleOPoutVsE->Write();
  h_ele_dEtaSc_propVtx->Write();
  h_ele_dEtaSc_propVtx_eg->Write();
  h_ele_dEtaSc_propVtx_barrel->Write();
  h_ele_dEtaSc_propVtx_eg_barrel->Write();
  h_ele_dEtaSc_propVtx_endcaps->Write();
  h_ele_dEtaSc_propVtx_eg_endcaps->Write();
  h_ele_dEtaScVsEta_propVtx->Write();
  h_ele_dEtaScVsPhi_propVtx->Write();
  h_ele_dEtaScVsPt_propVtx->Write();
  h_ele_dPhiSc_propVtx->Write();
  h_ele_dPhiSc_propVtx_eg->Write();
  h_ele_dPhiSc_propVtx_barrel->Write();
  h_ele_dPhiSc_propVtx_eg_barrel->Write();
  h_ele_dPhiSc_propVtx_endcaps->Write();
  h_ele_dPhiSc_propVtx_eg_endcaps->Write();
  h_ele_dPhiScVsEta_propVtx->Write();
  h_ele_dPhiScVsPhi_propVtx->Write();
  h_ele_dPhiScVsPt_propVtx->Write();
  h_ele_dEtaCl_propOut->Write();
  h_ele_dEtaCl_propOut_eg->Write();
  h_ele_dEtaCl_propOut_barrel->Write();
  h_ele_dEtaCl_propOut_eg_barrel->Write();
  h_ele_dEtaCl_propOut_endcaps->Write();
  h_ele_dEtaCl_propOut_eg_endcaps->Write();
  h_ele_dEtaClVsEta_propOut->Write();
  h_ele_dEtaClVsPhi_propOut->Write();
  h_ele_dEtaClVsPt_propOut->Write();
  h_ele_dPhiCl_propOut->Write();
  h_ele_dPhiCl_propOut_eg->Write();
  h_ele_dPhiCl_propOut_barrel->Write();
  h_ele_dPhiCl_propOut_eg_barrel->Write();
  h_ele_dPhiCl_propOut_endcaps->Write();
  h_ele_dPhiCl_propOut_eg_endcaps->Write();
  h_ele_dPhiClVsEta_propOut->Write();
  h_ele_dPhiClVsPhi_propOut->Write();
  h_ele_dPhiClVsPt_propOut->Write();
  h_ele_dEtaEleCl_propOut->Write();
  h_ele_dEtaEleCl_propOut_eg->Write();
  h_ele_dEtaEleCl_propOut_barrel->Write();
  h_ele_dEtaEleCl_propOut_eg_barrel->Write();
  h_ele_dEtaEleCl_propOut_endcaps->Write();
  h_ele_dEtaEleCl_propOut_eg_endcaps->Write();
  h_ele_dEtaEleClVsEta_propOut->Write();
  h_ele_dEtaEleClVsPhi_propOut->Write();
  h_ele_dEtaEleClVsPt_propOut->Write();
  h_ele_dPhiEleCl_propOut->Write();
  h_ele_dPhiEleCl_propOut_eg->Write();
  h_ele_dPhiEleCl_propOut_barrel->Write();
  h_ele_dPhiEleCl_propOut_eg_barrel->Write();
  h_ele_dPhiEleCl_propOut_endcaps->Write();
  h_ele_dPhiEleCl_propOut_eg_endcaps->Write();
  h_ele_dPhiEleClVsEta_propOut->Write();
  h_ele_dPhiEleClVsPhi_propOut->Write();
  h_ele_dPhiEleClVsPt_propOut->Write();
  h_ele_HoE->Write();
  h_ele_HoE_eg->Write();
  h_ele_HoE_barrel->Write();
  h_ele_HoE_eg_barrel->Write();
  h_ele_HoE_endcaps->Write();
  h_ele_HoE_eg_endcaps->Write();
  h_ele_HoE_fiducial->Write();
  h_ele_HoEVsEta->Write();
  h_ele_HoEVsPhi->Write();
  h_ele_HoEVsE->Write();

  h_ele_seed_dphi2_->Write();
  h_ele_seed_subdet2_->Write();
  TProfile *p_ele_seed_dphi2VsEta_ = h_ele_seed_dphi2VsEta_->ProfileX();
  p_ele_seed_dphi2VsEta_->SetTitle("mean ele seed dphi 2nd layer vs eta");
  p_ele_seed_dphi2VsEta_->GetXaxis()->SetTitle("#eta");
  p_ele_seed_dphi2VsEta_->GetYaxis()->SetTitle("<#phi_{pred} - #phi_{hit}, 2nd layer> (rad)");
  p_ele_seed_dphi2VsEta_->SetMinimum(-0.004);
  p_ele_seed_dphi2VsEta_->SetMaximum(0.004);
  p_ele_seed_dphi2VsEta_->Write();
  TProfile *p_ele_seed_dphi2VsPt_ = h_ele_seed_dphi2VsPt_->ProfileX();
  p_ele_seed_dphi2VsPt_->SetTitle("mean ele seed dphi 2nd layer vs pt");
  p_ele_seed_dphi2VsPt_->GetXaxis()->SetTitle("p_{T} (GeV/c)");
  p_ele_seed_dphi2VsPt_->GetYaxis()->SetTitle("<#phi_{pred} - #phi_{hit}, 2nd layer> (rad)");
  p_ele_seed_dphi2VsPt_->SetMinimum(-0.004);
  p_ele_seed_dphi2VsPt_->SetMaximum(0.004);
  p_ele_seed_dphi2VsPt_->Write();
  h_ele_seed_drz2_->Write();
  TProfile *p_ele_seed_drz2VsEta_ = h_ele_seed_drz2VsEta_->ProfileX();
  p_ele_seed_drz2VsEta_->SetTitle("mean ele seed dr(dz) 2nd layer vs eta");
  p_ele_seed_drz2VsEta_->GetXaxis()->SetTitle("#eta");
  p_ele_seed_drz2VsEta_->GetYaxis()->SetTitle("<r(z)_{pred} - r(z)_{hit}, 2nd layer> (cm)");
  p_ele_seed_drz2VsEta_->SetMinimum(-0.15);
  p_ele_seed_drz2VsEta_->SetMaximum(0.15);
  p_ele_seed_drz2VsEta_->Write();
  TProfile *p_ele_seed_drz2VsPt_ = h_ele_seed_drz2VsPt_->ProfileX();
  p_ele_seed_drz2VsPt_->SetTitle("mean ele seed dr(dz) 2nd layer vs pt");
  p_ele_seed_drz2VsPt_->GetXaxis()->SetTitle("p_{T} (GeV/c)");
  p_ele_seed_drz2VsPt_->GetYaxis()->SetTitle("<r(z)_{pred} - r(z)_{hit}, 2nd layer> (cm)");
  p_ele_seed_drz2VsPt_->SetMinimum(-0.15);
  p_ele_seed_drz2VsPt_->SetMaximum(0.15);
  p_ele_seed_drz2VsPt_->Write();

  // classes
  h_ele_classes->Write();
  h_ele_eta->Write();
  h_ele_eta_golden->Write();
  h_ele_eta_bbrem->Write();
  h_ele_eta_narrow->Write();
  h_ele_eta_shower->Write();
  h_ele_PinVsPoutGolden_mode->Write();
  h_ele_PinVsPoutShowering_mode->Write();
  h_ele_PinVsPoutGolden_mean->Write();
  h_ele_PinVsPoutShowering_mean->Write();
  h_ele_PtinVsPtoutGolden_mode->Write();
  h_ele_PtinVsPtoutShowering_mode->Write();
  h_ele_PtinVsPtoutGolden_mean->Write();
  h_ele_PtinVsPtoutShowering_mean->Write();
  histSclEoEtrueGolden_barrel->Write();
  histSclEoEtrueGolden_endcaps->Write();
  histSclEoEtrueShowering_barrel->Write();
  histSclEoEtrueShowering_endcaps->Write();

  // fbrem
  h_ele_fbrem->Write();
  h_ele_fbrem_eg->Write();
  h_ele_fbremVsEta_mode->GetXaxis()->SetTitle("#eta");
  h_ele_fbremVsEta_mode->GetYaxis()->SetTitle("<P_{in} - P_{out} / P_{in}>");
  h_ele_fbremVsEta_mode->Write();
  h_ele_fbremVsEta_mean->GetXaxis()->SetTitle("#eta");
  h_ele_fbremVsEta_mean->GetYaxis()->SetTitle("<P_{in} - P_{out} / P_{in}>");
  h_ele_fbremVsEta_mean->Write();
  h_ele_eta_goldenFrac->Write();
  h_ele_eta_bbremFrac->Write();
  h_ele_eta_narrowFrac->Write();
  h_ele_eta_showerFrac->Write();
  h_ele_xOverX0VsEta->Write();

  // efficiencies
  h_ele_etaEff->Write();
  h_ele_zEff->Write();
  h_ele_phiEff->Write();
  h_ele_absetaEff->Write();
  h_ele_ptEff->Write();
  h_ele_ptEtaEff->Write();
  h_ele_etaEff_all->Write();
  h_ele_ptEff_all->Write();

  // q misid
  h_ele_etaQmisid->Write();
  h_ele_zQmisid->Write();
  h_ele_absetaQmisid->Write();
  h_ele_ptQmisid->Write();

  // e/g et pflow electrons
  h_ele_mva->Write();
  h_ele_mva_eg->Write();
  h_ele_provenance->Write();

  // isolation
  h_ele_tkSumPt_dr03->GetXaxis()->SetTitle("TkIsoSum, cone 0.3 (GeV/c)");
  h_ele_tkSumPt_dr03->GetYaxis()->SetTitle("Events");
  h_ele_tkSumPt_dr03->Write();
  h_ele_ecalRecHitSumEt_dr03->GetXaxis()->SetTitle("EcalIsoSum, cone 0.3 (GeV)");
  h_ele_ecalRecHitSumEt_dr03->GetYaxis()->SetTitle("Events");
  h_ele_ecalRecHitSumEt_dr03->Write();
  h_ele_hcalDepth1TowerSumEt_dr03->GetXaxis()->SetTitle("Hcal1IsoSum, cone 0.3 (GeV)");
  h_ele_hcalDepth1TowerSumEt_dr03->GetYaxis()->SetTitle("Events");
  h_ele_hcalDepth1TowerSumEt_dr03->Write();
  h_ele_hcalDepth2TowerSumEt_dr03->GetXaxis()->SetTitle("Hcal2IsoSum, cone 0.3 (GeV)");
  h_ele_hcalDepth2TowerSumEt_dr03->GetYaxis()->SetTitle("Events");
  h_ele_hcalDepth2TowerSumEt_dr03->Write();
  h_ele_tkSumPt_dr04->GetXaxis()->SetTitle("TkIsoSum, cone 0.4 (GeV/c)");
  h_ele_tkSumPt_dr04->GetYaxis()->SetTitle("Events");
  h_ele_tkSumPt_dr04->Write();
  h_ele_ecalRecHitSumEt_dr04->GetXaxis()->SetTitle("EcalIsoSum, cone 0.4 (GeV)");
  h_ele_ecalRecHitSumEt_dr04->GetYaxis()->SetTitle("Events");
  h_ele_ecalRecHitSumEt_dr04->Write();
  h_ele_hcalDepth1TowerSumEt_dr04->GetXaxis()->SetTitle("Hcal1IsoSum, cone 0.4 (GeV)");
  h_ele_hcalDepth1TowerSumEt_dr04->GetYaxis()->SetTitle("Events");
  h_ele_hcalDepth1TowerSumEt_dr04->Write();
  h_ele_hcalDepth2TowerSumEt_dr04->GetXaxis()->SetTitle("Hcal2IsoSum, cone 0.4 (GeV)");
  h_ele_hcalDepth2TowerSumEt_dr04->GetYaxis()->SetTitle("Events");
  h_ele_hcalDepth2TowerSumEt_dr04->Write();
}

GsfElectronMCAnalyzer::~GsfElectronMCAnalyzer() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  histfile_->Write();
  histfile_->Close();
}

//=========================================================================
// Main method
//=========================================================================

void GsfElectronMCAnalyzer::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  std::cout << "analyzing new event " << std::endl;
  // get electrons

  edm::Handle<GsfElectronCollection> gsfElectrons;
  iEvent.getByLabel(electronCollection_, gsfElectrons);
  edm::LogInfo("") << "\n\n =================> Treating event " << iEvent.id() << " Number of electrons "
                   << gsfElectrons.product()->size();

  edm::Handle<GenParticleCollection> genParticles;
  iEvent.getByLabel(mcTruthCollection_, genParticles);

  histNum_->Fill((*gsfElectrons).size());

  // all rec electrons
  for (reco::GsfElectronCollection::const_iterator gsfIter = gsfElectrons->begin(); gsfIter != gsfElectrons->end();
       gsfIter++) {
    // preselect electrons
    if (gsfIter->pt() > maxPt_ || std::abs(gsfIter->eta()) > maxAbsEta_)
      continue;
    h_ele_EoverP_all->Fill(gsfIter->eSuperClusterOverP());
    h_ele_EseedOP_all->Fill(gsfIter->eSeedClusterOverP());
    h_ele_EoPout_all->Fill(gsfIter->eSeedClusterOverPout());
    h_ele_EeleOPout_all->Fill(gsfIter->eEleClusterOverPout());
    h_ele_dEtaSc_propVtx_all->Fill(gsfIter->deltaEtaSuperClusterTrackAtVtx());
    h_ele_dPhiSc_propVtx_all->Fill(gsfIter->deltaPhiSuperClusterTrackAtVtx());
    h_ele_dEtaCl_propOut_all->Fill(gsfIter->deltaEtaSeedClusterTrackAtCalo());
    h_ele_dPhiCl_propOut_all->Fill(gsfIter->deltaPhiSeedClusterTrackAtCalo());
    h_ele_HoE_all->Fill(gsfIter->hadronicOverEm());
    double d = gsfIter->vertex().x() * gsfIter->vertex().x() + gsfIter->vertex().y() * gsfIter->vertex().y();
    h_ele_TIP_all->Fill(sqrt(d));
    h_ele_vertexEta_all->Fill(gsfIter->eta());
    h_ele_vertexPt_all->Fill(gsfIter->pt());
    h_ele_Et_all->Fill(gsfIter->superCluster()->energy() / cosh(gsfIter->superCluster()->eta()));
    float enrj1 = gsfIter->superCluster()->energy();
    // mee
    for (reco::GsfElectronCollection::const_iterator gsfIter2 = gsfIter + 1; gsfIter2 != gsfElectrons->end();
         gsfIter2++) {
      math::XYZTLorentzVector p12 = (*gsfIter).p4() + (*gsfIter2).p4();
      float mee2 = p12.Dot(p12);
      float enrj2 = gsfIter2->superCluster()->energy();
      h_ele_mee_all->Fill(sqrt(mee2));
      h_ele_E2mnE1vsMee_all->Fill(sqrt(mee2), enrj2 - enrj1);
      if (gsfIter->ecalDrivenSeed() && gsfIter2->ecalDrivenSeed())
        h_ele_E2mnE1vsMee_egeg_all->Fill(sqrt(mee2), enrj2 - enrj1);
      if (gsfIter->charge() * gsfIter2->charge() < 0.) {
        h_ele_mee_os->Fill(sqrt(mee2));
        if (gsfIter->isEB() && gsfIter2->isEB())
          h_ele_mee_os_ebeb->Fill(sqrt(mee2));
        if ((gsfIter->isEB() && gsfIter2->isEE()) || (gsfIter->isEE() && gsfIter2->isEB()))
          h_ele_mee_os_ebee->Fill(sqrt(mee2));
        if (gsfIter->isEE() && gsfIter2->isEE())
          h_ele_mee_os_eeee->Fill(sqrt(mee2));
        if ((gsfIter->classification()==GsfElectron::GOLDEN && gsfIter2->classification()==GsfElectron::GOLDEN) ||
	     (gsfIter->classification()==GsfElectron::GOLDEN && gsfIter2->classification()==GsfElectron::BIGBREM) ||
	     //(gsfIter->classification()==GsfElectron::GOLDEN && gsfIter2->classification()==GsfElectron::OLDNARROW) ||
	     (gsfIter->classification()==GsfElectron::BIGBREM && gsfIter2->classification()==GsfElectron::GOLDEN) ||
	     (gsfIter->classification()==GsfElectron::BIGBREM && gsfIter2->classification()==GsfElectron::BIGBREM)/* ||
	     (gsfIter->classification()==GsfElectron::BIGBREM && gsfIter2->classification()==GsfElectron::OLDNARROW) ||
	     (gsfIter->classification()==GsfElectron::OLDNARROW && gsfIter2->classification()==GsfElectron::GOLDEN) ||
	     (gsfIter->classification()==GsfElectron::OLDNARROW && gsfIter2->classification()==GsfElectron::BIGBREM) ||
	     (gsfIter->classification()==GsfElectron::OLDNARROW && gsfIter2->classification()==GsfElectron::OLDNARROW)*/ )
	   {
          h_ele_mee_os_gg->Fill(sqrt(mee2));
        } else if ((gsfIter->classification() == GsfElectron::SHOWERING &&
                    gsfIter2->classification() == GsfElectron::SHOWERING) ||
                   (gsfIter->classification() == GsfElectron::SHOWERING && gsfIter2->isGap()) ||
                   (gsfIter->isGap() && gsfIter2->classification() == GsfElectron::SHOWERING) ||
                   (gsfIter->isGap() && gsfIter2->isGap())) {
          h_ele_mee_os_bb->Fill(sqrt(mee2));
        } else {
          h_ele_mee_os_gb->Fill(sqrt(mee2));
        }
      }
    }
  }

  int mcNum = 0, gamNum = 0, eleNum = 0;
  bool matchingID, matchingMotherID;

  // charge mis-ID
  for (reco::GenParticleCollection::const_iterator mcIter = genParticles->begin(); mcIter != genParticles->end();
       mcIter++) {
    // select requested matching gen particle
    matchingID = false;
    for (unsigned int i = 0; i < matchingIDs_.size(); i++)
      if (mcIter->pdgId() == matchingIDs_[i])
        matchingID = true;

    if (matchingID) {
      // select requested mother matching gen particle
      // always include single particle with no mother
      const Candidate *mother = mcIter->mother();
      matchingMotherID = false;
      for (unsigned int i = 0; i < matchingMotherIDs_.size(); i++)
        if ((mother == nullptr) || ((mother != nullptr) && mother->pdgId() == matchingMotherIDs_[i]))
          matchingMotherID = true;

      if (matchingMotherID) {
        if (mcIter->pt() > maxPt_ || std::abs(mcIter->eta()) > maxAbsEta_)
          continue;

        // suppress the endcaps
        //if (std::abs(mcIter->eta()) > 1.5) continue;
        // select central z
        //if ( std::abs(mcIter->production_vertex()->position().z())>50.) continue;

        // looking for the best matching gsf electron
        bool okGsfFound = false;
        double gsfOkRatio = 999999.;

        // find best matched electron
        reco::GsfElectron bestGsfElectron;
        for (reco::GsfElectronCollection::const_iterator gsfIter = gsfElectrons->begin();
             gsfIter != gsfElectrons->end();
             gsfIter++) {
          double dphi = gsfIter->phi() - mcIter->phi();
          if (std::abs(dphi) > CLHEP::pi)
            dphi = dphi < 0 ? (CLHEP::twopi) + dphi : dphi - CLHEP::twopi;
          double deltaR = sqrt(std::pow((gsfIter->eta() - mcIter->eta()), 2) + std::pow(dphi, 2));
          if (deltaR < deltaR_) {
            double mc_charge = mcIter->pdgId() == 11 ? -1. : 1.;
            h_ele_ChargeMnChargeTrue->Fill(std::abs(gsfIter->charge() - mc_charge));
            // require here a charge mismatch
            if (((mcIter->pdgId() == 11) && (gsfIter->charge() > 0.)) ||
                ((mcIter->pdgId() == -11) && (gsfIter->charge() < 0.))) {
              double tmpGsfRatio = gsfIter->p() / mcIter->p();
              if (std::abs(tmpGsfRatio - 1) < std::abs(gsfOkRatio - 1)) {
                gsfOkRatio = tmpGsfRatio;
                bestGsfElectron = *gsfIter;
                okGsfFound = true;
              }
            }
          }
        }  // loop over rec ele to look for the best one

        // analysis when the mc track is found
        if (okGsfFound) {
          // generated distributions for matched electrons
          h_ele_simPt_matched_qmisid->Fill(mcIter->pt());
          h_ele_simPhi_matched_qmisid->Fill(mcIter->phi());
          h_ele_simAbsEta_matched_qmisid->Fill(std::abs(mcIter->eta()));
          h_ele_simEta_matched_qmisid->Fill(mcIter->eta());
          h_ele_simZ_matched_qmisid->Fill(mcIter->vz());
        }
      }
    }
  }

  // association mc-reco
  for (reco::GenParticleCollection::const_iterator mcIter = genParticles->begin(); mcIter != genParticles->end();
       mcIter++) {
    // number of mc particles
    mcNum++;

    // counts photons
    if (mcIter->pdgId() == 22) {
      gamNum++;
    }

    // select requested matching gen particle
    matchingID = false;
    for (unsigned int i = 0; i < matchingIDs_.size(); i++)
      if (mcIter->pdgId() == matchingIDs_[i])
        matchingID = true;

    if (matchingID) {
      // select requested mother matching gen particle
      // always include single particle with no mother
      const Candidate *mother = mcIter->mother();
      matchingMotherID = false;
      for (unsigned int i = 0; i < matchingMotherIDs_.size(); i++)
        if ((mother == nullptr) || ((mother != nullptr) && mother->pdgId() == matchingMotherIDs_[i]))
          matchingMotherID = true;

      if (matchingMotherID) {
        if (mcIter->pt() > maxPt_ || std::abs(mcIter->eta()) > maxAbsEta_)
          continue;

        // suppress the endcaps
        //if (std::abs(mcIter->eta()) > 1.5) continue;
        // select central z
        //if ( std::abs(mcIter->production_vertex()->position().z())>50.) continue;

        eleNum++;
        h_simEta->Fill(mcIter->eta());
        h_simAbsEta->Fill(std::abs(mcIter->eta()));
        h_simP->Fill(mcIter->p());
        h_simPt->Fill(mcIter->pt());
        h_simPhi->Fill(mcIter->phi());
        h_simZ->Fill(mcIter->vz());
        h_simPtEta->Fill(mcIter->eta(), mcIter->pt());

        // looking for the best matching gsf electron
        bool okGsfFound = false;
        double gsfOkRatio = 999999.;

        // find best matched electron
        reco::GsfElectron bestGsfElectron;
        for (reco::GsfElectronCollection::const_iterator gsfIter = gsfElectrons->begin();
             gsfIter != gsfElectrons->end();
             gsfIter++) {
          double dphi = gsfIter->phi() - mcIter->phi();
          if (std::abs(dphi) > CLHEP::pi)
            dphi = dphi < 0 ? (CLHEP::twopi) + dphi : dphi - CLHEP::twopi;
          double deltaR = sqrt(std::pow((gsfIter->eta() - mcIter->eta()), 2) + std::pow(dphi, 2));
          if (deltaR < deltaR_) {
            if (((mcIter->pdgId() == 11) && (gsfIter->charge() < 0.)) ||
                ((mcIter->pdgId() == -11) && (gsfIter->charge() > 0.))) {
              double tmpGsfRatio = gsfIter->p() / mcIter->p();
              if (std::abs(tmpGsfRatio - 1) < std::abs(gsfOkRatio - 1)) {
                gsfOkRatio = tmpGsfRatio;
                bestGsfElectron = *gsfIter;
                okGsfFound = true;
              }
            }
          }
        }  // loop over rec ele to look for the best one

        // analysis when the mc track is found
        if (okGsfFound) {
          // electron related distributions
          h_ele_charge->Fill(bestGsfElectron.charge());
          h_ele_chargeVsEta->Fill(bestGsfElectron.eta(), bestGsfElectron.charge());
          h_ele_chargeVsPhi->Fill(bestGsfElectron.phi(), bestGsfElectron.charge());
          h_ele_chargeVsPt->Fill(bestGsfElectron.pt(), bestGsfElectron.charge());
          h_ele_vertexP->Fill(bestGsfElectron.p());
          h_ele_vertexPt->Fill(bestGsfElectron.pt());
          h_ele_Et->Fill(bestGsfElectron.superCluster()->energy() / cosh(bestGsfElectron.superCluster()->eta()));
          h_ele_vertexPtVsEta->Fill(bestGsfElectron.eta(), bestGsfElectron.pt());
          h_ele_vertexPtVsPhi->Fill(bestGsfElectron.phi(), bestGsfElectron.pt());
          h_ele_vertexEta->Fill(bestGsfElectron.eta());
          // generated distributions for matched electrons
          h_ele_simPt_matched->Fill(mcIter->pt());
          h_ele_simPhi_matched->Fill(mcIter->phi());
          h_ele_simAbsEta_matched->Fill(std::abs(mcIter->eta()));
          h_ele_simEta_matched->Fill(mcIter->eta());
          h_ele_simPtEta_matched->Fill(mcIter->eta(), mcIter->pt());
          h_ele_vertexEtaVsPhi->Fill(bestGsfElectron.phi(), bestGsfElectron.eta());
          h_ele_vertexPhi->Fill(bestGsfElectron.phi());
          h_ele_vertexX->Fill(bestGsfElectron.vertex().x());
          h_ele_vertexY->Fill(bestGsfElectron.vertex().y());
          h_ele_vertexZ->Fill(bestGsfElectron.vertex().z());
          h_ele_simZ_matched->Fill(mcIter->vz());
          double d = (bestGsfElectron.vertex().x() - mcIter->vx()) * (bestGsfElectron.vertex().x() - mcIter->vx()) +
                     (bestGsfElectron.vertex().y() - mcIter->vy()) * (bestGsfElectron.vertex().y() - mcIter->vy());
          d = sqrt(d);
          h_ele_vertexTIP->Fill(d);
          h_ele_vertexTIPVsEta->Fill(bestGsfElectron.eta(), d);
          h_ele_vertexTIPVsPhi->Fill(bestGsfElectron.phi(), d);
          h_ele_vertexTIPVsPt->Fill(bestGsfElectron.pt(), d);
          h_ele_EtaMnEtaTrue->Fill(bestGsfElectron.eta() - mcIter->eta());
          if (bestGsfElectron.isEB())
            h_ele_EtaMnEtaTrue_barrel->Fill(bestGsfElectron.eta() - mcIter->eta());
          if (bestGsfElectron.isEE())
            h_ele_EtaMnEtaTrue_endcaps->Fill(bestGsfElectron.eta() - mcIter->eta());
          h_ele_EtaMnEtaTrueVsEta->Fill(bestGsfElectron.eta(), bestGsfElectron.eta() - mcIter->eta());
          h_ele_EtaMnEtaTrueVsPhi->Fill(bestGsfElectron.phi(), bestGsfElectron.eta() - mcIter->eta());
          h_ele_EtaMnEtaTrueVsPt->Fill(bestGsfElectron.pt(), bestGsfElectron.eta() - mcIter->eta());
          h_ele_PhiMnPhiTrue->Fill(bestGsfElectron.phi() - mcIter->phi());
          if (bestGsfElectron.isEB())
            h_ele_PhiMnPhiTrue_barrel->Fill(bestGsfElectron.phi() - mcIter->phi());
          if (bestGsfElectron.isEE())
            h_ele_PhiMnPhiTrue_endcaps->Fill(bestGsfElectron.phi() - mcIter->phi());
          h_ele_PhiMnPhiTrue2->Fill(bestGsfElectron.phi() - mcIter->phi());
          h_ele_PhiMnPhiTrueVsEta->Fill(bestGsfElectron.eta(), bestGsfElectron.phi() - mcIter->phi());
          h_ele_PhiMnPhiTrueVsPhi->Fill(bestGsfElectron.phi(), bestGsfElectron.phi() - mcIter->phi());
          h_ele_PhiMnPhiTrueVsPt->Fill(bestGsfElectron.pt(), bestGsfElectron.phi() - mcIter->phi());
          h_ele_PoPtrue->Fill(bestGsfElectron.p() / mcIter->p());
          h_ele_PtoPttrue->Fill(bestGsfElectron.pt() / mcIter->pt());
          h_ele_PoPtrueVsEta->Fill(bestGsfElectron.eta(), bestGsfElectron.p() / mcIter->p());
          h_ele_PoPtrueVsPhi->Fill(bestGsfElectron.phi(), bestGsfElectron.p() / mcIter->p());
          h_ele_PoPtrueVsPt->Fill(bestGsfElectron.py(), bestGsfElectron.p() / mcIter->p());
          if (bestGsfElectron.isEB())
            h_ele_PoPtrue_barrel->Fill(bestGsfElectron.p() / mcIter->p());
          if (bestGsfElectron.isEE())
            h_ele_PoPtrue_endcaps->Fill(bestGsfElectron.p() / mcIter->p());
          if (bestGsfElectron.isEB() && bestGsfElectron.classification() == GsfElectron::GOLDEN)
            h_ele_PoPtrue_golden_barrel->Fill(bestGsfElectron.p() / mcIter->p());
          if (bestGsfElectron.isEE() && bestGsfElectron.classification() == GsfElectron::GOLDEN)
            h_ele_PoPtrue_golden_endcaps->Fill(bestGsfElectron.p() / mcIter->p());
          if (bestGsfElectron.isEB() && bestGsfElectron.classification() == GsfElectron::SHOWERING)
            h_ele_PoPtrue_showering_barrel->Fill(bestGsfElectron.p() / mcIter->p());
          if (bestGsfElectron.isEE() && bestGsfElectron.classification() == GsfElectron::SHOWERING)
            h_ele_PoPtrue_showering_endcaps->Fill(bestGsfElectron.p() / mcIter->p());
          if (bestGsfElectron.isEB())
            h_ele_PtoPttrue_barrel->Fill(bestGsfElectron.pt() / mcIter->pt());
          if (bestGsfElectron.isEE())
            h_ele_PtoPttrue_endcaps->Fill(bestGsfElectron.pt() / mcIter->pt());

          // supercluster related distributions
          reco::SuperClusterRef sclRef = bestGsfElectron.superCluster();
          if (!bestGsfElectron.ecalDrivenSeed() && bestGsfElectron.trackerDrivenSeed())
            sclRef = bestGsfElectron.parentSuperCluster();
          histSclEn_->Fill(sclRef->energy());
          double R = TMath::Sqrt(sclRef->x() * sclRef->x() + sclRef->y() * sclRef->y() + sclRef->z() * sclRef->z());
          double Rt = TMath::Sqrt(sclRef->x() * sclRef->x() + sclRef->y() * sclRef->y());
          histSclEt_->Fill(sclRef->energy() * (Rt / R));
          histSclEtVsEta_->Fill(sclRef->eta(), sclRef->energy() * (Rt / R));
          histSclEtVsPhi_->Fill(sclRef->phi(), sclRef->energy() * (Rt / R));
          if (bestGsfElectron.isEB())
            histSclEoEtrue_barrel->Fill(sclRef->energy() / mcIter->p());
          if (bestGsfElectron.isEE())
            histSclEoEtrue_endcaps->Fill(sclRef->energy() / mcIter->p());
          if (bestGsfElectron.isEB() && bestGsfElectron.ecalDrivenSeed())
            histSclEoEtrue_barrel_eg->Fill(sclRef->energy() / mcIter->p());
          if (bestGsfElectron.isEE() && bestGsfElectron.ecalDrivenSeed())
            histSclEoEtrue_endcaps_eg->Fill(sclRef->energy() / mcIter->p());
          if (bestGsfElectron.isEB() && bestGsfElectron.isEBEtaGap())
            histSclEoEtrue_barrel_etagap->Fill(sclRef->energy() / mcIter->p());
          if (bestGsfElectron.isEB() && bestGsfElectron.isEBPhiGap())
            histSclEoEtrue_barrel_phigap->Fill(sclRef->energy() / mcIter->p());
          if (bestGsfElectron.isEBEEGap())
            histSclEoEtrue_ebeegap->Fill(sclRef->energy() / mcIter->p());
          //if (bestGsfElectron.isEE())  histSclEoEtrue_endcaps->Fill(sclRef->energy()/mcIter->p());
          if (bestGsfElectron.isEE() && bestGsfElectron.isEEDeeGap())
            histSclEoEtrue_endcaps_deegap->Fill(sclRef->energy() / mcIter->p());
          if (bestGsfElectron.isEE() && bestGsfElectron.isEERingGap())
            histSclEoEtrue_endcaps_ringgap->Fill(sclRef->energy() / mcIter->p());
          if (bestGsfElectron.isEB())
            histSclEoEtrue_barrel_new->Fill(sclRef->energy() / mcIter->p());
          if (bestGsfElectron.isEE())
            histSclEoEtrue_endcaps_new->Fill(sclRef->energy() / mcIter->p());
          if (bestGsfElectron.isEB() && bestGsfElectron.ecalDrivenSeed())
            histSclEoEtrue_barrel_eg_new->Fill(sclRef->energy() / mcIter->p());
          if (bestGsfElectron.isEE() && bestGsfElectron.ecalDrivenSeed())
            histSclEoEtrue_endcaps_eg_new->Fill(sclRef->energy() / mcIter->p());
          if (bestGsfElectron.isEB() && bestGsfElectron.isEBEtaGap())
            histSclEoEtrue_barrel_etagap_new->Fill(sclRef->energy() / mcIter->p());
          if (bestGsfElectron.isEB() && bestGsfElectron.isEBPhiGap())
            histSclEoEtrue_barrel_phigap_new->Fill(sclRef->energy() / mcIter->p());
          if (bestGsfElectron.isEBEEGap())
            histSclEoEtrue_ebeegap_new->Fill(sclRef->energy() / mcIter->p());
          //if (bestGsfElectron.isEE())  histSclEoEtrue_endcaps_new->Fill(sclRef->energy()/mcIter->p());
          if (bestGsfElectron.isEE() && bestGsfElectron.isEEDeeGap())
            histSclEoEtrue_endcaps_deegap_new->Fill(sclRef->energy() / mcIter->p());
          if (bestGsfElectron.isEE() && bestGsfElectron.isEERingGap())
            histSclEoEtrue_endcaps_ringgap_new->Fill(sclRef->energy() / mcIter->p());
          histSclEta_->Fill(sclRef->eta());
          histSclEtaVsPhi_->Fill(sclRef->phi(), sclRef->eta());
          histSclPhi_->Fill(sclRef->phi());
          histSclSigEtaEta_->Fill(bestGsfElectron.scSigmaEtaEta());
          if (bestGsfElectron.isEB())
            histSclSigEtaEta_barrel_->Fill(bestGsfElectron.scSigmaEtaEta());
          if (bestGsfElectron.isEE())
            histSclSigEtaEta_endcaps_->Fill(bestGsfElectron.scSigmaEtaEta());
          histSclSigIEtaIEta_->Fill(bestGsfElectron.scSigmaIEtaIEta());
          if (bestGsfElectron.isEB())
            histSclSigIEtaIEta_barrel_->Fill(bestGsfElectron.scSigmaIEtaIEta());
          if (bestGsfElectron.isEE())
            histSclSigIEtaIEta_endcaps_->Fill(bestGsfElectron.scSigmaIEtaIEta());
          histSclE1x5_->Fill(bestGsfElectron.scE1x5());
          if (bestGsfElectron.isEB())
            histSclE1x5_barrel_->Fill(bestGsfElectron.scE1x5());
          if (bestGsfElectron.isEE())
            histSclE1x5_endcaps_->Fill(bestGsfElectron.scE1x5());
          histSclE2x5max_->Fill(bestGsfElectron.scE2x5Max());
          if (bestGsfElectron.isEB())
            histSclE2x5max_barrel_->Fill(bestGsfElectron.scE2x5Max());
          if (bestGsfElectron.isEE())
            histSclE2x5max_endcaps_->Fill(bestGsfElectron.scE2x5Max());
          histSclE5x5_->Fill(bestGsfElectron.scE5x5());
          if (bestGsfElectron.isEB())
            histSclE5x5_barrel_->Fill(bestGsfElectron.scE5x5());
          if (bestGsfElectron.isEE())
            histSclE5x5_endcaps_->Fill(bestGsfElectron.scE5x5());
          if (bestGsfElectron.ecalDrivenSeed())
            histSclSigIEtaIEta_eg_->Fill(bestGsfElectron.scSigmaIEtaIEta());
          if (bestGsfElectron.isEB() && bestGsfElectron.ecalDrivenSeed())
            histSclSigIEtaIEta_eg_barrel_->Fill(bestGsfElectron.scSigmaIEtaIEta());
          if (bestGsfElectron.isEE() && bestGsfElectron.ecalDrivenSeed())
            histSclSigIEtaIEta_eg_endcaps_->Fill(bestGsfElectron.scSigmaIEtaIEta());
          if (bestGsfElectron.ecalDrivenSeed())
            histSclE1x5_eg_->Fill(bestGsfElectron.scE1x5());
          if (bestGsfElectron.isEB() && bestGsfElectron.ecalDrivenSeed())
            histSclE1x5_eg_barrel_->Fill(bestGsfElectron.scE1x5());
          if (bestGsfElectron.isEE() && bestGsfElectron.ecalDrivenSeed())
            histSclE1x5_eg_endcaps_->Fill(bestGsfElectron.scE1x5());
          if (bestGsfElectron.ecalDrivenSeed())
            histSclE2x5max_eg_->Fill(bestGsfElectron.scE2x5Max());
          if (bestGsfElectron.isEB() && bestGsfElectron.ecalDrivenSeed())
            histSclE2x5max_eg_barrel_->Fill(bestGsfElectron.scE2x5Max());
          if (bestGsfElectron.isEE() && bestGsfElectron.ecalDrivenSeed())
            histSclE2x5max_eg_endcaps_->Fill(bestGsfElectron.scE2x5Max());
          if (bestGsfElectron.ecalDrivenSeed())
            histSclE5x5_eg_->Fill(bestGsfElectron.scE5x5());
          if (bestGsfElectron.isEB() && bestGsfElectron.ecalDrivenSeed())
            histSclE5x5_eg_barrel_->Fill(bestGsfElectron.scE5x5());
          if (bestGsfElectron.isEE() && bestGsfElectron.ecalDrivenSeed())
            histSclE5x5_eg_endcaps_->Fill(bestGsfElectron.scE5x5());
          float pfEnergy = 0., egEnergy = 0.;
          if (!bestGsfElectron.superCluster().isNull())
            egEnergy = bestGsfElectron.superCluster()->energy();
          if (!bestGsfElectron.parentSuperCluster().isNull())
            pfEnergy = bestGsfElectron.parentSuperCluster()->energy();
          histSclEoEtruePfVsEg->Fill(egEnergy / mcIter->p(), pfEnergy / mcIter->p());

          // track related distributions
          h_ele_ambiguousTracks->Fill(bestGsfElectron.ambiguousGsfTracksSize());
          h_ele_ambiguousTracksVsEta->Fill(bestGsfElectron.eta(), bestGsfElectron.ambiguousGsfTracksSize());
          h_ele_ambiguousTracksVsPhi->Fill(bestGsfElectron.phi(), bestGsfElectron.ambiguousGsfTracksSize());
          h_ele_ambiguousTracksVsPt->Fill(bestGsfElectron.pt(), bestGsfElectron.ambiguousGsfTracksSize());
          if (!readAOD_) {  // track extra does not exist in AOD
            h_ele_foundHits->Fill(bestGsfElectron.gsfTrack()->numberOfValidHits());
            if (bestGsfElectron.isEB())
              h_ele_foundHits_barrel->Fill(bestGsfElectron.gsfTrack()->numberOfValidHits());
            if (bestGsfElectron.isEE())
              h_ele_foundHits_endcaps->Fill(bestGsfElectron.gsfTrack()->numberOfValidHits());
            h_ele_foundHitsVsEta->Fill(bestGsfElectron.eta(), bestGsfElectron.gsfTrack()->numberOfValidHits());
            h_ele_foundHitsVsPhi->Fill(bestGsfElectron.phi(), bestGsfElectron.gsfTrack()->numberOfValidHits());
            h_ele_foundHitsVsPt->Fill(bestGsfElectron.pt(), bestGsfElectron.gsfTrack()->numberOfValidHits());
            h_ele_lostHits->Fill(bestGsfElectron.gsfTrack()->numberOfLostHits());
            if (bestGsfElectron.isEB())
              h_ele_lostHits_barrel->Fill(bestGsfElectron.gsfTrack()->numberOfLostHits());
            if (bestGsfElectron.isEE())
              h_ele_lostHits_endcaps->Fill(bestGsfElectron.gsfTrack()->numberOfLostHits());
            h_ele_lostHitsVsEta->Fill(bestGsfElectron.eta(), bestGsfElectron.gsfTrack()->numberOfLostHits());
            h_ele_lostHitsVsPhi->Fill(bestGsfElectron.phi(), bestGsfElectron.gsfTrack()->numberOfLostHits());
            h_ele_lostHitsVsPt->Fill(bestGsfElectron.pt(), bestGsfElectron.gsfTrack()->numberOfLostHits());
            h_ele_chi2->Fill(bestGsfElectron.gsfTrack()->normalizedChi2());
            if (bestGsfElectron.isEB())
              h_ele_chi2_barrel->Fill(bestGsfElectron.gsfTrack()->normalizedChi2());
            if (bestGsfElectron.isEE())
              h_ele_chi2_endcaps->Fill(bestGsfElectron.gsfTrack()->normalizedChi2());
            h_ele_chi2VsEta->Fill(bestGsfElectron.eta(), bestGsfElectron.gsfTrack()->normalizedChi2());
            h_ele_chi2VsPhi->Fill(bestGsfElectron.phi(), bestGsfElectron.gsfTrack()->normalizedChi2());
            h_ele_chi2VsPt->Fill(bestGsfElectron.pt(), bestGsfElectron.gsfTrack()->normalizedChi2());
          }
          // from gsf track interface, hence using mean
          if (!readAOD_) {  // track extra does not exist in AOD
            h_ele_PinMnPout->Fill(bestGsfElectron.gsfTrack()->innerMomentum().R() -
                                  bestGsfElectron.gsfTrack()->outerMomentum().R());
            h_ele_outerP->Fill(bestGsfElectron.gsfTrack()->outerMomentum().R());
            h_ele_outerPt->Fill(bestGsfElectron.gsfTrack()->outerMomentum().Rho());
          }
          // from electron interface, hence using mode
          h_ele_PinMnPout_mode->Fill(bestGsfElectron.trackMomentumAtVtx().R() - bestGsfElectron.trackMomentumOut().R());
          h_ele_PinMnPoutVsEta_mode->Fill(
              bestGsfElectron.eta(), bestGsfElectron.trackMomentumAtVtx().R() - bestGsfElectron.trackMomentumOut().R());
          h_ele_PinMnPoutVsPhi_mode->Fill(
              bestGsfElectron.phi(), bestGsfElectron.trackMomentumAtVtx().R() - bestGsfElectron.trackMomentumOut().R());
          h_ele_PinMnPoutVsPt_mode->Fill(
              bestGsfElectron.pt(), bestGsfElectron.trackMomentumAtVtx().R() - bestGsfElectron.trackMomentumOut().R());
          h_ele_PinMnPoutVsE_mode->Fill(
              bestGsfElectron.caloEnergy(),
              bestGsfElectron.trackMomentumAtVtx().R() - bestGsfElectron.trackMomentumOut().R());
          if (!readAOD_)  // track extra does not exist in AOD
            h_ele_PinMnPoutVsChi2_mode->Fill(
                bestGsfElectron.gsfTrack()->normalizedChi2(),
                bestGsfElectron.trackMomentumAtVtx().R() - bestGsfElectron.trackMomentumOut().R());
          h_ele_outerP_mode->Fill(bestGsfElectron.trackMomentumOut().R());
          h_ele_outerPVsEta_mode->Fill(bestGsfElectron.eta(), bestGsfElectron.trackMomentumOut().R());
          h_ele_outerPt_mode->Fill(bestGsfElectron.trackMomentumOut().Rho());
          h_ele_outerPtVsEta_mode->Fill(bestGsfElectron.eta(), bestGsfElectron.trackMomentumOut().Rho());
          h_ele_outerPtVsPhi_mode->Fill(bestGsfElectron.phi(), bestGsfElectron.trackMomentumOut().Rho());
          h_ele_outerPtVsPt_mode->Fill(bestGsfElectron.pt(), bestGsfElectron.trackMomentumOut().Rho());

          if (!readAOD_) {  // track extra does not exist in AOD
            edm::RefToBase<TrajectorySeed> seed = bestGsfElectron.gsfTrack()->extra()->seedRef();
            ElectronSeedRef elseed = seed.castTo<ElectronSeedRef>();
            h_ele_seed_dphi2_->Fill(elseed->dPhiNeg(1));
            h_ele_seed_dphi2VsEta_->Fill(bestGsfElectron.eta(), elseed->dPhiNeg(1));
            h_ele_seed_dphi2VsPt_->Fill(bestGsfElectron.pt(), elseed->dPhiNeg(1));
            h_ele_seed_drz2_->Fill(elseed->dRZNeg(1));
            h_ele_seed_drz2VsEta_->Fill(bestGsfElectron.eta(), elseed->dRZNeg(1));
            h_ele_seed_drz2VsPt_->Fill(bestGsfElectron.pt(), elseed->dRZNeg(1));
            h_ele_seed_subdet2_->Fill(elseed->subDet(1));
          }
          // match distributions
          h_ele_EoP->Fill(bestGsfElectron.eSuperClusterOverP());
          if (bestGsfElectron.ecalDrivenSeed())
            h_ele_EoP_eg->Fill(bestGsfElectron.eSuperClusterOverP());
          if (bestGsfElectron.isEB())
            h_ele_EoP_barrel->Fill(bestGsfElectron.eSuperClusterOverP());
          if (bestGsfElectron.isEB() && bestGsfElectron.ecalDrivenSeed())
            h_ele_EoP_eg_barrel->Fill(bestGsfElectron.eSuperClusterOverP());
          if (bestGsfElectron.isEE())
            h_ele_EoP_endcaps->Fill(bestGsfElectron.eSuperClusterOverP());
          if (bestGsfElectron.isEE() && bestGsfElectron.ecalDrivenSeed())
            h_ele_EoP_eg_endcaps->Fill(bestGsfElectron.eSuperClusterOverP());
          h_ele_EoPVsEta->Fill(bestGsfElectron.eta(), bestGsfElectron.eSuperClusterOverP());
          h_ele_EoPVsPhi->Fill(bestGsfElectron.phi(), bestGsfElectron.eSuperClusterOverP());
          h_ele_EoPVsE->Fill(bestGsfElectron.caloEnergy(), bestGsfElectron.eSuperClusterOverP());
          h_ele_EseedOP->Fill(bestGsfElectron.eSeedClusterOverP());
          if (bestGsfElectron.ecalDrivenSeed())
            h_ele_EseedOP_eg->Fill(bestGsfElectron.eSeedClusterOverP());
          if (bestGsfElectron.isEB())
            h_ele_EseedOP_barrel->Fill(bestGsfElectron.eSeedClusterOverP());
          if (bestGsfElectron.isEB() && bestGsfElectron.ecalDrivenSeed())
            h_ele_EseedOP_eg_barrel->Fill(bestGsfElectron.eSeedClusterOverP());
          if (bestGsfElectron.isEE())
            h_ele_EseedOP_endcaps->Fill(bestGsfElectron.eSeedClusterOverP());
          if (bestGsfElectron.isEE() && bestGsfElectron.ecalDrivenSeed())
            h_ele_EseedOP_eg_endcaps->Fill(bestGsfElectron.eSeedClusterOverP());
          h_ele_EseedOPVsEta->Fill(bestGsfElectron.eta(), bestGsfElectron.eSeedClusterOverP());
          h_ele_EseedOPVsPhi->Fill(bestGsfElectron.phi(), bestGsfElectron.eSeedClusterOverP());
          h_ele_EseedOPVsE->Fill(bestGsfElectron.caloEnergy(), bestGsfElectron.eSeedClusterOverP());
          h_ele_EoPout->Fill(bestGsfElectron.eSeedClusterOverPout());
          if (bestGsfElectron.ecalDrivenSeed())
            h_ele_EoPout_eg->Fill(bestGsfElectron.eSeedClusterOverPout());
          if (bestGsfElectron.isEB())
            h_ele_EoPout_barrel->Fill(bestGsfElectron.eSeedClusterOverPout());
          if (bestGsfElectron.isEB() && bestGsfElectron.ecalDrivenSeed())
            h_ele_EoPout_eg_barrel->Fill(bestGsfElectron.eSeedClusterOverPout());
          if (bestGsfElectron.isEE())
            h_ele_EoPout_endcaps->Fill(bestGsfElectron.eSeedClusterOverPout());
          if (bestGsfElectron.isEE() && bestGsfElectron.ecalDrivenSeed())
            h_ele_EoPout_eg_endcaps->Fill(bestGsfElectron.eSeedClusterOverPout());
          h_ele_EoPoutVsEta->Fill(bestGsfElectron.eta(), bestGsfElectron.eSeedClusterOverPout());
          h_ele_EoPoutVsPhi->Fill(bestGsfElectron.phi(), bestGsfElectron.eSeedClusterOverPout());
          h_ele_EoPoutVsE->Fill(bestGsfElectron.caloEnergy(), bestGsfElectron.eSeedClusterOverPout());
          h_ele_EeleOPout->Fill(bestGsfElectron.eEleClusterOverPout());
          if (bestGsfElectron.ecalDrivenSeed())
            h_ele_EeleOPout_eg->Fill(bestGsfElectron.eEleClusterOverPout());
          if (bestGsfElectron.isEB())
            h_ele_EeleOPout_barrel->Fill(bestGsfElectron.eEleClusterOverPout());
          if (bestGsfElectron.isEB() && bestGsfElectron.ecalDrivenSeed())
            h_ele_EeleOPout_eg_barrel->Fill(bestGsfElectron.eEleClusterOverPout());
          if (bestGsfElectron.isEE())
            h_ele_EeleOPout_endcaps->Fill(bestGsfElectron.eEleClusterOverPout());
          if (bestGsfElectron.isEE() && bestGsfElectron.ecalDrivenSeed())
            h_ele_EeleOPout_eg_endcaps->Fill(bestGsfElectron.eEleClusterOverPout());
          h_ele_EeleOPoutVsEta->Fill(bestGsfElectron.eta(), bestGsfElectron.eEleClusterOverPout());
          h_ele_EeleOPoutVsPhi->Fill(bestGsfElectron.phi(), bestGsfElectron.eEleClusterOverPout());
          h_ele_EeleOPoutVsE->Fill(bestGsfElectron.caloEnergy(), bestGsfElectron.eEleClusterOverPout());
          h_ele_dEtaSc_propVtx->Fill(bestGsfElectron.deltaEtaSuperClusterTrackAtVtx());
          if (bestGsfElectron.ecalDrivenSeed())
            h_ele_dEtaSc_propVtx_eg->Fill(bestGsfElectron.deltaEtaSuperClusterTrackAtVtx());
          if (bestGsfElectron.isEB())
            h_ele_dEtaSc_propVtx_barrel->Fill(bestGsfElectron.deltaEtaSuperClusterTrackAtVtx());
          if (bestGsfElectron.isEB() && bestGsfElectron.ecalDrivenSeed())
            h_ele_dEtaSc_propVtx_eg_barrel->Fill(bestGsfElectron.deltaEtaSuperClusterTrackAtVtx());
          if (bestGsfElectron.isEE())
            h_ele_dEtaSc_propVtx_endcaps->Fill(bestGsfElectron.deltaEtaSuperClusterTrackAtVtx());
          if (bestGsfElectron.isEE() && bestGsfElectron.ecalDrivenSeed())
            h_ele_dEtaSc_propVtx_eg_endcaps->Fill(bestGsfElectron.deltaEtaSuperClusterTrackAtVtx());
          h_ele_dEtaScVsEta_propVtx->Fill(bestGsfElectron.eta(), bestGsfElectron.deltaEtaSuperClusterTrackAtVtx());
          h_ele_dEtaScVsPhi_propVtx->Fill(bestGsfElectron.phi(), bestGsfElectron.deltaEtaSuperClusterTrackAtVtx());
          h_ele_dEtaScVsPt_propVtx->Fill(bestGsfElectron.pt(), bestGsfElectron.deltaEtaSuperClusterTrackAtVtx());
          h_ele_dPhiSc_propVtx->Fill(bestGsfElectron.deltaPhiSuperClusterTrackAtVtx());
          if (bestGsfElectron.ecalDrivenSeed())
            h_ele_dPhiSc_propVtx_eg->Fill(bestGsfElectron.deltaPhiSuperClusterTrackAtVtx());
          if (bestGsfElectron.isEB())
            h_ele_dPhiSc_propVtx_barrel->Fill(bestGsfElectron.deltaPhiSuperClusterTrackAtVtx());
          if (bestGsfElectron.isEB() && bestGsfElectron.ecalDrivenSeed())
            h_ele_dPhiSc_propVtx_eg_barrel->Fill(bestGsfElectron.deltaPhiSuperClusterTrackAtVtx());
          if (bestGsfElectron.isEE())
            h_ele_dPhiSc_propVtx_endcaps->Fill(bestGsfElectron.deltaPhiSuperClusterTrackAtVtx());
          if (bestGsfElectron.isEE() && bestGsfElectron.ecalDrivenSeed())
            h_ele_dPhiSc_propVtx_eg_endcaps->Fill(bestGsfElectron.deltaPhiSuperClusterTrackAtVtx());
          h_ele_dPhiScVsEta_propVtx->Fill(bestGsfElectron.eta(), bestGsfElectron.deltaPhiSuperClusterTrackAtVtx());
          h_ele_dPhiScVsPhi_propVtx->Fill(bestGsfElectron.phi(), bestGsfElectron.deltaPhiSuperClusterTrackAtVtx());
          h_ele_dPhiScVsPt_propVtx->Fill(bestGsfElectron.pt(), bestGsfElectron.deltaPhiSuperClusterTrackAtVtx());
          h_ele_dEtaCl_propOut->Fill(bestGsfElectron.deltaEtaSeedClusterTrackAtCalo());
          if (bestGsfElectron.ecalDrivenSeed())
            h_ele_dEtaCl_propOut_eg->Fill(bestGsfElectron.deltaEtaSeedClusterTrackAtCalo());
          if (bestGsfElectron.isEB())
            h_ele_dEtaCl_propOut_barrel->Fill(bestGsfElectron.deltaEtaSeedClusterTrackAtCalo());
          if (bestGsfElectron.isEB() && bestGsfElectron.ecalDrivenSeed())
            h_ele_dEtaCl_propOut_eg_barrel->Fill(bestGsfElectron.deltaEtaSeedClusterTrackAtCalo());
          if (bestGsfElectron.isEE())
            h_ele_dEtaCl_propOut_endcaps->Fill(bestGsfElectron.deltaEtaSeedClusterTrackAtCalo());
          if (bestGsfElectron.isEE() && bestGsfElectron.ecalDrivenSeed())
            h_ele_dEtaCl_propOut_eg_endcaps->Fill(bestGsfElectron.deltaEtaSeedClusterTrackAtCalo());
          h_ele_dEtaClVsEta_propOut->Fill(bestGsfElectron.eta(), bestGsfElectron.deltaEtaSeedClusterTrackAtCalo());
          h_ele_dEtaClVsPhi_propOut->Fill(bestGsfElectron.phi(), bestGsfElectron.deltaEtaSeedClusterTrackAtCalo());
          h_ele_dEtaClVsPt_propOut->Fill(bestGsfElectron.pt(), bestGsfElectron.deltaEtaSeedClusterTrackAtCalo());
          h_ele_dPhiCl_propOut->Fill(bestGsfElectron.deltaPhiSeedClusterTrackAtCalo());
          if (bestGsfElectron.ecalDrivenSeed())
            h_ele_dPhiCl_propOut_eg->Fill(bestGsfElectron.deltaPhiSeedClusterTrackAtCalo());
          if (bestGsfElectron.isEB())
            h_ele_dPhiCl_propOut_barrel->Fill(bestGsfElectron.deltaPhiSeedClusterTrackAtCalo());
          if (bestGsfElectron.isEB() && bestGsfElectron.ecalDrivenSeed())
            h_ele_dPhiCl_propOut_eg_barrel->Fill(bestGsfElectron.deltaPhiSeedClusterTrackAtCalo());
          if (bestGsfElectron.isEE())
            h_ele_dPhiCl_propOut_endcaps->Fill(bestGsfElectron.deltaPhiSeedClusterTrackAtCalo());
          if (bestGsfElectron.isEE() && bestGsfElectron.ecalDrivenSeed())
            h_ele_dPhiCl_propOut_eg_endcaps->Fill(bestGsfElectron.deltaPhiSeedClusterTrackAtCalo());
          h_ele_dPhiClVsEta_propOut->Fill(bestGsfElectron.eta(), bestGsfElectron.deltaPhiSeedClusterTrackAtCalo());
          h_ele_dPhiClVsPhi_propOut->Fill(bestGsfElectron.phi(), bestGsfElectron.deltaPhiSeedClusterTrackAtCalo());
          h_ele_dPhiClVsPt_propOut->Fill(bestGsfElectron.pt(), bestGsfElectron.deltaPhiSeedClusterTrackAtCalo());
          h_ele_dEtaEleCl_propOut->Fill(bestGsfElectron.deltaEtaEleClusterTrackAtCalo());
          if (bestGsfElectron.ecalDrivenSeed())
            h_ele_dEtaEleCl_propOut_eg->Fill(bestGsfElectron.deltaEtaEleClusterTrackAtCalo());
          if (bestGsfElectron.isEB())
            h_ele_dEtaEleCl_propOut_barrel->Fill(bestGsfElectron.deltaEtaEleClusterTrackAtCalo());
          if (bestGsfElectron.isEB() && bestGsfElectron.ecalDrivenSeed())
            h_ele_dEtaEleCl_propOut_eg_barrel->Fill(bestGsfElectron.deltaEtaEleClusterTrackAtCalo());
          if (bestGsfElectron.isEE())
            h_ele_dEtaEleCl_propOut_endcaps->Fill(bestGsfElectron.deltaEtaEleClusterTrackAtCalo());
          if (bestGsfElectron.isEE() && bestGsfElectron.ecalDrivenSeed())
            h_ele_dEtaEleCl_propOut_eg_endcaps->Fill(bestGsfElectron.deltaEtaEleClusterTrackAtCalo());
          h_ele_dEtaEleClVsEta_propOut->Fill(bestGsfElectron.eta(), bestGsfElectron.deltaEtaEleClusterTrackAtCalo());
          h_ele_dEtaEleClVsPhi_propOut->Fill(bestGsfElectron.phi(), bestGsfElectron.deltaEtaEleClusterTrackAtCalo());
          h_ele_dEtaEleClVsPt_propOut->Fill(bestGsfElectron.pt(), bestGsfElectron.deltaEtaEleClusterTrackAtCalo());
          h_ele_dPhiEleCl_propOut->Fill(bestGsfElectron.deltaPhiEleClusterTrackAtCalo());
          if (bestGsfElectron.ecalDrivenSeed())
            h_ele_dPhiEleCl_propOut_eg->Fill(bestGsfElectron.deltaPhiEleClusterTrackAtCalo());
          if (bestGsfElectron.isEB())
            h_ele_dPhiEleCl_propOut_barrel->Fill(bestGsfElectron.deltaPhiEleClusterTrackAtCalo());
          if (bestGsfElectron.isEB() && bestGsfElectron.ecalDrivenSeed())
            h_ele_dPhiEleCl_propOut_eg_barrel->Fill(bestGsfElectron.deltaPhiEleClusterTrackAtCalo());
          if (bestGsfElectron.isEE())
            h_ele_dPhiEleCl_propOut_endcaps->Fill(bestGsfElectron.deltaPhiEleClusterTrackAtCalo());
          if (bestGsfElectron.isEE() && bestGsfElectron.ecalDrivenSeed())
            h_ele_dPhiEleCl_propOut_eg_endcaps->Fill(bestGsfElectron.deltaPhiEleClusterTrackAtCalo());
          h_ele_dPhiEleClVsEta_propOut->Fill(bestGsfElectron.eta(), bestGsfElectron.deltaPhiEleClusterTrackAtCalo());
          h_ele_dPhiEleClVsPhi_propOut->Fill(bestGsfElectron.phi(), bestGsfElectron.deltaPhiEleClusterTrackAtCalo());
          h_ele_dPhiEleClVsPt_propOut->Fill(bestGsfElectron.pt(), bestGsfElectron.deltaPhiEleClusterTrackAtCalo());
          h_ele_HoE->Fill(bestGsfElectron.hadronicOverEm());
          if (bestGsfElectron.ecalDrivenSeed())
            h_ele_HoE_eg->Fill(bestGsfElectron.hadronicOverEm());
          if (bestGsfElectron.isEB())
            h_ele_HoE_barrel->Fill(bestGsfElectron.hadronicOverEm());
          if (bestGsfElectron.isEB() && bestGsfElectron.ecalDrivenSeed())
            h_ele_HoE_eg_barrel->Fill(bestGsfElectron.hadronicOverEm());
          if (bestGsfElectron.isEE())
            h_ele_HoE_endcaps->Fill(bestGsfElectron.hadronicOverEm());
          if (bestGsfElectron.isEE() && bestGsfElectron.ecalDrivenSeed())
            h_ele_HoE_eg_endcaps->Fill(bestGsfElectron.hadronicOverEm());
          if (!bestGsfElectron.isEBEtaGap() && !bestGsfElectron.isEBPhiGap() && !bestGsfElectron.isEBEEGap() &&
              !bestGsfElectron.isEERingGap() && !bestGsfElectron.isEEDeeGap())
            h_ele_HoE_fiducial->Fill(bestGsfElectron.hadronicOverEm());
          h_ele_HoEVsEta->Fill(bestGsfElectron.eta(), bestGsfElectron.hadronicOverEm());
          h_ele_HoEVsPhi->Fill(bestGsfElectron.phi(), bestGsfElectron.hadronicOverEm());
          h_ele_HoEVsE->Fill(bestGsfElectron.caloEnergy(), bestGsfElectron.hadronicOverEm());

          //classes
          int eleClass = bestGsfElectron.classification();
          if (bestGsfElectron.isEE())
            eleClass += 10;
          h_ele_classes->Fill(eleClass);

          if (bestGsfElectron.classification() == GsfElectron::GOLDEN && bestGsfElectron.isEB())
            histSclEoEtrueGolden_barrel->Fill(sclRef->energy() / mcIter->p());
          if (bestGsfElectron.classification() == GsfElectron::GOLDEN && bestGsfElectron.isEE())
            histSclEoEtrueGolden_endcaps->Fill(sclRef->energy() / mcIter->p());
          if (bestGsfElectron.classification() == GsfElectron::SHOWERING && bestGsfElectron.isEB())
            histSclEoEtrueShowering_barrel->Fill(sclRef->energy() / mcIter->p());
          if (bestGsfElectron.classification() == GsfElectron::SHOWERING && bestGsfElectron.isEE())
            histSclEoEtrueShowering_endcaps->Fill(sclRef->energy() / mcIter->p());

          //eleClass = eleClass%100; // get rid of barrel/endcap distinction
          h_ele_eta->Fill(std::abs(bestGsfElectron.eta()));
          if (bestGsfElectron.classification() == GsfElectron::GOLDEN)
            h_ele_eta_golden->Fill(std::abs(bestGsfElectron.eta()));
          if (bestGsfElectron.classification() == GsfElectron::BIGBREM)
            h_ele_eta_bbrem->Fill(std::abs(bestGsfElectron.eta()));
          //if (bestGsfElectron.classification() == GsfElectron::NARROW) h_ele_eta_narrow ->Fill(std::abs(bestGsfElectron.eta()));
          if (bestGsfElectron.classification() == GsfElectron::SHOWERING)
            h_ele_eta_shower->Fill(std::abs(bestGsfElectron.eta()));

          //fbrem
          double fbrem_mean = 0.;
          if (!readAOD_)  // track extra does not exist in AOD
            fbrem_mean =
                1. - bestGsfElectron.gsfTrack()->outerMomentum().R() / bestGsfElectron.gsfTrack()->innerMomentum().R();
          double fbrem_mode = bestGsfElectron.fbrem();
          h_ele_fbrem->Fill(fbrem_mode);
          if (bestGsfElectron.ecalDrivenSeed())
            h_ele_fbrem_eg->Fill(fbrem_mode);
          h_ele_fbremVsEta_mode->Fill(bestGsfElectron.eta(), fbrem_mode);
          if (!readAOD_)  // track extra does not exist in AOD
            h_ele_fbremVsEta_mean->Fill(bestGsfElectron.eta(), fbrem_mean);

          if (bestGsfElectron.classification() == GsfElectron::GOLDEN)
            h_ele_PinVsPoutGolden_mode->Fill(bestGsfElectron.trackMomentumOut().R(),
                                             bestGsfElectron.trackMomentumAtVtx().R());
          if (bestGsfElectron.classification() == GsfElectron::SHOWERING)
            h_ele_PinVsPoutShowering_mode->Fill(bestGsfElectron.trackMomentumOut().R(),
                                                bestGsfElectron.trackMomentumAtVtx().R());
          if (!readAOD_) {  // track extra not available in AOD
            if (bestGsfElectron.classification() == GsfElectron::GOLDEN)
              h_ele_PinVsPoutGolden_mean->Fill(bestGsfElectron.gsfTrack()->outerMomentum().R(),
                                               bestGsfElectron.gsfTrack()->innerMomentum().R());
            if (bestGsfElectron.classification() == GsfElectron::SHOWERING)
              h_ele_PinVsPoutShowering_mean->Fill(bestGsfElectron.gsfTrack()->outerMomentum().R(),
                                                  bestGsfElectron.gsfTrack()->innerMomentum().R());
          }
          if (bestGsfElectron.classification() == GsfElectron::GOLDEN)
            h_ele_PtinVsPtoutGolden_mode->Fill(bestGsfElectron.trackMomentumOut().Rho(),
                                               bestGsfElectron.trackMomentumAtVtx().Rho());
          if (bestGsfElectron.classification() == GsfElectron::SHOWERING)
            h_ele_PtinVsPtoutShowering_mode->Fill(bestGsfElectron.trackMomentumOut().Rho(),
                                                  bestGsfElectron.trackMomentumAtVtx().Rho());
          if (!readAOD_) {  // track extra not available in AOD
            if (bestGsfElectron.classification() == GsfElectron::GOLDEN)
              h_ele_PtinVsPtoutGolden_mean->Fill(bestGsfElectron.gsfTrack()->outerMomentum().Rho(),
                                                 bestGsfElectron.gsfTrack()->innerMomentum().Rho());
            if (bestGsfElectron.classification() == GsfElectron::SHOWERING)
              h_ele_PtinVsPtoutShowering_mean->Fill(bestGsfElectron.gsfTrack()->outerMomentum().Rho(),
                                                    bestGsfElectron.gsfTrack()->innerMomentum().Rho());
          }

          h_ele_mva->Fill(bestGsfElectron.mva_e_pi());
          if (bestGsfElectron.ecalDrivenSeed())
            h_ele_mva_eg->Fill(bestGsfElectron.mva_e_pi());
          if (bestGsfElectron.ecalDrivenSeed())
            h_ele_provenance->Fill(1.);
          if (bestGsfElectron.trackerDrivenSeed())
            h_ele_provenance->Fill(-1.);
          if (bestGsfElectron.trackerDrivenSeed() || bestGsfElectron.ecalDrivenSeed())
            h_ele_provenance->Fill(0.);
          if (bestGsfElectron.trackerDrivenSeed() && !bestGsfElectron.ecalDrivenSeed())
            h_ele_provenance->Fill(-2.);
          if (!bestGsfElectron.trackerDrivenSeed() && bestGsfElectron.ecalDrivenSeed())
            h_ele_provenance->Fill(2.);

          h_ele_tkSumPt_dr03->Fill(bestGsfElectron.dr03TkSumPt());
          h_ele_ecalRecHitSumEt_dr03->Fill(bestGsfElectron.dr03EcalRecHitSumEt());
          h_ele_hcalDepth1TowerSumEt_dr03->Fill(bestGsfElectron.dr03HcalDepth1TowerSumEt());
          h_ele_hcalDepth2TowerSumEt_dr03->Fill(bestGsfElectron.dr03HcalDepth2TowerSumEt());
          h_ele_tkSumPt_dr04->Fill(bestGsfElectron.dr04TkSumPt());
          h_ele_ecalRecHitSumEt_dr04->Fill(bestGsfElectron.dr04EcalRecHitSumEt());
          h_ele_hcalDepth1TowerSumEt_dr04->Fill(bestGsfElectron.dr04HcalDepth1TowerSumEt());
          h_ele_hcalDepth2TowerSumEt_dr04->Fill(bestGsfElectron.dr04HcalDepth2TowerSumEt());

        }  // gsf electron found

      }  // mc particle found
    }

  }  // loop over mc particle

  h_mcNum->Fill(mcNum);
  h_eleNum->Fill(eleNum);
}
