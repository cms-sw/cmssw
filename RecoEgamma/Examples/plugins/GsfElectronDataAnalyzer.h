#ifndef GsfElectronDataAnalyzer_h
#define GsfElectronDataAnalyzer_h

//
// Package:         RecoEgamma/Examples
// Class:           GsfElectronDataAnalyzer
//

//
// Original Author:  Ursula Berthon, Claude Charlot
//         Created:  Mon Mar 27 13:22:06 CEST 2006
// $Id: GsfElectronDataAnalyzer.h,v 1.22 2012/09/13 20:08:32 wdd Exp $
//
//


#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

class MagneticField;
class TFile;
class TH1F;
class TH2F;
class TH1I;
class TProfile;
class TTree;

class GsfElectronDataAnalyzer : public edm::EDAnalyzer
{
 public:

  explicit GsfElectronDataAnalyzer(const edm::ParameterSet& conf);

  virtual ~GsfElectronDataAnalyzer();

  virtual void beginJob();
  virtual void endJob();
  virtual void analyze(const edm::Event& e, const edm::EventSetup& c);

 private:

  bool trigger(const edm::Event & e);
  
  unsigned int nEvents_;
  unsigned int nAfterTrigger_;

  edm::InputTag triggerResults_;
  edm::InputTag beamSpot_;

  std::vector<std::string > HLTPathsByName_;
  std::vector<unsigned int> HLTPathsByIndex_;

  std::string outputFile_;
  edm::InputTag electronCollection_;
  edm::InputTag matchingObjectCollection_;
  std::string matchingCondition_;
  std::string type_;
  bool readAOD_;
  // matching 
  double maxPtMatchingObject_;
  double maxAbsEtaMatchingObject_;
  double deltaR_;

  TrajectoryStateTransform transformer_;
  edm::ESHandle<TrackerGeometry> pDD;
  edm::ESHandle<MagneticField> theMagField;

  TFile *histfile_;
  TTree *tree_;
  float mcEnergy[10], mcEta[10], mcPhi[10], mcPt[10], mcQ[10];
  float superclusterEnergy[10], superclusterEta[10], superclusterPhi[10], superclusterEt[10];
  float seedMomentum[10], seedEta[10], seedPhi[10], seedPt[10], seedQ[10];

  TH1F *h_matchingObjectNum;

  TH1F *h_matchingObjectEta;
  TH1F *h_matchingObjectAbsEta;
  TH1F *h_matchingObjectP;
  TH1F *h_matchingObjectPt;
  TH1F *h_matchingObjectPhi;
  TH1F *h_matchingObjectZ;

  TH1F *h_ele_matchingObjectEta_matched;
  TH1F *h_ele_matchingObjectAbsEta_matched;
  TH1F *h_ele_matchingObjectPt_matched;
  TH1F *h_ele_matchingObjectPhi_matched;
  TH1F *h_ele_matchingObjectZ_matched;

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

  TH1F *histSclEn_ ;
  TH1F *histSclEoEmatchingObject_barrel_matched;
  TH1F *histSclEoEmatchingObject_endcaps_matched;
  TH1F *histSclEoEmatchingObject_barrel_new_matched;
  TH1F *histSclEoEmatchingObject_endcaps_new_matched;
  TH1F *histSclEt_ ;
  TH2F *histSclEtVsEta_ ;
  TH2F *histSclEtVsPhi_ ;
  TH2F *histSclEtaVsPhi_ ;
  TH1F *histSclEta_ ;
  TH1F *histSclPhi_ ;
  TH1F *histSclSigEtaEta_ ;
  TH1F *histSclSigEtaEta_barrel_ ;
  TH1F *histSclSigEtaEta_endcaps_ ;
  TH1F *histSclSigIEtaIEta_ ;
  TH1F *histSclSigIEtaIEta_barrel_ ;
  TH1F *histSclSigIEtaIEta_endcaps_ ;
  TH1F *histSclE1x5_ ;
  TH1F *histSclE1x5_barrel_ ;
  TH1F *histSclE1x5_endcaps_ ;
  TH1F *histSclE2x5max_ ;
  TH1F *histSclE2x5max_barrel_ ;
  TH1F *histSclE2x5max_endcaps_ ;
  TH1F *histSclE5x5_ ;
  TH1F *histSclE5x5_barrel_ ;
  TH1F *histSclE5x5_endcaps_ ;

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
  TH1F *h_ele_chi2_barrel_;
  TH1F *h_ele_chi2_endcaps_;
  TH2F *h_ele_chi2VsEta;
  TH2F *h_ele_chi2VsPhi;
  TH2F *h_ele_chi2VsPt;

  TH1F *h_ele_PtoPtmatchingObject_matched;
  TH1F *h_ele_PoPmatchingObject_matched;
  TH2F *h_ele_PoPmatchingObjectVsEta_matched;
  TH2F *h_ele_PoPmatchingObjectVsPhi_matched;
  TH2F *h_ele_PoPmatchingObjectVsPt_matched;
  TH1F *h_ele_PoPmatchingObject_barrel_matched;
  TH1F *h_ele_PoPmatchingObject_endcaps_matched;
  TH1F *h_ele_PtoPtmatchingObject_barrel_matched;
  TH1F *h_ele_PtoPtmatchingObject_endcaps_matched;
  TH1F *h_ele_EtaMnEtamatchingObject_matched;
  TH2F *h_ele_EtaMnEtamatchingObjectVsEta_matched;
  TH2F *h_ele_EtaMnEtamatchingObjectVsPhi_matched;
  TH2F *h_ele_EtaMnEtamatchingObjectVsPt_matched;
  TH1F *h_ele_PhiMnPhimatchingObject_matched;
  TH1F *h_ele_PhiMnPhimatchingObject2_matched;
  TH2F *h_ele_PhiMnPhimatchingObjectVsEta_matched;
  TH2F *h_ele_PhiMnPhimatchingObjectVsPhi_matched;
  TH2F *h_ele_PhiMnPhimatchingObjectVsPt_matched;
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
  TH2F *h_ele_EoPVsEta;
  TH2F *h_ele_EoPVsPhi;
  TH2F *h_ele_EoPVsE;
  TH1F *h_ele_EseedOP;
  TH1F *h_ele_EseedOP_barrel;
  TH1F *h_ele_EseedOP_endcaps;
  TH2F *h_ele_EseedOPVsEta;
  TH2F *h_ele_EseedOPVsPhi;
  TH2F *h_ele_EseedOPVsE;
  TH1F *h_ele_EoPout;
  TH1F *h_ele_EoPout_barrel;
  TH1F *h_ele_EoPout_endcaps;
  TH2F *h_ele_EoPoutVsEta;
  TH2F *h_ele_EoPoutVsPhi;
  TH2F *h_ele_EoPoutVsE;
  TH1F *h_ele_EeleOPout;
  TH1F *h_ele_EeleOPout_barrel;
  TH1F *h_ele_EeleOPout_endcaps;
  TH2F *h_ele_EeleOPoutVsEta;
  TH2F *h_ele_EeleOPoutVsPhi;
  TH2F *h_ele_EeleOPoutVsE;

  TH1F *h_ele_dEtaSc_propVtx;
  TH1F *h_ele_dEtaSc_propVtx_barrel;
  TH1F *h_ele_dEtaSc_propVtx_endcaps;
  TH2F *h_ele_dEtaScVsEta_propVtx;
  TH2F *h_ele_dEtaScVsPhi_propVtx;
  TH2F *h_ele_dEtaScVsPt_propVtx;
  TH1F *h_ele_dPhiSc_propVtx;
  TH1F *h_ele_dPhiSc_propVtx_barrel;
  TH1F *h_ele_dPhiSc_propVtx_endcaps;
  TH2F *h_ele_dPhiScVsEta_propVtx;
  TH2F *h_ele_dPhiScVsPhi_propVtx;
  TH2F *h_ele_dPhiScVsPt_propVtx;
  TH1F *h_ele_dEtaCl_propOut;
  TH1F *h_ele_dEtaCl_propOut_barrel;
  TH1F *h_ele_dEtaCl_propOut_endcaps;
  TH2F *h_ele_dEtaClVsEta_propOut;
  TH2F *h_ele_dEtaClVsPhi_propOut;
  TH2F *h_ele_dEtaClVsPt_propOut;
  TH1F *h_ele_dPhiCl_propOut;
  TH1F *h_ele_dPhiCl_propOut_barrel;
  TH1F *h_ele_dPhiCl_propOut_endcaps;
  TH2F *h_ele_dPhiClVsEta_propOut;
  TH2F *h_ele_dPhiClVsPhi_propOut;
  TH2F *h_ele_dPhiClVsPt_propOut;
  TH1F *h_ele_dEtaEleCl_propOut;
  TH1F *h_ele_dEtaEleCl_propOut_barrel;
  TH1F *h_ele_dEtaEleCl_propOut_endcaps;
  TH2F *h_ele_dEtaEleClVsEta_propOut;
  TH2F *h_ele_dEtaEleClVsPhi_propOut;
  TH2F *h_ele_dEtaEleClVsPt_propOut;
  TH1F *h_ele_dPhiEleCl_propOut;
  TH1F *h_ele_dPhiEleCl_propOut_barrel;
  TH1F *h_ele_dPhiEleCl_propOut_endcaps;
  TH2F *h_ele_dPhiEleClVsEta_propOut;
  TH2F *h_ele_dPhiEleClVsPhi_propOut;
  TH2F *h_ele_dPhiEleClVsPt_propOut;

  TH1F *h_ele_seed_dphi2_;
  TH2F *h_ele_seed_dphi2VsEta_;
  TH2F *h_ele_seed_dphi2VsPt_ ;
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
  TH1F *h_ele_HoE_fiducial;
  TH2F *h_ele_HoEVsEta;
  TH2F *h_ele_HoEVsPhi;
  TH2F *h_ele_HoEVsE;

  TH1F *h_ele_fbrem;
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
  TH1F *histSclEoEmatchingObjectGolden_barrel;
  TH1F *histSclEoEmatchingObjectGolden_endcaps;
  TH1F *histSclEoEmatchingObjectShowering_barrel;
  TH1F *histSclEoEmatchingObjectShowering_endcaps;
  
  TH1F *h_ele_mva; 
  TH1F *h_ele_provenance; 

  TH1F *h_ele_tkSumPt_dr03; 
  TH1F *h_ele_ecalRecHitSumEt_dr03;
  TH1F *h_ele_hcalDepth1TowerSumEt_dr03;
  TH1F *h_ele_hcalDepth2TowerSumEt_dr03;
  TH1F *h_ele_tkSumPt_dr04; 
  TH1F *h_ele_ecalRecHitSumEt_dr04;
  TH1F *h_ele_hcalDepth1TowerSumEt_dr04;
  TH1F *h_ele_hcalDepth2TowerSumEt_dr04;

  // electron selection
  double minEt_;
  double minPt_;
  double maxAbsEta_;
  bool isEB_;
  bool isEE_;
  bool isNotEBEEGap_;
  bool isEcalDriven_;
  bool isTrackerDriven_;
  double eOverPMinBarrel_;
  double eOverPMaxBarrel_;
  double eOverPMinEndcaps_;
  double eOverPMaxEndcaps_;
  double dEtaMinBarrel_;
  double dEtaMaxBarrel_;
  double dEtaMinEndcaps_;
  double dEtaMaxEndcaps_;
  double dPhiMinBarrel_;
  double dPhiMaxBarrel_;
  double dPhiMinEndcaps_;
  double dPhiMaxEndcaps_;
  double sigIetaIetaMinBarrel_;
  double sigIetaIetaMaxBarrel_;
  double sigIetaIetaMinEndcaps_;
  double sigIetaIetaMaxEndcaps_;
  double hadronicOverEmMaxBarrel_;
  double hadronicOverEmMaxEndcaps_;
  double mvaMin_;
  double tipMaxBarrel_;
  double tipMaxEndcaps_;
  double tkIso03Max_;
  double hcalIso03Depth1MaxBarrel_;
  double hcalIso03Depth1MaxEndcaps_;
  double hcalIso03Depth2MaxEndcaps_;
  double ecalIso03MaxBarrel_;
  double ecalIso03MaxEndcaps_;
  
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

#endif



