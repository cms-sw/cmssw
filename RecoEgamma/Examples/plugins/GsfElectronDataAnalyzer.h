#ifndef GsfElectronDataAnalyzer_h
#define GsfElectronDataAnalyzer_h
  
//
// Package:         RecoEgamma/Examples
// Class:           GsfElectronDataAnalyzer
// 

//
// Original Author:  Ursula Berthon, Claude Charlot
//         Created:  Mon Mar 27 13:22:06 CEST 2006
// $Id: GsfElectronDataAnalyzer.h,v 1.3 2008/03/15 01:05:52 charlot Exp $
//
//
  
  
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
 
#include "DataFormats/Common/interface/EDProduct.h"
 
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
  
  virtual void beginJob(edm::EventSetup const& iSetup);
  virtual void endJob();
  virtual void analyze(const edm::Event& e, const edm::EventSetup& c);
  
 private:

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

  TH1F *h_ele_TIP_all;
  TH1F *h_ele_EoverP_all;
  TH1F *h_ele_vertexEta_all;
  TH1F *h_ele_vertexPt_all;

  TH1F *h_ele_charge;
  TH2F *h_ele_chargeVsEta;
  TH2F *h_ele_chargeVsPhi;
  TH2F *h_ele_chargeVsPt;
  TH1F *h_ele_vertexP;
  TH1F *h_ele_vertexPt;
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
  TH1F *histSclEoEmatchingObject_barrel;
  TH1F *histSclEoEmatchingObject_endcaps;
  TH1F *histSclEt_ ;
  TH2F *histSclEtVsEta_ ;
  TH2F *histSclEtVsPhi_ ;
  TH2F *histSclEtaVsPhi_ ;
  TH1F *histSclEta_ ;
  TH1F *histSclPhi_ ;

  TH1F *h_ctf_foundHits;
  TH2F *h_ctf_foundHitsVsEta;
  TH2F *h_ctf_lostHitsVsEta;

  TH1F *h_ele_foundHits;
  TH2F *h_ele_foundHitsVsEta;
  TH2F *h_ele_foundHitsVsPhi;
  TH2F *h_ele_foundHitsVsPt;
  TH1F *h_ele_lostHits;
  TH2F *h_ele_lostHitsVsEta;
  TH2F *h_ele_lostHitsVsPhi;
  TH2F *h_ele_lostHitsVsPt;
  TH1F *h_ele_chi2;
  TH2F *h_ele_chi2VsEta;
  TH2F *h_ele_chi2VsPhi;
  TH2F *h_ele_chi2VsPt;
  
  TH1F *h_ele_PoPmatchingObject;
  TH2F *h_ele_PoPmatchingObjectVsEta;
  TH2F *h_ele_PoPmatchingObjectVsPhi;
  TH2F *h_ele_PoPmatchingObjectVsPt;
  TH1F *h_ele_PoPmatchingObject_barrel;
  TH1F *h_ele_PoPmatchingObject_endcaps;
  TH1F *h_ele_EtaMnEtamatchingObject;
  TH2F *h_ele_EtaMnEtamatchingObjectVsEta;
  TH2F *h_ele_EtaMnEtamatchingObjectVsPhi;
  TH2F *h_ele_EtaMnEtamatchingObjectVsPt;
  TH1F *h_ele_PhiMnPhimatchingObject;
  TH1F *h_ele_PhiMnPhimatchingObject2;
  TH2F *h_ele_PhiMnPhimatchingObjectVsEta;
  TH2F *h_ele_PhiMnPhimatchingObjectVsPhi;
  TH2F *h_ele_PhiMnPhimatchingObjectVsPt;
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
  TH2F *h_ele_EoPVsEta;
  TH2F *h_ele_EoPVsPhi;
  TH2F *h_ele_EoPVsE;
  TH1F *h_ele_EoPout;
  TH2F *h_ele_EoPoutVsEta;
  TH2F *h_ele_EoPoutVsPhi;
  TH2F *h_ele_EoPoutVsE;
  
  TH1F *h_ele_dEtaSc_propVtx;
  TH2F *h_ele_dEtaScVsEta_propVtx;
  TH2F *h_ele_dEtaScVsPhi_propVtx;
  TH2F *h_ele_dEtaScVsPt_propVtx;
  TH1F *h_ele_dPhiSc_propVtx;
  TH2F *h_ele_dPhiScVsEta_propVtx;
  TH2F *h_ele_dPhiScVsPhi_propVtx;
  TH2F *h_ele_dPhiScVsPt_propVtx;
  TH1F *h_ele_dEtaCl_propOut;
  TH2F *h_ele_dEtaClVsEta_propOut;
  TH2F *h_ele_dEtaClVsPhi_propOut;
  TH2F *h_ele_dEtaClVsPt_propOut;
  TH1F *h_ele_dPhiCl_propOut;
  TH2F *h_ele_dPhiClVsEta_propOut;
  TH2F *h_ele_dPhiClVsPhi_propOut;
  TH2F *h_ele_dPhiClVsPt_propOut;
  
  TH1F *h_ele_classes;
  TH1F *h_ele_eta;
  TH1F *h_ele_eta_golden;
  TH1F *h_ele_eta_bbrem;
  TH1F *h_ele_eta_narrow;
  TH1F *h_ele_eta_shower;
  
  TH1F *h_ele_HoE;
  TH2F *h_ele_HoEVsEta;
  TH2F *h_ele_HoEVsPhi;
  TH2F *h_ele_HoEVsE;
  
  TProfile *h_ele_fbremVsEta_mode;
  TProfile *h_ele_fbremVsEta_mean;
  
  TH2F *h_ele_PinVsPoutGolden_mode;
  TH2F *h_ele_PinVsPoutShowering0_mode;
  TH2F *h_ele_PinVsPoutShowering1234_mode;
  TH2F *h_ele_PinVsPoutGolden_mean;
  TH2F *h_ele_PinVsPoutShowering0_mean;
  TH2F *h_ele_PinVsPoutShowering1234_mean;
  TH2F *h_ele_PtinVsPtoutGolden_mode;
  TH2F *h_ele_PtinVsPtoutShowering0_mode;
  TH2F *h_ele_PtinVsPtoutShowering1234_mode;
  TH2F *h_ele_PtinVsPtoutGolden_mean;
  TH2F *h_ele_PtinVsPtoutShowering0_mean;
  TH2F *h_ele_PtinVsPtoutShowering1234_mean;
  TH1F *histSclEoEmatchingObjectGolden_barrel;
  TH1F *histSclEoEmatchingObjectGolden_endcaps;
  TH1F *histSclEoEmatchingObjectShowering0_barrel;
  TH1F *histSclEoEmatchingObjectShowering0_endcaps;
  TH1F *histSclEoEmatchingObjectShowering1234_barrel;
  TH1F *histSclEoEmatchingObjectShowering1234_endcaps;

  std::string outputFile_; 
  edm::InputTag electronCollection_;
  edm::InputTag matchingObjectCollection_;
  std::string type_;
  
  double maxPt_;
  double maxAbsEta_;
  double deltaR_; 
  
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
  
 };
  
#endif
 


