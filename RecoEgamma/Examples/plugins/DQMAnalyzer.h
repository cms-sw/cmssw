#ifndef DQMAnalyzer_h
#define DQMAnalyzer_h

//
// Package:         RecoEgamma/Examples
// Class:           GsfElectronDataAnalyzer
//

//
// Original Author:  Ursula Berthon, Claude Charlot
//         Created:  Mon Mar 27 13:22:06 CEST 2006
// $Id: GsfElectronDataAnalyzer.h,v 1.18 2009/09/27 16:45:33 charlot Exp $
//
//


#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/TriggerNames.h"

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

class DQMAnalyzer : public edm::EDAnalyzer
{
 public:

  explicit DQMAnalyzer(const edm::ParameterSet& conf);

  virtual ~DQMAnalyzer();

  virtual void beginJob();
  virtual void endJob();
  virtual void analyze(const edm::Event& e, const edm::EventSetup& c);

 private:

  bool trigger(const edm::Event & e);
  
  unsigned int nEvents_;
  unsigned int nAfterTrigger_;

  edm::InputTag triggerResults_;
  edm::TriggerNames triggerNames_;

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
  
  int Selection_;
  double massLow_;
  double massHigh_;  
  bool TPchecksign_;
  bool TAGcheckclass_;
  bool PROBEetcut_;
  bool PROBEcheckclass_;

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


  TH1F *h_ele_charge;

  TH1F *h_ele_vertexP;
  TH1F *h_ele_vertexPt;
  TH1F *h_ele_Et;
  TH1F *h_ele_vertexEta;
  TH1F *h_ele_vertexAbsEta;
  TH1F *h_ele_vertexPhi;
  TH1F *h_ele_vertexX;
  TH1F *h_ele_vertexY;
  TH1F *h_ele_vertexZ;
  TH1F *h_ele_vertexTIP;
  

  TH1F *histNum_;

  TH1F *histSclEn_ ;
  TH1F *histSclEt_ ;
  TH1F *histSclEta_ ;
  TH1F *histSclPhi_ ;
  TH1F *histSclSigEtaEta_ ;


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
  TH1F *h_ele_PoPmatchingObject_barrel_matched;
  TH1F *h_ele_PoPmatchingObject_endcaps_matched;
  TH1F *h_ele_PtoPtmatchingObject_barrel_matched;
  TH1F *h_ele_PtoPtmatchingObject_endcaps_matched;
  TH1F *h_ele_EtaMnEtamatchingObject_matched;
  TH1F *h_ele_PhiMnPhimatchingObject_matched;
  TH1F *h_ele_PhiMnPhimatchingObject2_matched;
  TH1F *h_ele_PinMnPout;
  TH1F *h_ele_PinMnPout_mode;

  TH1F *h_ele_outerP;
  TH1F *h_ele_outerP_mode;
  TH1F *h_ele_outerPt;
  TH1F *h_ele_outerPt_mode;
  TH1F *h_ele_EoP;
  TH1F *h_ele_EoPout;
  TH1F *h_ele_EeleOPout;


  TH1F *h_ele_dEtaSc_propVtx;
  TH1F *h_ele_dPhiSc_propVtx;
  TH1F *h_ele_dEtaCl_propOut;
  TH1F *h_ele_dPhiCl_propOut;
  TH1F *h_ele_dEtaEleCl_propOut;
  TH1F *h_ele_dPhiEleCl_propOut;
  TH1F *h_ele_dPhiEleCl_propOut_barrel;
  TH1F *h_ele_dPhiEleCl_propOut_endcaps;
  
 

  TH1F *h_ele_HoE;

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


  TH1F *h_ele_mee_os;



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
