#ifndef DQMAnalyzer_h
#define DQMAnalyzer_h

//
// Package:         RecoEgamma/Examples
// Class:           GsfElectronDataAnalyzer
//

//
// Original Author:  Ursula Berthon, Claude Charlot
//         Created:  Mon Mar 27 13:22:06 CEST 2006
// $Id: DQMAnalyzer.h,v 1.6 2012/09/13 20:08:31 wdd Exp $
//
//


#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

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

  explicit DQMAnalyzer( const edm::ParameterSet & conf ) ;

  virtual ~DQMAnalyzer() ;

  virtual void beginJob() ;
  virtual void endJob() ;
  virtual void analyze( const edm::Event & e, const edm::EventSetup & c) ;

 private:

   //=========================================
   // parameters
   //=========================================

  std::string outputFile_;
  edm::InputTag electronCollection_;
  edm::InputTag matchingObjectCollection_;
  edm::InputTag beamSpot_;
  std::string matchingCondition_;
  //std::string type_;
  bool readAOD_;

  // matching
  double maxPtMatchingObject_;
  double maxAbsEtaMatchingObject_;
  double deltaR_;

  // tag and probe
  int Selection_;
  double massLow_;
  double massHigh_;
  bool TPchecksign_;
  bool TAGcheckclass_;
  bool PROBEetcut_;
  bool PROBEcheckclass_;

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

  // for trigger
  edm::InputTag triggerResults_;
  std::vector<std::string > HLTPathsByName_;

  // histos limits and binning
  int nbineta; int nbineta2D; double etamin; double etamax;
  int nbinphi; int nbinphi2D; double phimin; double phimax;
  int nbinpt; int nbinpteff; int nbinpt2D; double ptmax;
  int nbinp; int nbinp2D; double pmax;
  int nbineop; int nbineop2D; double eopmax; double eopmaxsht;
  int nbindeta; double detamin; double detamax;
  int nbindphi; double dphimin; double dphimax;
  int nbindetamatch; int nbindetamatch2D; double detamatchmin; double detamatchmax;
  int nbindphimatch; int nbindphimatch2D; double dphimatchmin; double dphimatchmax;
  int nbinfhits; double fhitsmax;
  int nbinlhits; double lhitsmax;
  int nbinxyz;
  int nbinpoptrue; double poptruemin; double poptruemax;
  int nbinmee; double meemin; double meemax;
  int nbinhoe; double hoemin; double hoemax;


  //=========================================
  // usual attributes and methods
  //=========================================


  unsigned int nEvents_ ;

  bool selected( const reco::GsfElectronCollection::const_iterator & gsfIter , double vertexTIP ) ;
  bool generalCut( const reco::GsfElectronCollection::const_iterator & gsfIter) ;
  bool etCut( const reco::GsfElectronCollection::const_iterator & gsfIter ) ;
  bool isolationCut( const reco::GsfElectronCollection::const_iterator & gsfIter, double vertexTIP ) ;
  bool idCut( const reco::GsfElectronCollection::const_iterator & gsfIter ) ;

  bool trigger( const edm::Event & e ) ;
  unsigned int nAfterTrigger_;
  std::vector<unsigned int> HLTPathsByIndex_;

  TrajectoryStateTransform transformer_;
  edm::ESHandle<TrackerGeometry> pDD;
  edm::ESHandle<MagneticField> theMagField;

  TFile * histfile_ ;
  TTree * tree_ ;

  float mcEnergy[10], mcEta[10], mcPhi[10], mcPt[10], mcQ[10] ;
  float superclusterEnergy[10], superclusterEta[10], superclusterPhi[10], superclusterEt[10] ;
  float seedMomentum[10], seedEta[10], seedPhi[10], seedPt[10], seedQ[10] ;


  //=========================================
  // histograms
  //=========================================

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

  //TH1F *h_ele_vertexP;
  TH1F *h_ele_vertexPt;
  TH1F *h_ele_Et;
  TH1F *h_ele_vertexEta;
  //TH1F *h_ele_vertexAbsEta;
  TH1F *h_ele_vertexPhi;
  TH1F *h_ele_vertexX;
  TH1F *h_ele_vertexY;
  TH1F *h_ele_vertexZ;
  TH1F *h_ele_vertexTIP;
  TH1F *h_ele_charge;

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
//  TH1F *h_ele_foundHits_barrel;
//  TH1F *h_ele_foundHits_endcaps;
  TH2F *h_ele_foundHitsVsEta;
  TH2F *h_ele_foundHitsVsPhi;
  TH2F *h_ele_foundHitsVsPt;
  TH1F *h_ele_lostHits;
//  TH1F *h_ele_lostHits_barrel;
//  TH1F *h_ele_lostHits_endcaps;
  TH2F *h_ele_lostHitsVsEta;
  TH2F *h_ele_lostHitsVsPhi;
  TH2F *h_ele_lostHitsVsPt;
  TH1F *h_ele_chi2;
//  TH1F *h_ele_chi2_barrel_;
//  TH1F *h_ele_chi2_endcaps_;
  TH2F *h_ele_chi2VsEta;
  TH2F *h_ele_chi2VsPhi;
  TH2F *h_ele_chi2VsPt;

  TH1F *h_ele_EoP;
//  TH1F *h_ele_EoPout;
  TH1F *h_ele_EeleOPout;
  TH1F *h_ele_dEtaSc_propVtx;
  TH1F *h_ele_dPhiSc_propVtx;
  TH1F *h_ele_dEtaCl_propOut;
  TH1F *h_ele_dPhiCl_propOut;
  TH1F *h_ele_dEtaEleCl_propOut;
  TH1F *h_ele_dPhiEleCl_propOut;
//  TH1F *h_ele_dPhiEleCl_propOut_barrel;
//  TH1F *h_ele_dPhiEleCl_propOut_endcaps;
  TH1F *h_ele_HoE;
  TH1F *h_ele_outerP;
  TH1F *h_ele_outerP_mode;
  TH1F *h_ele_outerPt;
  TH1F *h_ele_outerPt_mode;

  TH1F *h_ele_PinMnPout;
  TH1F *h_ele_PinMnPout_mode;

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

  // OBSOLETE
  //  TH1F *h_ele_PtoPtmatchingObject_matched; //OBSOLETE ?
  //  TH1F *h_ele_PtoPtmatchingObject_barrel_matched; //OBSOLETE ?
  //  TH1F *h_ele_PtoPtmatchingObject_endcaps_matched; //OBSOLETE ?
  //  TH1F *h_ele_PoPmatchingObject_matched; //OBSOLETE ?
  //  TH1F *h_ele_PoPmatchingObject_barrel_matched; //OBSOLETE ?
  //  TH1F *h_ele_PoPmatchingObject_endcaps_matched; //OBSOLETE ?
  //  TH1F *h_ele_EtaMnEtamatchingObject_matched; //OBSOLETE ?
  //  TH1F *h_ele_PhiMnPhimatchingObject_matched; //OBSOLETE ?
  //  TH1F *h_ele_PhiMnPhimatchingObject2_matched; //OBSOLETE ?

 };

#endif
