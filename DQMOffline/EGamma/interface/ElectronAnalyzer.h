#ifndef ElectronAnalyzer_h
#define ElectronAnalyzer_h
  
//
// Package:         DQMOffline/EGamma
// Class:           ElectronAnalyzer
// 

//
// Original Author:  Ursula Berthon, Claude Charlot
//         Created:  Mon Mar 27 13:22:06 CEST 2006
// $Id: ElectronAnalyzer.h,v 1.2 2008/09/12 11:40:09 uberthon Exp $
//
//
  
  
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
 
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/EDProduct.h"
 
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

//DQM services
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/MonitorElement.h"


class MagneticField;  
class TFile;
class TH1F;
class TH2F;
class TH1I;
class TProfile;
class TTree;

class ElectronAnalyzer : public edm::EDAnalyzer
{
 public:
  
  explicit ElectronAnalyzer(const edm::ParameterSet& conf);
  
  virtual ~ElectronAnalyzer();
  
  virtual void beginJob(edm::EventSetup const& iSetup);
  virtual void endJob();
  virtual void analyze(const edm::Event& e, const edm::EventSetup& c);
  
 private:

  edm::ParameterSet parameters_;

  TrajectoryStateTransform transformer_;
  edm::ESHandle<TrackerGeometry> pDD;
  edm::ESHandle<MagneticField> theMagField;
  TFile *histfile_;
  TTree *tree_;
  float mcEnergy[10], mcEta[10], mcPhi[10], mcPt[10], mcQ[10];
  float superclusterEnergy[10], superclusterEta[10], superclusterPhi[10], superclusterEt[10];
  float seedMomentum[10], seedEta[10], seedPhi[10], seedPt[10], seedQ[10];

  MonitorElement   *h_ele_matchingObjectEta;
  MonitorElement   *h_ele_matchingObjectPt;
  MonitorElement   *h_ele_matchingObjectPhi;
  MonitorElement   *h_ele_matchingObjectZ;
  
  MonitorElement   *h_ele_matchingObjectEta_matched;
  MonitorElement   *h_ele_matchingObjectPt_matched;
  MonitorElement   *h_ele_matchingObjectPhi_matched;
  MonitorElement   *h_ele_matchingObjectZ_matched;

  // electron basic quantities
  MonitorElement* h_ele_vertexP;
  MonitorElement* h_ele_vertexPt;
  MonitorElement* h_ele_vertexEta;
  MonitorElement* h_ele_vertexPhi;
  MonitorElement* h_ele_vertexX;
  MonitorElement* h_ele_vertexY;
  MonitorElement* h_ele_vertexZ;
  MonitorElement* h_ele_charge;

  // electron matching and ID
  MonitorElement* h_ele_EoP;
  MonitorElement* h_ele_EoPout;
  MonitorElement* h_ele_dEtaSc_propVtx;
  MonitorElement* h_ele_dPhiSc_propVtx;
  MonitorElement* h_ele_dPhiCl_propOut;
  MonitorElement* h_ele_HoE;
  MonitorElement* h_ele_PinMnPout_mode;
  MonitorElement* h_ele_classes;
  MonitorElement* h_ele_eta_golden;
  MonitorElement* h_ele_eta_shower;
  MonitorElement* h_ele_eta_goldenFrac;
  MonitorElement* h_ele_eta_showerFrac;

  MonitorElement *h_ele_eta;

  // electron track
  MonitorElement* h_ele_foundHits;
  MonitorElement* h_ele_chi2;

  // efficiencies
  MonitorElement* h_ele_etaEff;
  MonitorElement* h_ele_ptEff;
  MonitorElement* h_ele_phiEff;
  MonitorElement* h_ele_zEff;

  DQMStore *dbe_;
  int verbosity_;

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
 


