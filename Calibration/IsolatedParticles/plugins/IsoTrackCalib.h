#ifndef CalibrationIsolatedParticlesIsoTrackCalib_h
#define CalibrationIsolatedParticlesIsoTrackCalib_h

// system include files
#include <memory>

// Root objects
#include "TROOT.h"
#include "TSystem.h"
#include "TFile.h"
#include "TProfile.h"
#include "TDirectory.h"
#include "TTree.h"
#include "TLorentzVector.h"
#include "TInterpreter.h"

//Tracks
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/HitPattern.h"
#include "DataFormats/TrackReco/interface/TrackBase.h"
// Vertices
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

//Triggers
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/Luminosity/interface/LumiDetails.h"
#include "DataFormats/Math/interface/LorentzVector.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "HLTrigger/HLTcore/interface/HLTPrescaleProvider.h"

#include "Calibration/IsolatedParticles/interface/CaloPropagateTrack.h"
#include "Calibration/IsolatedParticles/interface/ChargeIsolation.h"
#include "Calibration/IsolatedParticles/interface/eCone.h"
#include "Calibration/IsolatedParticles/interface/eECALMatrix.h"
#include "Calibration/IsolatedParticles/interface/TrackSelection.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"
#include "Geometry/CaloTopology/interface/EcalTrigTowerConstituentsMap.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgoRcd.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"
#include "DataFormats/JetReco/interface/PFJet.h"
//#include "DataFormats/PatCandidates/interface/Jet.h"

class IsoTrackCalib : public edm::EDAnalyzer {

public:
  explicit IsoTrackCalib(const edm::ParameterSet&);
  ~IsoTrackCalib();
 
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  double dR(double eta1, double eta2, double phi1, double phi2);

private:
  virtual void beginJob() ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  virtual void beginRun(edm::Run const&, edm::EventSetup const&);
  virtual void endRun(edm::Run const&, edm::EventSetup const&);
  virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);
  virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);
 


  double dPt(math::XYZTLorentzVector&, math::XYZTLorentzVector&);
  double dP(math::XYZTLorentzVector&, math::XYZTLorentzVector&);
  double dinvPt(math::XYZTLorentzVector&, math::XYZTLorentzVector&);
  double dEta(math::XYZTLorentzVector&, math::XYZTLorentzVector&);
  double dPhi(math::XYZTLorentzVector&, math::XYZTLorentzVector&);
  double dR(math::XYZTLorentzVector&, math::XYZTLorentzVector&);
  void clearTreeVectors();

  bool                       changed;
  edm::Service<TFileService> fs;
  HLTPrescaleProvider hltPrescaleProvider_;
  std::vector<std::string>   trigNames, HLTNames;
  int                        verbosity;
  spr::trackSelectionParameters selectionParameters;
  std::string                theTrackQuality;
  double                     dr_L1, a_mipR, a_coneR, a_charIsoR, a_neutIsoR;
  double                     a_neutR1, a_neutR2, cutMip, cutCharge, cutNeutral;
  int                        minRunNo, maxRunNo, nRun;
  std::vector<double>        drCuts;

  edm::InputTag              triggerEvent_, theTriggerResultsLabel;
  edm::EDGetTokenT<LumiDetails>            tok_lumi;
  edm::EDGetTokenT<trigger::TriggerEvent>  tok_trigEvt;
  edm::EDGetTokenT<edm::TriggerResults>    tok_trigRes;

  edm::EDGetTokenT<reco::TrackCollection>  tok_genTrack_;
  edm::EDGetTokenT<reco::VertexCollection> tok_recVtx_;
  edm::EDGetTokenT<reco::BeamSpot>         tok_bs_;
  edm::EDGetTokenT<EcalRecHitCollection>   tok_EB_;
  edm::EDGetTokenT<EcalRecHitCollection>   tok_EE_;
  edm::EDGetTokenT<HBHERecHitCollection>   tok_hbhe_;
  edm::EDGetTokenT<GenEventInfoProduct>    tok_ew_; 
  edm::EDGetTokenT<reco::PFJetCollection>  tok_pf_;  
  
  bool                       firstEvent;
  std::map<std::pair<unsigned int, std::string>, unsigned int> TrigList;
  std::map<std::pair<unsigned int, std::string>, const std::pair<int, int> > TrigPreList;
  TH1I                       *h_nHLT, *h_HLTAccept;
  std::vector<TH1I*>          h_HLTAccepts;
  TH1I                       *g_Pre, *g_PreL1, *g_PreHLT, *g_Accepts;

  TTree                      *tree;
  int                         Run, Event; 
  std::vector<double>        *t_trackP, *t_trackPx, *t_trackPy, *t_trackPz;
  std::vector<double>        *t_trackEta, *t_trackPhi, *t_trackPt, *t_neu_iso;
  std::vector<double>        *t_charge_iso, *t_emip, *t_ehcal, *t_trkL3mindr; 
  std::vector<int>           *t_ieta;
  std::vector<double>        *t_disthotcell, *t_ietahotcell, *t_eventweight;
  std::vector<double>        *t_l1pt, *t_l1eta, *t_l1phi;
  std::vector<double>        *t_l3pt, *t_l3eta, *t_l3phi;
  std::vector<double>        *t_leadingpt, *t_leadingeta, *t_leadingphi;  
};
#endif
