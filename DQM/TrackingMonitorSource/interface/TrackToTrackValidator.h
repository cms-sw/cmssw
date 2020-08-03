//
// Original Author:  John Alison, Mia Tosi
//         Created:  27 July 2020
//
//


// system include files
#include <DQMServices/Core/interface/DQMEDAnalyzer.h>
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Utilities/interface/EDGetToken.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include <iostream>
#include <sstream>
#include <string>

//
// class declaration
//
class DQMStore;
namespace reco {
  class Track;
  class BeamSpot;
  class Vertex;
}
class DQMStore;
class GenericTriggerEventFlag;

class TrackToTrackValidator : public DQMEDAnalyzer {


 public:

  struct generalME {
    std::string label;
    MonitorElement *h_tracks, *h_pt, *h_eta, *h_phi, *h_dxy, *h_dz, *h_dxyWRTpv, *h_dzWRTpv, *h_charge, *h_hits;
    MonitorElement *h_dRmin, *h_dRmin_l;
    MonitorElement *h_pt_vs_eta;
  };
  
  struct matchingME {
    std::string label;
    MonitorElement *h_hits_vs_hits, *h_pt_vs_pt, *h_eta_vs_eta, *h_phi_vs_phi;
    MonitorElement *h_dPt, *h_dEta, *h_dPhi, *h_dDxy, *h_dDz, *h_dDxyWRTpv, *h_dDzWRTpv, *h_dCharge, *h_dHits;
  };
  
  typedef std::vector<std::pair<int, std::map<double, int> > > idx2idxByDoubleColl;

  explicit TrackToTrackValidator(const edm::ParameterSet&);
  ~TrackToTrackValidator();

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  static void fillHistoPSetDescription(edm::ParameterSetDescription& pset);

 protected:

  void beginJob(const edm::EventSetup& iSetup);
  void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup);
  void bookHistograms(DQMStore::IBooker & iBooker, edm::Run const & iRun, edm::EventSetup const & iSetup) override;

  void fillMap(reco::TrackCollection tracks1, reco::TrackCollection tracks2, idx2idxByDoubleColl& map, float dRMin);

  void initialize_parameter(const edm::ParameterSet& iConfig);
  void bookHistos(DQMStore::IBooker & ibooker, generalME& mes, TString label, std::string & dir);
  void book_generic_tracks_histos(DQMStore::IBooker & ibooker, generalME& mes, TString label, std::string & dir);
  void book_matching_tracks_histos(DQMStore::IBooker & ibooker, matchingME& mes, TString label, std::string & dir);

  void fill_generic_tracks_histos(generalME& mes, reco::Track* trk, reco::BeamSpot* bs, reco::Vertex* pv, bool requirePlateau = true);
  void fill_matching_tracks_histos(matchingME& mes, reco::Track* mon, reco::Track* ref, reco::BeamSpot* bs, reco::Vertex* pv);

  DQMStore* dqmStore_;

  edm::InputTag monitoredTrackInputTag_;
  edm::InputTag referenceTrackInputTag_;

  //these are used by MTVGenPs
  edm::EDGetTokenT<reco::TrackCollection>  monitoredTrackToken_;
  edm::EDGetTokenT<reco::TrackCollection>  referenceTrackToken_;
  edm::EDGetTokenT<reco::BeamSpot>         monitoredBSToken_;
  edm::EDGetTokenT<reco::BeamSpot>         referenceBSToken_;
  edm::EDGetTokenT<reco::VertexCollection> monitoredPVToken_;
  edm::EDGetTokenT<reco::VertexCollection> referencePVToken_;

 private:
  
  //  edm::ParameterSet conf_;
  std::string topDirName_;
  double dRmin_; 
  double pTCutForPlateau_; 
  double dxyCutForPlateau_; 
  double dzWRTPvCut_; 
  bool requireValidHLTPaths_;
  bool hltPathsAreValid_ = false;
  std::unique_ptr<GenericTriggerEventFlag> genTriggerEventFlag_;

  // reference tracks All and matched
  generalME referenceTracksMEs_;  
  generalME matchedReferenceTracksMEs_;  

  // monitored tracks All and unmatched
  generalME monitoredTracksMEs_;  
  generalME unMatchedMonitoredTracksMEs_;  

  // Track matching statistics
  matchingME matchTracksMEs_;

  double Eta_rangeMin, Eta_rangeMax;  unsigned int Eta_nbin;
  double Pt_rangeMin,  Pt_rangeMax;   unsigned int Pt_nbin;   //bool useInvPt;   bool useLogPt;
  double Phi_rangeMin, Phi_rangeMax;  unsigned int Phi_nbin;
  double Dxy_rangeMin, Dxy_rangeMax;  unsigned int Dxy_nbin;
  double Dz_rangeMin,  Dz_rangeMax;   unsigned int Dz_nbin;

  double ptRes_rangeMin,  ptRes_rangeMax;   unsigned int ptRes_nbin;
  double phiRes_rangeMin, phiRes_rangeMax;  unsigned int phiRes_nbin;
  double etaRes_rangeMin, etaRes_rangeMax;  unsigned int etaRes_nbin;
  double dxyRes_rangeMin, dxyRes_rangeMax;  unsigned int dxyRes_nbin;
  double dzRes_rangeMin,  dzRes_rangeMax;   unsigned int dzRes_nbin;

  //std::vector<int> totSIMeta,totRECeta,totASSeta,totASS2eta,totloopeta,totmisideta,totASS2etaSig;
  //std::vector<int> totSIMpT,totRECpT,totASSpT,totASS2pT,totlooppT,totmisidpT;
  //std::vector<int> totSIM_hit,totREC_hit,totASS_hit,totASS2_hit,totloop_hit,totmisid_hit;
  //std::vector<int> totSIM_phi,totREC_phi,totASS_phi,totASS2_phi,totloop_phi,totmisid_phi;
  //std::vector<int> totSIM_dxy,totREC_dxy,totASS_dxy,totASS2_dxy,totloop_dxy,totmisid_dxy;
  //std::vector<int> totSIM_dz,totREC_dz,totASS_dz,totASS2_dz,totloop_dz,totmisid_dz;

  //std::vector<int> totSIM_vertpos,totASS_vertpos,totSIM_zpos,totASS_zpos;
  //std::vector<int> totSIM_vertcount_entire,totASS_vertcount_entire,totREC_vertcount_entire,totASS2_vertcount_entire,totASS2_vertcount_entire_signal;
  //std::vector<int> totSIM_vertcount_barrel,totASS_vertcount_barrel,totREC_vertcount_barrel,totASS2_vertcount_barrel;
  //std::vector<int> totSIM_vertcount_fwdpos,totASS_vertcount_fwdpos,totREC_vertcount_fwdpos,totASS2_vertcount_fwdpos;
  //std::vector<int> totSIM_vertcount_fwdneg,totASS_vertcount_fwdneg,totREC_vertcount_fwdneg,totASS2_vertcount_fwdneg;
  //std::vector<int> totSIM_vertz_entire,totASS_vertz_entire;
  //std::vector<int> totSIM_vertz_barrel,totASS_vertz_barrel;
  //std::vector<int> totSIM_vertz_fwdpos,totASS_vertz_fwdpos;
  //std::vector<int> totSIM_vertz_fwdneg,totASS_vertz_fwdneg;
  //std::vector<int> totREC_algo;
  //std::vector<int> totREC_ootpu_entire, totASS2_ootpu_entire;
  //std::vector<int> totREC_ootpu_barrel, totASS2_ootpu_barrel;
  //std::vector<int> totREC_ootpu_fwdpos, totASS2_ootpu_fwdpos;
  //std::vector<int> totREC_ootpu_fwdneg, totASS2_ootpu_fwdneg;
  //std::vector<int> totREC_ootpu_eta_entire, totASS2_ootpu_eta_entire;
  //std::vector<int> totASS2_itpu_eta_entire, totASS2_itpu_eta_entire_signal, totASS2_itpu_vertcount_entire, totASS2_itpu_vertcount_entire_signal;
  //std::vector<int> totFOMT_eta, totFOMT_vertcount;
  //std::vector<int> totCONeta, totCONvertcount, totCONzpos;
  
};
