//
// Original Author:  John Alison, Mia Tosi
//         Created:  27 July 2020
//
//

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/Scalers/interface/LumiScalers.h"
#include "DataFormats/OnlineMetaData/interface/OnlineLuminosityRecord.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

// system include files
#include <memory>
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
}  // namespace reco
class DQMStore;
class GenericTriggerEventFlag;

class TrackToTrackComparisonHists : public DQMEDAnalyzer {
public:
  struct generalME {
    std::string label;
    MonitorElement *h_tracks, *h_pt, *h_eta, *h_phi, *h_dxy, *h_dz, *h_dxyWRTpv, *h_dzWRTpv, *h_charge, *h_hits;
    MonitorElement *h_dRmin, *h_dRmin_l;
    MonitorElement* h_pt_vs_eta;
    MonitorElement *h_onlinelumi, *h_PU, *h_ls;
  };

  struct matchingME {
    std::string label;
    MonitorElement *h_hits_vs_hits, *h_pt_vs_pt, *h_eta_vs_eta, *h_phi_vs_phi;
    MonitorElement *h_dPt, *h_dEta, *h_dPhi, *h_dDxy, *h_dDz, *h_dDxyWRTpv, *h_dDzWRTpv, *h_dCharge, *h_dHits;
  };

  typedef std::vector<std::pair<int, std::map<double, int> > > idx2idxByDoubleColl;

  explicit TrackToTrackComparisonHists(const edm::ParameterSet&);
  ~TrackToTrackComparisonHists() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  static void fillHistoPSetDescription(edm::ParameterSetDescription& pset);

protected:
  void beginJob(const edm::EventSetup& iSetup);
  void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) override;
  void bookHistograms(DQMStore::IBooker& iBooker, edm::Run const& iRun, edm::EventSetup const& iSetup) override;

  void fillMap(reco::TrackCollection tracks1, reco::TrackCollection tracks2, idx2idxByDoubleColl& map, float dRMin);

  void initialize_parameter(const edm::ParameterSet& iConfig);
  void bookHistos(DQMStore::IBooker& ibooker, generalME& mes, TString label, std::string& dir);
  void book_generic_tracks_histos(DQMStore::IBooker& ibooker, generalME& mes, TString label, std::string& dir);
  void book_matching_tracks_histos(DQMStore::IBooker& ibooker, matchingME& mes, TString label, std::string& dir);

  void fill_generic_tracks_histos(generalME& mes,
                                  reco::Track* trk,
                                  reco::BeamSpot* bs,
                                  reco::Vertex* pv,
                                  unsigned int ls,
                                  double onlinelumi,
                                  double PU,
                                  bool requirePlateau = true);
  void fill_matching_tracks_histos(
      matchingME& mes, reco::Track* mon, reco::Track* ref, reco::BeamSpot* bs, reco::Vertex* pv);

  DQMStore* dqmStore_;

  edm::InputTag monitoredTrackInputTag_;
  edm::InputTag referenceTrackInputTag_;

  //these are used by MTVGenPs
  edm::EDGetTokenT<reco::TrackCollection> monitoredTrackToken_;
  edm::EDGetTokenT<reco::TrackCollection> referenceTrackToken_;
  edm::EDGetTokenT<reco::BeamSpot> monitoredBSToken_;
  edm::EDGetTokenT<reco::BeamSpot> referenceBSToken_;
  edm::EDGetTokenT<reco::VertexCollection> monitoredPVToken_;
  edm::EDGetTokenT<reco::VertexCollection> referencePVToken_;
  edm::EDGetTokenT<LumiScalersCollection> lumiScalersToken_;
  edm::EDGetTokenT<OnlineLuminosityRecord> onlineMetaDataDigisToken_;

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

  double Eta_rangeMin, Eta_rangeMax;
  unsigned int Eta_nbin;
  double Pt_rangeMin, Pt_rangeMax;
  unsigned int Pt_nbin;  //bool useInvPt;   bool useLogPt;
  double Phi_rangeMin, Phi_rangeMax;
  unsigned int Phi_nbin;
  double Dxy_rangeMin, Dxy_rangeMax;
  unsigned int Dxy_nbin;
  double Dz_rangeMin, Dz_rangeMax;
  unsigned int Dz_nbin;

  double ptRes_rangeMin, ptRes_rangeMax;
  unsigned int ptRes_nbin;
  double phiRes_rangeMin, phiRes_rangeMax;
  unsigned int phiRes_nbin;
  double etaRes_rangeMin, etaRes_rangeMax;
  unsigned int etaRes_nbin;
  double dxyRes_rangeMin, dxyRes_rangeMax;
  unsigned int dxyRes_nbin;
  double dzRes_rangeMin, dzRes_rangeMax;
  unsigned int dzRes_nbin;
  unsigned int ls_rangeMin, ls_rangeMax;
  unsigned int ls_nbin;
  double PU_rangeMin, PU_rangeMax;
  unsigned int PU_nbin;
  double onlinelumi_rangeMin, onlinelumi_rangeMax;
  unsigned int onlinelumi_nbin;
};
