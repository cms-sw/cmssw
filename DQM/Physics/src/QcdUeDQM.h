
/*
    This is the DQM code for UE physics plots
    11/12/2009 Sunil Bansal
*/

#ifndef QcdUeDQM_H
#define QcdUeDQM_H

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/GeometryVector/interface/VectorUtil.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "AnalysisDataFormats/TrackInfo/interface/TrackInfo.h"
#include "AnalysisDataFormats/TrackInfo/interface/TrackInfoTrackAssociation.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/CandMatchMap.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/TrackJetCollection.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"

#include <TMath.h>
#include <vector>

#define PI 3.141592654

class DQMStore;
class MonitorElement;
class TrackerGeometry;
class TH1F;
class TH2F;
class TH3F;
class TProfile;

class PtSorter {
 public:
  template <class T>
  bool operator()(const T &a, const T &b) {
    return (a.pt() > b.pt());
  }
};

class QcdUeDQM : public DQMEDAnalyzer {
 public:
  QcdUeDQM(const edm::ParameterSet &parameters);
  virtual ~QcdUeDQM();
  void dqmBeginRun(const edm::Run &, const edm::EventSetup &);
  void bookHistograms(DQMStore::IBooker &, edm::Run const &,
                      edm::EventSetup const &) override;
  void analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup);

 private:
  bool isHltConfigSuccessful_;  // to prevent processing in case of problems

  void book1D(DQMStore::IBooker &, std::vector<MonitorElement *> &mes,
              const std::string &name, const std::string &title, int nx,
              double x1, double x2, bool sumw2 = 1, bool sbox = 1);
  void bookProfile(DQMStore::IBooker &, std::vector<MonitorElement *> &mes,
                   const std::string &name, const std::string &title, int nx,
                   double x1, double x2, double y1, double y2, bool sumw2 = 1,
                   bool sbox = 1);
  void fill1D(std::vector<TH1F *> &hs, double val, double w = 1.);
  void fill1D(std::vector<MonitorElement *> &mes, double val, double w = 1.);
  void fill2D(std::vector<TH2F *> &hs, double valx, double valy, double w = 1.);
  void fill2D(std::vector<MonitorElement *> &mes, double valx, double valy,
              double w = 1.);
  void fillProfile(std::vector<TProfile *> &hs, double valx, double valy,
                   double w = 1.);
  void fillProfile(std::vector<MonitorElement *> &mes, double valx, double valy,
                   double w = 1.);
  void fill3D(std::vector<TH3F *> &hs, int gbin, double w = 1.);
  void setLabel1D(std::vector<MonitorElement *> &mes);

  bool trackSelection(const reco::Track &trk, const reco::BeamSpot *bs,
                      const reco::Vertex &vtx, int sizevtx);
  void fillHltBits(const edm::Event &iEvent, const edm::EventSetup &iSetup);
  bool fillVtxPlots(const reco::BeamSpot *bs,
                    const edm::Handle<reco::VertexCollection> vtxColl);
  void fillpTMaxRelated(const std::vector<const reco::Track *> &track);
  void fillChargedJetSpectra(
      const edm::Handle<reco::TrackJetCollection> trackJets);
  void fillUE_with_ChargedJets(
      const std::vector<const reco::Track *> &track,
      const edm::Handle<reco::TrackJetCollection> &trackJets);
  void fillUE_with_MaxpTtrack(const std::vector<const reco::Track *> &track);

  template <typename TYPE>
  void getProduct(const std::string name, edm::Handle<TYPE> &prod,
                  const edm::Event &event) const;
  template <typename TYPE>
  bool getProductSafe(const std::string name, edm::Handle<TYPE> &prod,
                      const edm::Event &event) const;

  HLTConfigProvider hltConfig;

  std::string hltResName_;                 // HLT trigger results name
  std::vector<std::string> hltProcNames_;  // HLT process name(s)
  std::vector<std::string> hltTrgNames_;   // HLT trigger name(s)

  std::vector<int> hltTrgBits_;               // HLT trigger bit(s)
  std::vector<bool> hltTrgDeci_;              // HLT trigger descision(s)
  std::vector<std::string> hltTrgUsedNames_;  // HLT used trigger name(s)
  std::string hltUsedResName_;                // used HLT trigger results name
  int verbose_;                  // verbosity (0=debug,1=warn,2=error,3=throw)
  const TrackerGeometry *tgeo_;  // tracker geometry
  MonitorElement *repSumMap_;    // report summary map
  MonitorElement *repSummary_;   // report summary
  MonitorElement *h2TrigCorr_;   // trigger correlation plot

  std::vector<MonitorElement *> hNevts_;
  std::vector<MonitorElement *> hNtrackerLayer_;
  std::vector<MonitorElement *> hNtrackerPixelLayer_;
  std::vector<MonitorElement *> hNtrackerStripPixelLayer_;
  std::vector<MonitorElement *> hRatioPtErrorPt_;
  std::vector<MonitorElement *> hTrkPt_;
  std::vector<MonitorElement *> hTrkEta_;
  std::vector<MonitorElement *> hTrkPhi_;
  std::vector<MonitorElement *> hNgoodTrk_;
  std::vector<MonitorElement *> hGoodTrkPt500_;
  std::vector<MonitorElement *> hGoodTrkEta500_;
  std::vector<MonitorElement *> hGoodTrkPhi500_;
  std::vector<MonitorElement *> hGoodTrkPt900_;
  std::vector<MonitorElement *> hGoodTrkEta900_;
  std::vector<MonitorElement *> hGoodTrkPhi900_;
  std::vector<MonitorElement *> hRatioDxySigmaDxyBS_;
  std::vector<MonitorElement *> hRatioDxySigmaDxyPV_;
  std::vector<MonitorElement *> hRatioDzSigmaDzBS_;
  std::vector<MonitorElement *> hRatioDzSigmaDzPV_;
  std::vector<MonitorElement *> hTrkChi2_;
  std::vector<MonitorElement *> hTrkNdof_;

  std::vector<MonitorElement *> hNvertices_;  // Number of vertices
  std::vector<MonitorElement *> hVertex_z_;   // z-position of vertex
  std::vector<MonitorElement *> hVertex_y_;
  std::vector<MonitorElement *> hVertex_x_;
  std::vector<MonitorElement *> hVertex_ndof_;
  std::vector<MonitorElement *> hVertex_rho_;
  std::vector<MonitorElement *> hVertex_z_bs_;

  std::vector<MonitorElement *> hBeamSpot_z_;  // z-position of vertex
  std::vector<MonitorElement *> hBeamSpot_y_;
  std::vector<MonitorElement *> hBeamSpot_x_;

  std::vector<MonitorElement *> hLeadingTrack_pTSpectrum_;   // pt spectrum of
                                                             // leading track
  std::vector<MonitorElement *> hLeadingTrack_etaSpectrum_;  // eta spectrum of
                                                             // leading track
  std::vector<MonitorElement *> hLeadingTrack_phiSpectrum_;  // phi spectrum of
                                                             // leading track

  std::vector<MonitorElement *> hChargedJetMulti_;  // Number of charged jets
  std::vector<MonitorElement *> hChargedJetConstituent_;  // Number of
                                                          // constituent of
                                                          // charged jets
  std::vector<MonitorElement *> hLeadingChargedJet_pTSpectrum_;  // pT spectrum
                                                                 // of charged
                                                                 // jets

  /*std::vector<MonitorElement*>   hCaloJetMulti_;         // Number of calo
  jets
  std::vector<MonitorElement*>   hCaloJetConstituent_;         // Number of
  constituent of calo jets
  std::vector<MonitorElement*>   hLeadingCaloJet_pTSpectrum_;         // pT
  spectrum of calo jets
  std::vector<MonitorElement*>  hLeadingCaloJet_etaSpectrum_;      //eta
  spectrum of leading calo jet
  std::vector<MonitorElement*>  hLeadingCaloJet_phiSpectrum_;      //phi
  spectrum of leading calo jet
 */
  std::vector<MonitorElement *> hdPhi_maxpTTrack_tracks_;  // delta phi between
                                                           // leading track and
                                                           // tracks
  // std::vector<MonitorElement*>  hdPhi_caloJet_tracks_;  // delta phi between
  // leading calo jet and tracks
  std::vector<MonitorElement *> hdPhi_chargedJet_tracks_;  // delta phi between
                                                           // leading charged
                                                           // jet and tracks

  std::vector<MonitorElement *> hLeadingChargedJet_etaSpectrum_;  // eta
                                                                  // spectrum of
                                                                  // leading
                                                                  // charged jet
  std::vector<MonitorElement *> hLeadingChargedJet_phiSpectrum_;  // phi
                                                                  // spectrum of
                                                                  // leading
                                                                  // charged jet

  std::vector<MonitorElement *> hdNdEtadPhi_pTMax_Toward500_;  // number of
                                                               // tracks in
                                                               // toward region
                                                               // of leadin
                                                               // track (pT >
                                                               // 500 MeV)
  std::vector<MonitorElement *> hdNdEtadPhi_pTMax_Transverse500_;  // number of
                                                                   // tracks in
                                                                   // transverse
                                                                   // region of
                                                                   // leadin
                                                                   // track (pT
                                                                   // > 500 MeV)
  std::vector<MonitorElement *> hdNdEtadPhi_pTMax_Away500_;  // number of tracks
                                                             // in away region
                                                             // of leadin track
                                                             // (pT > 500 MeV)
  /* std::vector<MonitorElement*>  hdNdEtadPhi_caloJet_Toward500_;    // number
   of tracks in toward region of leadin calo Jet (pT > 500 MeV)
   std::vector<MonitorElement*>  hdNdEtadPhi_caloJet_Transverse500_;    //
   number of tracks in transverse region of leadin calo Jet (pT > 500 MeV)
   std::vector<MonitorElement*>  hdNdEtadPhi_caloJet_Away500_;    // number of
   tracks in away region of leadin calo Jet (pT > 500 MeV)
*/
  std::vector<MonitorElement *> hdNdEtadPhi_trackJet_Toward500_;  // number of
                                                                  // tracks in
                                                                  // toward
                                                                  // region of
                                                                  // leadin calo
                                                                  // Jet (pT >
                                                                  // 500 MeV)
  std::vector<MonitorElement *>
      hdNdEtadPhi_trackJet_Transverse500_;  // number of tracks in transverse
                                            // region of leadin calo Jet (pT >
                                            // 500 MeV)
  std::vector<MonitorElement *> hdNdEtadPhi_trackJet_Away500_;  // number of
                                                                // tracks in
                                                                // away region
                                                                // of leadin
                                                                // calo Jet (pT
                                                                // > 500 MeV)

  std::vector<MonitorElement *> hpTSumdEtadPhi_pTMax_Toward500_;  // pT sum of
                                                                  // tracks in
                                                                  // toward
                                                                  // region of
                                                                  // leadin
                                                                  // track (pT >
                                                                  // 500 MeV)
  std::vector<MonitorElement *>
      hpTSumdEtadPhi_pTMax_Transverse500_;  // pT sum of tracks in transverse
                                            // region of leadin track (pT > 500
                                            // MeV)
  std::vector<MonitorElement *> hpTSumdEtadPhi_pTMax_Away500_;  // pT sum of
                                                                // tracks in
                                                                // away region
                                                                // of leadin
                                                                // track (pT >
                                                                // 500 MeV)
  /*  std::vector<MonitorElement*>  hpTSumdEtadPhi_caloJet_Toward500_;    // pT
    sum of tracks in toward region of leadin calo Jet (pT > 500 MeV)
    std::vector<MonitorElement*>  hpTSumdEtadPhi_caloJet_Transverse500_;    //
    pT sum of tracks in transverse region of leadin calo Jet (pT > 500 MeV)
    std::vector<MonitorElement*>  hpTSumdEtadPhi_caloJet_Away500_;    // pT sum
    of tracks in away region of leadin calo Jet (pT > 500 MeV)
*/
  std::vector<MonitorElement *> hpTSumdEtadPhi_trackJet_Toward500_;  // pT sum
                                                                     // of
                                                                     // tracks
                                                                     // in
                                                                     // toward
                                                                     // region
                                                                     // of
                                                                     // leadin
                                                                     // calo Jet
                                                                     // (pT >
                                                                     // 500 MeV)
  std::vector<MonitorElement *>
      hpTSumdEtadPhi_trackJet_Transverse500_;  // pT sum of tracks in transverse
                                               // region of leadin calo Jet (pT
                                               // > 500 MeV)
  std::vector<MonitorElement *> hpTSumdEtadPhi_trackJet_Away500_;  // pT sum of
                                                                   // tracks in
                                                                   // away
                                                                   // region of
                                                                   // leadin
                                                                   // calo Jet
                                                                   // (pT > 500
                                                                   // MeV)

  std::vector<MonitorElement *> hdNdEtadPhi_pTMax_Toward900_;  // number of
                                                               // tracks in
                                                               // toward region
                                                               // of leadin
                                                               // track (pT >
                                                               // 900 MeV)
  std::vector<MonitorElement *> hdNdEtadPhi_pTMax_Transverse900_;  // number of
                                                                   // tracks in
                                                                   // transverse
                                                                   // region of
                                                                   // leadin
                                                                   // track (pT
                                                                   // > 900 MeV)
  std::vector<MonitorElement *> hdNdEtadPhi_pTMax_Away900_;  // number of tracks
                                                             // in away region
                                                             // of leadin track
                                                             // (pT > 900 MeV)
  /*  std::vector<MonitorElement*>  hdNdEtadPhi_caloJet_Toward900_;    // number
    of tracks in toward region of leadin calo Jet (pT > 900 MeV)
    std::vector<MonitorElement*>  hdNdEtadPhi_caloJet_Transverse900_;    //
    number of tracks in transverse region of leadin calo Jet (pT > 900 MeV)
    std::vector<MonitorElement*>  hdNdEtadPhi_caloJet_Away900_;    // number of
    tracks in away region of leadin calo Jet (pT > 900 MeV)
*/
  std::vector<MonitorElement *> hdNdEtadPhi_trackJet_Toward900_;  // number of
                                                                  // tracks in
                                                                  // toward
                                                                  // region of
                                                                  // leadin calo
                                                                  // Jet (pT >
                                                                  // 900 MeV)
  std::vector<MonitorElement *>
      hdNdEtadPhi_trackJet_Transverse900_;  // number of tracks in transverse
                                            // region of leadin calo Jet (pT >
                                            // 900 MeV)
  std::vector<MonitorElement *> hdNdEtadPhi_trackJet_Away900_;  // number of
                                                                // tracks in
                                                                // away region
                                                                // of leadin
                                                                // calo Jet (pT
                                                                // > 900 MeV)

  std::vector<MonitorElement *> hpTSumdEtadPhi_pTMax_Toward900_;  // pT sum of
                                                                  // tracks in
                                                                  // toward
                                                                  // region of
                                                                  // leadin
                                                                  // track (pT >
                                                                  // 900 MeV)
  std::vector<MonitorElement *>
      hpTSumdEtadPhi_pTMax_Transverse900_;  // pT sum of tracks in transverse
                                            // region of leadin track (pT > 900
                                            // MeV)
  std::vector<MonitorElement *> hpTSumdEtadPhi_pTMax_Away900_;  // pT sum of
                                                                // tracks in
                                                                // away region
                                                                // of leadin
                                                                // track (pT >
                                                                // 900 MeV)
  /*  std::vector<MonitorElement*>  hpTSumdEtadPhi_caloJet_Toward900_;    // pT
    sum of tracks in toward region of leadin calo Jet (pT > 900 MeV)
    std::vector<MonitorElement*>  hpTSumdEtadPhi_caloJet_Transverse900_;    //
    pT sum of tracks in transverse region of leadin calo Jet (pT > 900 MeV)
    std::vector<MonitorElement*>  hpTSumdEtadPhi_caloJet_Away900_;    // pT sum
    of tracks in away region of leadin calo Jet (pT > 900 MeV)
*/
  std::vector<MonitorElement *> hpTSumdEtadPhi_trackJet_Toward900_;  // pT sum
                                                                     // of
                                                                     // tracks
                                                                     // in
                                                                     // toward
                                                                     // region
                                                                     // of
                                                                     // leadin
                                                                     // calo Jet
                                                                     // (pT >
                                                                     // 900 MeV)
  std::vector<MonitorElement *>
      hpTSumdEtadPhi_trackJet_Transverse900_;  // pT sum of tracks in transverse
                                               // region of leadin calo Jet (pT
                                               // > 900 MeV)
  std::vector<MonitorElement *> hpTSumdEtadPhi_trackJet_Away900_;  // pT sum of
                                                                   // tracks in
                                                                   // away
                                                                   // region of
                                                                   // leadin
                                                                   // calo Jet
                                                                   // (pT > 900
                                                                   // MeV)

  double ptMin_;
  double minRapidity_;
  double maxRapidity_;
  double tip_;
  double lip_;
  double diffvtxbs_;
  double ptErr_pt_;
  double vtxntk_;
  int minHit_;
  double pxlLayerMinCut_;
  bool requirePIX1_;
  int min3DHit_;
  double maxChi2_;
  bool bsuse_;
  bool allowTriplets_;
  double bsPos_;
  edm::EDGetTokenT<reco::CaloJetCollection> caloJetLabel_;
  edm::EDGetTokenT<reco::TrackJetCollection> chargedJetLabel_;
  edm::EDGetTokenT<reco::TrackCollection> trackLabel_;
  edm::EDGetTokenT<reco::VertexCollection> vtxLabel_;
  edm::EDGetTokenT<reco::BeamSpot> bsLabel_;
  std::vector<reco::TrackBase::TrackQuality> quality_;
  std::vector<reco::TrackBase::TrackAlgorithm> algorithm_;
  typedef std::vector<const reco::Track *> container;
  container selected_;
  reco::Vertex vtx1;
};

//--------------------------------------------------------------------------------------------------
template <typename TYPE>
inline void QcdUeDQM::getProduct(const std::string name,
                                 edm::Handle<TYPE> &prod,
                                 const edm::Event &event) const {
  // Try to access data collection from EDM file. We check if we really get just
  // one
  // product with the given name. If not we throw an exception.

  event.getByLabel(edm::InputTag(name), prod);
  if (!prod.isValid())
    throw edm::Exception(edm::errors::Configuration, "QcdUeDQM::GetProduct()\n")
        << "Collection with label " << name << " is not valid" << std::endl;
}

//--------------------------------------------------------------------------------------------------
template <typename TYPE>
inline bool QcdUeDQM::getProductSafe(const std::string name,
                                     edm::Handle<TYPE> &prod,
                                     const edm::Event &event) const {
  // Try to safely access data collection from EDM file. We check if we really
  // get just one
  // product with the given name. If not, we return false.

  if (name.size() == 0) return false;

  try {
    event.getByLabel(edm::InputTag(name), prod);
    if (!prod.isValid()) return false;
  } catch (...) {
    return false;
  }
  return true;
}

//--------------------------------------------------------------------------------------------------
#endif

// Local Variables:
// show-trailing-whitespace: t
// truncate-lines: t
// End:
