#ifndef DQM_Physics_EwkTauDQM_h
#define DQM_Physics_EwkTauDQM_h

/** \class EwkTauDQM
 *
 * Booking and filling of histograms for data-quality monitoring purposes
 * in EWK tau analyses; individual channels are implemented in separate
 *Ewk..HistManager classes,
 * so far:
 *  o Z --> electron + tau-jet channel (EwkElecTauHistManager)
 *  o Z --> muon + tau-jet channel (EwkMuTauHistManager)
 *
 * \authors Letizia Lusito,
 *          Joshua Swanson,
 *          Christian Veelken
 */

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include <string>
#include <Math/VectorUtil.h>

class EwkElecTauHistManager;
class EwkMuTauHistManager;

class EwkTauDQM : public DQMEDAnalyzer {
 public:
  EwkTauDQM(const edm::ParameterSet&);
  ~EwkTauDQM();

  void bookHistograms(DQMStore::IBooker&, edm::Run const&,
                      edm::EventSetup const&) override;
  void analyze(const edm::Event&, const edm::EventSetup&);
  void endRun(const edm::Run&, const edm::EventSetup&);

 private:
  std::string dqmDirectory_;
  int maxNumWarnings_;

  EwkElecTauHistManager* elecTauHistManager_;
  EwkMuTauHistManager* muTauHistManager_;
};

//-------------------------------------------------------------------------------
// code specific to Z --> e + tau-jet channel
//-------------------------------------------------------------------------------

/** \class EwkElecTauHistManager
 *
 * Booking and filling of histograms for data-quality monitoring purposes
 * in Z --> electron + tau-jet channel
 *
 * \author Joshua Swanson
 *        (modified by Christian Veelken)
 *
 *
 *
 */

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DQMServices/Core/interface/MonitorElement.h"

#include <string>

class EwkElecTauHistManager {
 public:
  EwkElecTauHistManager(const edm::ParameterSet&);

  void bookHistograms(DQMStore::IBooker&);
  void fillHistograms(const edm::Event&, const edm::EventSetup&);
  void finalizeHistograms();

 private:
  //--- labels of electron, tau-jet and MEt collections being used
  //    in event selection and when filling histograms
  edm::InputTag triggerResultsSource_;
  edm::InputTag vertexSource_;
  edm::InputTag beamSpotSource_;
  edm::InputTag electronSource_;
  edm::InputTag tauJetSource_;
  edm::InputTag caloMEtSource_;
  edm::InputTag pfMEtSource_;

  edm::InputTag tauDiscrByLeadTrackFinding_;
  edm::InputTag tauDiscrByLeadTrackPtCut_;
  edm::InputTag tauDiscrByTrackIso_;
  edm::InputTag tauDiscrByEcalIso_;
  edm::InputTag tauDiscrAgainstElectrons_;
  edm::InputTag tauDiscrAgainstMuons_;

  //--- event selection criteria
  typedef std::vector<std::string> vstring;
  vstring hltPaths_;

  double electronEtaCut_;
  double electronPtCut_;
  double electronTrackIsoCut_;
  double electronEcalIsoCut_;
  int electronIsoMode_;

  double tauJetEtaCut_;
  double tauJetPtCut_;

  double visMassCut_;

  //--- name of DQM directory in which histograms for Z --> electron + tau-jet
  // channel get stored
  std::string dqmDirectory_;

  //--- histograms
  // MonitorElement* hNumIdElectrons_;
  MonitorElement* hElectronPt_;
  MonitorElement* hElectronEta_;
  MonitorElement* hElectronPhi_;
  MonitorElement* hElectronTrackIsoPt_;
  MonitorElement* hElectronEcalIsoPt_;
  // MonitorElement* hElectronHcalIsoPt_;

  MonitorElement* hTauJetPt_;
  MonitorElement* hTauJetEta_;
  // MonitorElement* hTauJetPhi_;
  // MonitorElement* hTauLeadTrackPt_;
  // MonitorElement* hTauTrackIsoPt_;
  // MonitorElement* hTauEcalIsoPt_;
  // MonitorElement* hTauDiscrAgainstElectrons_;
  // MonitorElement* hTauDiscrAgainstMuons_;
  // MonitorElement* hTauJetCharge_;
  // MonitorElement* hTauJetNumSignalTracks_;
  // MonitorElement* hTauJetNumIsoTracks_;

  MonitorElement* hVisMass_;
  // MonitorElement* hMtElecCaloMEt_;
  MonitorElement* hMtElecPFMEt_;
  // MonitorElement* hPzetaCaloMEt_;
  // MonitorElement* hPzetaPFMEt_;
  MonitorElement* hElecTauAcoplanarity_;
  MonitorElement* hElecTauCharge_;

  // MonitorElement* hVertexChi2_;
  MonitorElement* hVertexZ_;
  // MonitorElement* hVertexD0_;

  MonitorElement* hCaloMEtPt_;
  // MonitorElement* hCaloMEtPhi_;

  MonitorElement* hPFMEtPt_;
  // MonitorElement* hPFMEtPhi_;

  MonitorElement* hCutFlowSummary_;
  enum {
    kPassedPreselection = 1,
    kPassedTrigger = 2,
    kPassedElectronId = 3,
    kPassedElectronTrackIso = 4,
    kPassedElectronEcalIso = 5,
    kPassedTauLeadTrack = 6,
    kPassedTauLeadTrackPt = 7,
    kPassedTauDiscrAgainstElectrons = 8,
    kPassedTauDiscrAgainstMuons = 9,
    kPassedTauTrackIso = 10,
    kPassedTauEcalIso = 11
  };

  //--- counters for filter-statistics output
  unsigned numEventsAnalyzed_;
  unsigned numEventsSelected_;

  int cfgError_;

  int maxNumWarnings_;

  long numWarningsTriggerResults_;
  long numWarningsHLTpath_;
  long numWarningsVertex_;
  long numWarningsBeamSpot_;
  long numWarningsElectron_;
  long numWarningsTauJet_;
  long numWarningsTauDiscrByLeadTrackFinding_;
  long numWarningsTauDiscrByLeadTrackPtCut_;
  long numWarningsTauDiscrByTrackIso_;
  long numWarningsTauDiscrByEcalIso_;
  long numWarningsTauDiscrAgainstElectrons_;
  long numWarningsTauDiscrAgainstMuons_;
  long numWarningsCaloMEt_;
  long numWarningsPFMEt_;
};

//-------------------------------------------------------------------------------
// code specific to Z --> mu + tau-jet channel
//-------------------------------------------------------------------------------

/** \class EwkMuTauHistManager
 *
 * Booking and filling of histograms for data-quality monitoring purposes
 * in Z --> muon + tau-jet channel
 *
 * \author Letizia Lusito,
 *         Christian Veelken
 *
 *
 *
 */

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DQMServices/Core/interface/MonitorElement.h"

#include <string>

class EwkMuTauHistManager {
 public:
  EwkMuTauHistManager(const edm::ParameterSet&);

  void bookHistograms(DQMStore::IBooker&);
  void fillHistograms(const edm::Event&, const edm::EventSetup&);
  void finalizeHistograms();

 private:
  //--- labels of muon, tau-jet and MEt collections being used
  //    in event selection and when filling histograms
  edm::InputTag triggerResultsSource_;
  edm::InputTag vertexSource_;
  edm::InputTag beamSpotSource_;
  edm::InputTag muonSource_;
  edm::InputTag tauJetSource_;
  edm::InputTag caloMEtSource_;
  edm::InputTag pfMEtSource_;

  edm::InputTag tauDiscrByLeadTrackFinding_;
  edm::InputTag tauDiscrByLeadTrackPtCut_;
  edm::InputTag tauDiscrByTrackIso_;
  edm::InputTag tauDiscrByEcalIso_;
  edm::InputTag tauDiscrAgainstMuons_;

  //--- event selection criteria
  typedef std::vector<std::string> vstring;
  vstring hltPaths_;

  double muonEtaCut_;
  double muonPtCut_;
  double muonTrackIsoCut_;
  double muonEcalIsoCut_;
  double muonCombIsoCut_;
  int muonIsoMode_;

  double tauJetEtaCut_;
  double tauJetPtCut_;

  double visMassCut_;
  double deltaRCut_;

  //--- name of DQM directory in which histograms for Z --> muon + tau-jet
  // channel get stored
  std::string dqmDirectory_;

  //--- histograms
  // MonitorElement* hNumGlobalMuons_;
  MonitorElement* hMuonPt_;
  MonitorElement* hMuonEta_;
  MonitorElement* hMuonPhi_;
  MonitorElement* hMuonTrackIsoPt_;
  MonitorElement* hMuonEcalIsoPt_;
  MonitorElement* hMuonCombIsoPt_;

  MonitorElement* hTauJetPt_;
  MonitorElement* hTauJetEta_;
  MonitorElement* hTauJetPhi_;
  MonitorElement* hTauLeadTrackPt_;
  MonitorElement* hTauTrackIsoPt_;
  MonitorElement* hTauEcalIsoPt_;
  MonitorElement* hTauDiscrAgainstMuons_;
  MonitorElement* hTauJetCharge_;
  MonitorElement* hTauJetNumSignalTracks_;
  MonitorElement* hTauJetNumIsoTracks_;

  MonitorElement* hVisMass_;
  MonitorElement* hMuTauDeltaR_;
  MonitorElement* hVisMassFinal_;
  // MonitorElement* hMtMuCaloMEt_;
  MonitorElement* hMtMuPFMEt_;
  // MonitorElement* hPzetaCmaxNumWarnings_aloMEt_;
  // MonitorElement* hPzetaPFMEt_;
  MonitorElement* hMuTauAcoplanarity_;
  // MonitorElement* hMuTauCharge_;

  // MonitorElement* hVertexChi2_;
  MonitorElement* hVertexZ_;
  // MonitorElement* hVertexD0_;

  MonitorElement* hCaloMEtPt_;
  // MonitorElement* hCaloMEtPhi_;

  MonitorElement* hPFMEtPt_;
  // MonitorElement* hPFMEtPhi_;

  MonitorElement* hCutFlowSummary_;
  enum {
    kPassedPreselection = 1,
    kPassedTrigger = 2,
    kPassedMuonId = 3,
    kPassedTauLeadTrack = 4,
    kPassedTauLeadTrackPt = 5,
    kPassedTauDiscrAgainstMuons = 6,
    kPassedDeltaR = 7,
    kPassedMuonTrackIso = 8,
    kPassedMuonEcalIso = 9,
    kPassedTauTrackIso = 10,
    kPassedTauEcalIso = 11
  };

  //--- counters for filter-statistics output
  unsigned numEventsAnalyzed_;
  unsigned numEventsSelected_;

  int cfgError_;

  int maxNumWarnings_;

  long numWarningsTriggerResults_;
  long numWarningsHLTpath_;
  long numWarningsVertex_;
  long numWarningsBeamSpot_;
  long numWarningsMuon_;
  long numWarningsTauJet_;
  long numWarningsTauDiscrByLeadTrackFinding_;
  long numWarningsTauDiscrByLeadTrackPtCut_;
  long numWarningsTauDiscrByTrackIso_;
  long numWarningsTauDiscrByEcalIso_;
  long numWarningsTauDiscrAgainstMuons_;
  long numWarningsCaloMEt_;
  long numWarningsPFMEt_;
};

//-------------------------------------------------------------------------------
// common auxiliary functions used by different channels
//-------------------------------------------------------------------------------

/**
 *
 * Auxiliary functions to compute quantities used by EWK Tau DQM
 * (shared by different channels)
 *
 * \author Joshua Swanson
 *
 *
 *
 */

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include <string>

enum { kAbsoluteIso, kRelativeIso, kUndefinedIso };

template <typename T>
void readEventData(const edm::Event& evt, const edm::InputTag& src,
                   edm::Handle<T>& handle, long& numWarnings,
                   int maxNumWarnings, bool& error, const char* errorMessage) {
  if (!evt.getByLabel(src, handle)) {
    if (numWarnings < maxNumWarnings || maxNumWarnings == -1)
      edm::LogWarning("readEventData") << errorMessage << " !!";
    ++numWarnings;
    error = true;
  }
}

int getIsoMode(const std::string&, int&);

double calcDeltaPhi(double, double);
double calcMt(double, double, double, double);
double calcPzeta(const reco::Candidate::LorentzVector&,
                 const reco::Candidate::LorentzVector&, double, double);

bool passesElectronPreId(const reco::GsfElectron&);
bool passesElectronId(const reco::GsfElectron&);

const reco::GsfElectron* getTheElectron(const reco::GsfElectronCollection&,
                                        double, double);
const reco::Muon* getTheMuon(const reco::MuonCollection&, double, double);
const reco::PFTau* getTheTauJet(const reco::PFTauCollection&, double, double,
                                int&);

double getVertexD0(const reco::Vertex&, const reco::BeamSpot&);

#endif
