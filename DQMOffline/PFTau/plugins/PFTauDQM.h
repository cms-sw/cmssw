#ifndef DQMOffline_PFTau_PFTauDQM_h
#define DQMOffline_PFTau_PFTauDQM_h

/** \class PFTauDQM
 *
 * Booking and filling of histograms for data-quality monitoring purposes
 * of observables relevant for particle-flow based tau id.
 * 
 * \authors Christian Veelken
 *
 * \version $Revision: 1.2 $
 *
 * $Id: PFTauDQM.h,v 1.2 2010/09/08 23:47:17 edelhoff Exp $
 *
 */

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DQMServices/Core/interface/MonitorElement.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include <TH1.h>

#include <string>

class PFTauDQM : public edm::EDAnalyzer 
{
 public:
  PFTauDQM(const edm::ParameterSet&);
  ~PFTauDQM();

  void beginJob();
  void analyze(const edm::Event&, const edm::EventSetup&);
  void endJob();
  
 private:
//--- pointer to DQM histogram management service
  DQMStore* dqmStore_;

//--- name of DQM directory in which histograms for Z --> electron + tau-jet channel get stored
  std::string dqmDirectory_;

  int dqmError_;

  long numWarningsTriggerResults_;
  long numWarningsHLTpath_;
  long numWarningsVertex_;
  long numWarningsTauJet_;
  long numWarningsHPSTauJet_;
  long numWarningsTauDiscrByLeadTrackFinding_;
  long numWarningsTauDiscrByLeadTrackPtCut_;
  long numWarningsTauDiscrByTrackIso_;
  long numWarningsTauDiscrByEcalIso_;
  long numWarningsTauDiscrAgainstElectrons_;
  long numWarningsTauDiscrAgainstMuons_; 
  long numWarningsTauDiscrTaNC_;
  long numWarningsTauDiscrTaNCWorkingPoint_;  
  long numWarningsTauDiscrHPSWorkingPoint_;

  int maxNumWarnings_;

//--- labels of tau-jet collection being used
//    in event selection and when filling histograms
  edm::InputTag triggerResultsSource_;
  edm::InputTag vertexSource_;
  edm::InputTag tauJetSource_;
  edm::InputTag hpsTauJetSource_;

  edm::InputTag tauDiscrByLeadTrackFinding_;
  edm::InputTag tauDiscrByLeadTrackPtCut_;
  edm::InputTag tauDiscrByTrackIso_;
  edm::InputTag tauDiscrByEcalIso_;
  edm::InputTag tauDiscrAgainstElectrons_;
  edm::InputTag tauDiscrAgainstMuons_;
  edm::InputTag tauDiscrTaNC_;
  edm::InputTag tauDiscrTaNCWorkingPoint_;
  edm::InputTag tauDiscrHPSWorkingPoint_;

//--- event selection criteria
  typedef std::vector<std::string> vstring;
  vstring hltPaths_;

  double tauJetPtCut_;
  double tauJetEtaCut_;
  double tauJetLeadTrkDxyCut_;
  double tauJetLeadTrkDzCut_;
  
//--- histograms  
  MonitorElement* hNumTauJets_;

  TH1* hJetPt_;
  TH1* hJetEta_;
  TH1* hJetPhi_;

  MonitorElement* hTauJetPt_;
  MonitorElement* hTauJetEta_;
  MonitorElement* hTauJetPhi_;

  TH1* hTauJetDiscrPassedPt_;
  TH1* hTauJetDiscrPassedEta_;
  TH1* hTauJetDiscrPassedPhi_;

  TH1* hTauTaNCDiscrPassedPt_;
  TH1* hTauTaNCDiscrPassedEta_;
  TH1* hTauTaNCDiscrPassedPhi_;

  TH1* hTauHPSDiscrPassedPt_;
  TH1* hTauHPSDiscrPassedEta_;
  TH1* hTauHPSDiscrPassedPhi_;

  MonitorElement* hTauJetCharge_;

  MonitorElement* hTauLeadTrackPt_;

  MonitorElement* hTauJetNumSignalTracks_;
  MonitorElement* hTauJetNumIsoTracks_;

  MonitorElement* hTauTrackIsoPt_;
  MonitorElement* hTauEcalIsoPt_;

  MonitorElement* hTauDiscrByLeadTrackFinding_;
  MonitorElement* hTauDiscrByLeadTrackPtCut_;
  MonitorElement* hTauDiscrByTrackIso_;
  MonitorElement* hTauDiscrByEcalIso_;
  MonitorElement* hTauDiscrAgainstElectrons_;
  MonitorElement* hTauDiscrAgainstMuons_;
  MonitorElement* hTauDiscrTaNC_;
  MonitorElement* hTauDiscrHPS_;

  MonitorElement* hFakeRatePt_;
  MonitorElement* hFakeRateEta_;
  MonitorElement* hFakeRatePhi_;
  MonitorElement* hTaNCFakeRatePt_;
  MonitorElement* hTaNCFakeRateEta_;
  MonitorElement* hTaNCFakeRatePhi_;
  MonitorElement* hHPSFakeRatePt_;
  MonitorElement* hHPSFakeRateEta_;
  MonitorElement* hHPSFakeRatePhi_;


  long numEventsAnalyzed_;
};

#endif     
