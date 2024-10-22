#ifndef L1TDTTF_H
#define L1TDTTF_H

/*
 * \file L1TDTTF.h
 *
 * \author J. Berryhill
 *
 */

// system include files
#include <string>

// user include files
#include "FWCore/Framework/interface/Event.h"

#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutCollection.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTTrackContainer.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
//
// class declaration
//
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"

class L1MuDTTrackCand;
class L1MuRegionalCand;

class L1TDTTF : public DQMEDAnalyzer {
public:
  // Constructor
  L1TDTTF(const edm::ParameterSet& ps);

  // Destructor
  ~L1TDTTF() override;

protected:
  // Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c) override;

  // BeginJob
  void dqmBeginRun(edm::Run const&, edm::EventSetup const&) override;
  void bookHistograms(DQMStore::IBooker& i, edm::Run const&, edm::EventSetup const&) override;

private:
  void fillMEs(std::vector<L1MuDTTrackCand> const* trackContainer, std::vector<L1MuRegionalCand>& gmtDttfCands);
  void setWheelLabel(MonitorElement* me);
  void setQualLabel(MonitorElement* me, int axis);
  void bookEta(int wh, int& nbins, float& start, float& stop);

  // ----------member data ---------------------------
  edm::InputTag dttpgSource_;
  edm::InputTag gmtSource_;
  edm::InputTag muonCollectionLabel_;
  std::string l1tsubsystemfolder_;
  bool online_;
  bool verbose_;
  std::string outputFile_;  //file name for ROOT ouput
  edm::InputTag trackInputTag_;

  MonitorElement* dttf_nTracksPerEvent_wheel[6];
  MonitorElement* dttf_quality_wheel_2ndTrack[6];
  MonitorElement* dttf_quality_summary_wheel_2ndTrack[6];
  MonitorElement* dttf_phi_eta_fine_wheel[6];
  MonitorElement* dttf_phi_eta_coarse_wheel[6];
  MonitorElement* dttf_phi_eta_wheel_2ndTrack[6];
  MonitorElement* dttf_eta_wheel_2ndTrack[6];
  MonitorElement* dttf_phi_wheel_2ndTrack[6];
  MonitorElement* dttf_pt_wheel_2ndTrack[6];
  MonitorElement* dttf_q_wheel_2ndTrack[6];

  MonitorElement* dttf_nTracksPerEv[6][12];
  MonitorElement* dttf_bx[6][12];
  MonitorElement* dttf_bx_2ndTrack[6][12];
  MonitorElement* dttf_qual[6][12];
  MonitorElement* dttf_eta_fine_fraction[6][12];
  MonitorElement* dttf_eta[6][12];
  MonitorElement* dttf_phi[6][12];
  MonitorElement* dttf_pt[6][12];
  MonitorElement* dttf_q[6][12];

  MonitorElement* dttf_nTracksPerEvent_integ;
  MonitorElement* dttf_spare;

  MonitorElement* dttf_gmt_match;
  MonitorElement* dttf_gmt_missed;
  MonitorElement* dttf_gmt_ghost;

  // MonitorElement* dttf_gmt_ghost_phys;

  int nev_;              // Number of events processed
  int nev_dttf_;         //Number of events with at least one DTTF track
  int nev_dttf_track2_;  //Number of events with at least one DTTF 2nd track
  int numTracks[6][12];

  //define Token(-s)
  edm::EDGetTokenT<L1MuDTTrackContainer> trackInputToken_;
  edm::EDGetTokenT<reco::MuonCollection> muonCollectionToken_;
  edm::EDGetTokenT<L1MuGMTReadoutCollection> gmtSourceToken_;
};

#endif
