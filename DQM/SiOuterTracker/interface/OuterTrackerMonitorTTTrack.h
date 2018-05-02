#ifndef SiOuterTracker_OuterTrackerMonitorTTTrack_h
#define SiOuterTracker_OuterTrackerMonitorTTTrack_h

#include <vector>
#include <memory>
#include <string>
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"


class DQMStore;

class OuterTrackerMonitorTTTrack : public DQMEDAnalyzer {

public:
  explicit OuterTrackerMonitorTTTrack(const edm::ParameterSet&);
  ~OuterTrackerMonitorTTTrack() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;

  // Distributions of all tracks
  MonitorElement* Track_NStubs = nullptr; // Number of stubs per track
  MonitorElement* Track_Eta_NStubs = nullptr; //Number of stubs per track vs eta

  /// Low-quality TTTracks (All tracks)
  MonitorElement* Track_LQ_N = nullptr; // Number of tracks per event
  MonitorElement* Track_LQ_Pt = nullptr; // pT distrubtion for tracks
  MonitorElement* Track_LQ_Eta = nullptr; // eta distrubtion for tracks
  MonitorElement* Track_LQ_Phi = nullptr; // phi distrubtion for tracks
  MonitorElement* Track_LQ_D0 = nullptr; // d0 distrubtion for tracks
  MonitorElement* Track_LQ_VtxZ = nullptr; // z0 distrubtion for tracks
  MonitorElement* Track_LQ_Chi2 = nullptr;  // chi2 distrubtion for tracks
  MonitorElement* Track_LQ_Chi2Red = nullptr; // chi2/dof distrubtion for tracks
  MonitorElement* Track_LQ_Chi2Red_NStubs = nullptr; // chi2/dof vs number of stubs
  MonitorElement* Track_LQ_Chi2Red_Eta = nullptr; // chi2/dof vs eta of track
  MonitorElement* Track_LQ_Eta_BarrelStubs = nullptr; // eta vs number of stubs in barrel
  MonitorElement* Track_LQ_Eta_ECStubs = nullptr; // eta vs number of stubs in end caps
  MonitorElement* Track_LQ_Chi2_Probability = nullptr; // chi2 probability

  /// High-quality TTTracks (NStubs >=5, chi2/dof<10)
  MonitorElement* Track_HQ_N = nullptr; // Number of tracks per event
  MonitorElement* Track_HQ_Pt = nullptr;  // pT distrubtion for tracks
  MonitorElement* Track_HQ_Eta = nullptr; // eta distrubtion for tracks
  MonitorElement* Track_HQ_Phi = nullptr; // phi distrubtion for tracks
  MonitorElement* Track_HQ_D0 = nullptr; // d0 distrubtion for tracks
  MonitorElement* Track_HQ_VtxZ = nullptr; // z0 distrubtion for tracks
  MonitorElement* Track_HQ_Chi2 = nullptr; // chi2 distrubtion for tracks
  MonitorElement* Track_HQ_Chi2Red = nullptr; // chi2/dof distrubtion for tracks
  MonitorElement* Track_HQ_Chi2Red_NStubs = nullptr; // chi2/dof vs number of stubs
  MonitorElement* Track_HQ_Chi2Red_Eta = nullptr; // chi2/dof vs eta of track
  MonitorElement* Track_HQ_Eta_BarrelStubs = nullptr; // eta vs number of stubs in barrel
  MonitorElement* Track_HQ_Eta_ECStubs = nullptr; // eta vs number of stubs in end caps
  MonitorElement* Track_HQ_Chi2_Probability = nullptr; // chi2 probability

 private:
  edm::ParameterSet conf_;
  edm::EDGetTokenT<std::vector< TTTrack< Ref_Phase2TrackerDigi_ > > >  ttTrackToken_;

  unsigned int HQNStubs_;
  double HQChi2dof_;
  std::string topFolderName_;
};
#endif
