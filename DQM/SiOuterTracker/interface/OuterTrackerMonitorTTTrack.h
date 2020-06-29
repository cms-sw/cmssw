#ifndef SiOuterTracker_OuterTrackerMonitorTTTrack_h
#define SiOuterTracker_OuterTrackerMonitorTTTrack_h

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include <memory>
#include <string>
#include <vector>

class OuterTrackerMonitorTTTrack : public DQMEDAnalyzer {
public:
  explicit OuterTrackerMonitorTTTrack(const edm::ParameterSet &);
  ~OuterTrackerMonitorTTTrack() override;
  void analyze(const edm::Event &, const edm::EventSetup &) override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;


  /// Low-quality TTTracks (All tracks)
  MonitorElement *Track_All_N = nullptr;                 // Number of tracks per event
  MonitorElement *Track_All_NStubs = nullptr;            // Number of stubs per track
  MonitorElement *Track_All_NLayersMissed = nullptr;     // Number of layers missed per track
  MonitorElement *Track_All_Eta_NStubs = nullptr;        // Number of stubs per track vs eta
  MonitorElement *Track_All_Pt = nullptr;                // pT distrubtion for tracks
  MonitorElement *Track_All_Eta = nullptr;               // eta distrubtion for tracks
  MonitorElement *Track_All_Phi = nullptr;               // phi distrubtion for tracks
  MonitorElement *Track_All_D0 = nullptr;                // d0 distrubtion for tracks
  MonitorElement *Track_All_VtxZ = nullptr;              // z0 distrubtion for tracks
  MonitorElement *Track_All_BendChi2 = nullptr;          // Bendchi2 distrubtion for tracks
  MonitorElement *Track_All_Chi2 = nullptr;              // chi2 distrubtion for tracks
  MonitorElement *Track_All_Chi2Red = nullptr;           // chi2/dof distrubtion for tracks
  MonitorElement *Track_All_Chi2RZ = nullptr;           // chi2 r-phi distrubtion for tracks
  MonitorElement *Track_All_Chi2RPhi = nullptr;           // chi2 r-z distrubtion for tracks
  MonitorElement *Track_All_Chi2Red_NStubs = nullptr;    // chi2/dof vs number of stubs
  MonitorElement *Track_All_Chi2Red_Eta = nullptr;       // chi2/dof vs eta of track
  MonitorElement *Track_All_Eta_BarrelStubs = nullptr;   // eta vs number of stubs in barrel
  MonitorElement *Track_All_Eta_ECStubs = nullptr;       // eta vs number of stubs in end caps
  MonitorElement *Track_All_Chi2_Probability = nullptr;  // chi2 probability

  /// High-quality TTTracks; different depending on prompt vs displaced tracks
  // Quality cuts: chi2/dof<10, bendchi2<2.2 (Prompt), default in config
  // Quality cuts: chi2/dof<40, bendchi2<2.4 (Extended/Displaced tracks)
  MonitorElement *Track_HQ_N = nullptr;                 // Number of tracks per event
  MonitorElement *Track_HQ_NStubs = nullptr;            // Number of stubs per track
  MonitorElement *Track_HQ_NLayersMissed = nullptr;     // Number of layers missed per track
  MonitorElement *Track_HQ_Eta_NStubs = nullptr;        // Number of stubs per track vs eta
  MonitorElement *Track_HQ_Pt = nullptr;                // pT distrubtion for tracks
  MonitorElement *Track_HQ_Eta = nullptr;               // eta distrubtion for tracks
  MonitorElement *Track_HQ_Phi = nullptr;               // phi distrubtion for tracks
  MonitorElement *Track_HQ_D0 = nullptr;                // d0 distrubtion for tracks
  MonitorElement *Track_HQ_VtxZ = nullptr;              // z0 distrubtion for tracks
  MonitorElement *Track_HQ_BendChi2 = nullptr;          // Bendchi2 distrubtion for tracks
  MonitorElement *Track_HQ_Chi2 = nullptr;              // chi2 distrubtion for tracks
  MonitorElement *Track_HQ_Chi2Red = nullptr;           // chi2/dof distrubtion for tracks
  MonitorElement *Track_HQ_Chi2RZ = nullptr;           // chi2 r-z distrubtion for tracks
  MonitorElement *Track_HQ_Chi2RPhi = nullptr;           // chi2 r-phi distrubtion for tracks
  MonitorElement *Track_HQ_Chi2Red_NStubs = nullptr;    // chi2/dof vs number of stubs
  MonitorElement *Track_HQ_Chi2Red_Eta = nullptr;       // chi2/dof vs eta of track
  MonitorElement *Track_HQ_Eta_BarrelStubs = nullptr;   // eta vs number of stubs in barrel
  MonitorElement *Track_HQ_Eta_ECStubs = nullptr;       // eta vs number of stubs in end caps
  MonitorElement *Track_HQ_Chi2_Probability = nullptr;  // chi2 probability

private:
  edm::ParameterSet conf_;
  edm::EDGetTokenT<std::vector<TTTrack<Ref_Phase2TrackerDigi_>>> ttTrackToken_;

  unsigned int HQNStubs_;
  double HQChi2dof_;
  double HQBendChi2_;
  std::string topFolderName_;
};
#endif
