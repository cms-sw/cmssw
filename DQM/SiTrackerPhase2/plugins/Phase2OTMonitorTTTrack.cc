// Package:    SiOuterTracker
// Class:      SiOuterTracker
//
// Original Author:  Isis Marina Van Parijs
// Modified by: Emily MacDonald (emily.kaelyn.macdonald@cern.ch)

// system include files
#include <memory>
#include <numeric>
#include <string>
#include <vector>
#include <bitset>

// user include files
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "CommonTools/Statistics/interface/ChiSquaredProbability.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/L1TrackTrigger/interface/TTCluster.h"
#include "DataFormats/L1TrackTrigger/interface/TTStub.h"
#include "DataFormats/L1TrackTrigger/interface/TTTrack.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQM/SiTrackerPhase2/interface/TrackerPhase2DQMUtil.h"

class Phase2OTMonitorTTTrack : public DQMEDAnalyzer {
public:
  explicit Phase2OTMonitorTTTrack(const edm::ParameterSet &);
  ~Phase2OTMonitorTTTrack() override;
  void analyze(const edm::Event &, const edm::EventSetup &) override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);
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
  MonitorElement *Track_All_Chi2RZ = nullptr;            // chi2 r-phi distrubtion for tracks
  MonitorElement *Track_All_Chi2RPhi = nullptr;          // chi2 r-z distrubtion for tracks
  MonitorElement *Track_All_Chi2Red_NStubs = nullptr;    // chi2/dof vs number of stubs
  MonitorElement *Track_All_Chi2Red_Eta = nullptr;       // chi2/dof vs eta of track
  MonitorElement *Track_All_Eta_BarrelStubs = nullptr;   // eta vs number of stubs in barrel
  MonitorElement *Track_All_Eta_ECStubs = nullptr;       // eta vs number of stubs in end caps
  MonitorElement *Track_All_Chi2_Probability = nullptr;  // chi2 probability
  MonitorElement *Track_All_MVA1 = nullptr;              // MVA1 (prompt quality) distribution

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
  MonitorElement *Track_HQ_Chi2RZ = nullptr;            // chi2 r-z distrubtion for tracks
  MonitorElement *Track_HQ_Chi2RPhi = nullptr;          // chi2 r-phi distrubtion for tracks
  MonitorElement *Track_HQ_Chi2Red_NStubs = nullptr;    // chi2/dof vs number of stubs
  MonitorElement *Track_HQ_Chi2Red_Eta = nullptr;       // chi2/dof vs eta of track
  MonitorElement *Track_HQ_Eta_BarrelStubs = nullptr;   // eta vs number of stubs in barrel
  MonitorElement *Track_HQ_Eta_ECStubs = nullptr;       // eta vs number of stubs in end caps
  MonitorElement *Track_HQ_Chi2_Probability = nullptr;  // chi2 probability
  MonitorElement *Track_HQ_MVA1 = nullptr;              // MVA1 (prompt quality) distribution

private:
  edm::ParameterSet conf_;
  edm::EDGetTokenT<std::vector<TTTrack<Ref_Phase2TrackerDigi_>>> ttTrackToken_;

  unsigned int HQNStubs_;
  double HQChi2dof_;
  double HQBendChi2_;
  std::string topFolderName_;
};

// constructors and destructor
Phase2OTMonitorTTTrack::Phase2OTMonitorTTTrack(const edm::ParameterSet &iConfig) : conf_(iConfig) {
  topFolderName_ = conf_.getParameter<std::string>("TopFolderName");
  ttTrackToken_ =
      consumes<std::vector<TTTrack<Ref_Phase2TrackerDigi_>>>(conf_.getParameter<edm::InputTag>("TTTracksTag"));
  HQNStubs_ = conf_.getParameter<int>("HQNStubs");
  HQChi2dof_ = conf_.getParameter<double>("HQChi2dof");
  HQBendChi2_ = conf_.getParameter<double>("HQBendChi2");
}

Phase2OTMonitorTTTrack::~Phase2OTMonitorTTTrack() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

// ------------ method called for each event  ------------
void Phase2OTMonitorTTTrack::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  // L1 Primaries
  edm::Handle<std::vector<TTTrack<Ref_Phase2TrackerDigi_>>> TTTrackHandle;
  iEvent.getByToken(ttTrackToken_, TTTrackHandle);

  /// Track Trigger Tracks
  unsigned int numAllTracks = 0;
  unsigned int numHQTracks = 0;

  // Adding protection
  if (!TTTrackHandle.isValid())
    return;

  /// Loop over TTTracks
  unsigned int tkCnt = 0;
  for (const auto &iterTTTrack : *TTTrackHandle) {
    edm::Ptr<TTTrack<Ref_Phase2TrackerDigi_>> tempTrackPtr(TTTrackHandle, tkCnt++);  /// Make the pointer

    unsigned int nStubs = tempTrackPtr->getStubRefs().size();
    int nBarrelStubs = 0;
    int nECStubs = 0;

    float track_eta = tempTrackPtr->momentum().eta();
    float track_d0 = tempTrackPtr->d0();
    float track_bendchi2 = tempTrackPtr->chi2BendRed();
    float track_chi2 = tempTrackPtr->chi2();
    float track_chi2dof = tempTrackPtr->chi2Red();
    float track_chi2rz = tempTrackPtr->chi2Z();
    float track_chi2rphi = tempTrackPtr->chi2XY();
    float track_MVA1 = tempTrackPtr->trkMVA1();
    int nLayersMissed = 0;
    unsigned int hitPattern_ = (unsigned int)tempTrackPtr->hitPattern();

    int nbits = floor(log2(hitPattern_)) + 1;
    int lay_i = 0;
    bool seq = false;
    for (int i = 0; i < nbits; i++) {
      lay_i = ((1 << i) & hitPattern_) >> i;  //0 or 1 in ith bit (right to left)
      if (lay_i && !seq)
        seq = true;  //sequence starts when first 1 found
      if (!lay_i && seq)
        nLayersMissed++;
    }

    std::vector<edm::Ref<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_>>, TTStub<Ref_Phase2TrackerDigi_>>>
        theStubs = iterTTTrack.getStubRefs();
    for (const auto &istub : theStubs) {
      bool inBarrel = false;
      bool inEC = false;

      if (istub->getDetId().subdetId() == StripSubdetector::TOB)
        inBarrel = true;
      else if (istub->getDetId().subdetId() == StripSubdetector::TID)
        inEC = true;
      if (inBarrel)
        nBarrelStubs++;
      else if (inEC)
        nECStubs++;
    }  // end loop over stubs

    // HQ tracks: bendchi2<2.2 and chi2/dof<10
    if (nStubs >= HQNStubs_ && track_chi2dof <= HQChi2dof_ && track_bendchi2 <= HQBendChi2_) {
      numHQTracks++;

      Track_HQ_NStubs->Fill(nStubs);
      Track_HQ_NLayersMissed->Fill(nLayersMissed);
      Track_HQ_Eta_NStubs->Fill(track_eta, nStubs);
      Track_HQ_Pt->Fill(tempTrackPtr->momentum().perp());
      Track_HQ_Eta->Fill(track_eta);
      Track_HQ_Phi->Fill(tempTrackPtr->momentum().phi());
      Track_HQ_D0->Fill(track_d0);
      Track_HQ_VtxZ->Fill(tempTrackPtr->z0());
      Track_HQ_BendChi2->Fill(track_bendchi2);
      Track_HQ_Chi2->Fill(track_chi2);
      Track_HQ_Chi2RZ->Fill(track_chi2rz);
      Track_HQ_Chi2RPhi->Fill(track_chi2rphi);
      Track_HQ_Chi2Red->Fill(track_chi2dof);
      Track_HQ_Chi2Red_NStubs->Fill(nStubs, track_chi2dof);
      Track_HQ_Chi2Red_Eta->Fill(track_eta, track_chi2dof);
      Track_HQ_Eta_BarrelStubs->Fill(track_eta, nBarrelStubs);
      Track_HQ_Eta_ECStubs->Fill(track_eta, nECStubs);
      Track_HQ_Chi2_Probability->Fill(ChiSquaredProbability(track_chi2, nStubs));
      Track_HQ_MVA1->Fill(track_MVA1);
    }

    // All tracks (including HQ tracks)
    numAllTracks++;
    Track_All_NStubs->Fill(nStubs);
    Track_All_NLayersMissed->Fill(nLayersMissed);
    Track_All_Eta_NStubs->Fill(track_eta, nStubs);
    Track_All_Pt->Fill(tempTrackPtr->momentum().perp());
    Track_All_Eta->Fill(track_eta);
    Track_All_Phi->Fill(tempTrackPtr->momentum().phi());
    Track_All_D0->Fill(track_d0);
    Track_All_VtxZ->Fill(tempTrackPtr->z0());
    Track_All_BendChi2->Fill(track_bendchi2);
    Track_All_Chi2->Fill(track_chi2);
    Track_All_Chi2RZ->Fill(track_chi2rz);
    Track_All_Chi2RPhi->Fill(track_chi2rphi);
    Track_All_Chi2Red->Fill(track_chi2dof);
    Track_All_Chi2Red_NStubs->Fill(nStubs, track_chi2dof);
    Track_All_Chi2Red_Eta->Fill(track_eta, track_chi2dof);
    Track_All_Eta_BarrelStubs->Fill(track_eta, nBarrelStubs);
    Track_All_Eta_ECStubs->Fill(track_eta, nECStubs);
    Track_All_Chi2_Probability->Fill(ChiSquaredProbability(track_chi2, nStubs));
    Track_All_MVA1->Fill(track_MVA1);
  }  // End of loop over TTTracks

  Track_HQ_N->Fill(numHQTracks);
  Track_All_N->Fill(numAllTracks);
}  // end of method

// ------------ method called once each job just before starting event loop
// ------------
// Creating all histograms for DQM file output
void Phase2OTMonitorTTTrack::bookHistograms(DQMStore::IBooker &iBooker,
                                            edm::Run const &run,
                                            edm::EventSetup const &es) {
  using namespace phase2tkutil;

  iBooker.setCurrentFolder(topFolderName_ + "/Tracks/All");
  Track_All_N = book1DFromPSet(conf_.getParameter<edm::ParameterSet>("L1Track_All_N"), iBooker);
  Track_All_NStubs = book1DFromPSet(conf_.getParameter<edm::ParameterSet>("L1Track_All_NStubs"), iBooker);
  Track_All_NLayersMissed = book1DFromPSet(conf_.getParameter<edm::ParameterSet>("L1Track_All_NLayersMissed"), iBooker);
  Track_All_Eta_NStubs = book2DFromPSet(conf_.getParameter<edm::ParameterSet>("L1Track_All_Eta_NStubs"), iBooker);
  Track_All_Pt = book1DFromPSet(conf_.getParameter<edm::ParameterSet>("L1Track_All_Pt"), iBooker);
  Track_All_Phi = book1DFromPSet(conf_.getParameter<edm::ParameterSet>("L1Track_All_Phi"), iBooker);
  Track_All_D0 = book1DFromPSet(conf_.getParameter<edm::ParameterSet>("L1Track_All_D0"), iBooker);
  Track_All_Eta = book1DFromPSet(conf_.getParameter<edm::ParameterSet>("L1Track_All_Eta"), iBooker);
  Track_All_VtxZ = book1DFromPSet(conf_.getParameter<edm::ParameterSet>("L1Track_All_VtxZ"), iBooker);
  Track_All_Chi2 = book1DFromPSet(conf_.getParameter<edm::ParameterSet>("L1Track_All_Chi2"), iBooker);
  Track_All_Chi2RZ = book1DFromPSet(conf_.getParameter<edm::ParameterSet>("L1Track_All_Chi2RZ"), iBooker);
  Track_All_Chi2RPhi = book1DFromPSet(conf_.getParameter<edm::ParameterSet>("L1Track_All_Chi2RPhi"), iBooker);
  Track_All_BendChi2 = book1DFromPSet(conf_.getParameter<edm::ParameterSet>("L1Track_All_BendChi2"), iBooker);
  Track_All_Chi2Red = book1DFromPSet(conf_.getParameter<edm::ParameterSet>("L1Track_All_Chi2Red"), iBooker);
  Track_All_Chi2_Probability =
      book1DFromPSet(conf_.getParameter<edm::ParameterSet>("L1Track_All_Chi2_Probability"), iBooker);
  Track_All_MVA1 = book1DFromPSet(conf_.getParameter<edm::ParameterSet>("L1Track_All_MVA1"), iBooker);
  Track_All_Chi2Red_NStubs = book2DFromPSet(conf_.getParameter<edm::ParameterSet>("L1Track_All_Chi2Red_NStubs"), iBooker);
  Track_All_Chi2Red_Eta = book2DFromPSet(conf_.getParameter<edm::ParameterSet>("L1Track_All_Chi2Red_Eta"), iBooker);
  Track_All_Eta_BarrelStubs =
      book2DFromPSet(conf_.getParameter<edm::ParameterSet>("L1Track_All_Eta_BarrelStubs"), iBooker);
  Track_All_Eta_ECStubs = book2DFromPSet(conf_.getParameter<edm::ParameterSet>("L1Track_All_Eta_ECStubs"), iBooker);

  iBooker.setCurrentFolder(topFolderName_ + "/Tracks/HQ");
  Track_HQ_N = book1DFromPSet(conf_.getParameter<edm::ParameterSet>("L1Track_HQ_N"), iBooker);
  Track_HQ_NStubs = book1DFromPSet(conf_.getParameter<edm::ParameterSet>("L1Track_HQ_NStubs"), iBooker);
  Track_HQ_NLayersMissed = book1DFromPSet(conf_.getParameter<edm::ParameterSet>("L1Track_HQ_NLayersMissed"), iBooker);
  Track_HQ_Eta_NStubs = book2DFromPSet(conf_.getParameter<edm::ParameterSet>("L1Track_HQ_Eta_NStubs"), iBooker);
  Track_HQ_Pt = book1DFromPSet(conf_.getParameter<edm::ParameterSet>("L1Track_HQ_Pt"), iBooker);
  Track_HQ_Phi = book1DFromPSet(conf_.getParameter<edm::ParameterSet>("L1Track_HQ_Phi"), iBooker);
  Track_HQ_D0 = book1DFromPSet(conf_.getParameter<edm::ParameterSet>("L1Track_HQ_D0"), iBooker);
  Track_HQ_Eta = book1DFromPSet(conf_.getParameter<edm::ParameterSet>("L1Track_HQ_Eta"), iBooker);
  Track_HQ_VtxZ = book1DFromPSet(conf_.getParameter<edm::ParameterSet>("L1Track_HQ_VtxZ"), iBooker);
  Track_HQ_Chi2 = book1DFromPSet(conf_.getParameter<edm::ParameterSet>("L1Track_HQ_Chi2"), iBooker);
  Track_HQ_BendChi2 = book1DFromPSet(conf_.getParameter<edm::ParameterSet>("L1Track_HQ_BendChi2"), iBooker);
  Track_HQ_Chi2RZ = book1DFromPSet(conf_.getParameter<edm::ParameterSet>("L1Track_HQ_Chi2RZ"), iBooker);
  Track_HQ_Chi2RPhi = book1DFromPSet(conf_.getParameter<edm::ParameterSet>("L1Track_HQ_Chi2RPhi"), iBooker);
  Track_HQ_Chi2Red = book1DFromPSet(conf_.getParameter<edm::ParameterSet>("L1Track_HQ_Chi2Red"), iBooker);
  Track_HQ_Chi2_Probability =
      book1DFromPSet(conf_.getParameter<edm::ParameterSet>("L1Track_HQ_Chi2_Probability"), iBooker);
  Track_HQ_MVA1 = book1DFromPSet(conf_.getParameter<edm::ParameterSet>("L1Track_HQ_MVA1"), iBooker);
  Track_HQ_Chi2Red_NStubs = book2DFromPSet(conf_.getParameter<edm::ParameterSet>("L1Track_HQ_Chi2Red_NStubs"), iBooker);
  Track_HQ_Chi2Red_Eta = book2DFromPSet(conf_.getParameter<edm::ParameterSet>("L1Track_HQ_Chi2Red_Eta"), iBooker);
  Track_HQ_Eta_BarrelStubs = book2DFromPSet(conf_.getParameter<edm::ParameterSet>("L1Track_HQ_Eta_BarrelStubs"), iBooker);
  Track_HQ_Eta_ECStubs = book2DFromPSet(conf_.getParameter<edm::ParameterSet>("L1Track_HQ_Eta_ECStubs"), iBooker);
}

void Phase2OTMonitorTTTrack::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;

  // All tracks
  phase2tkutil::add1DDesc(desc, "L1Track_All_N", "L1Track_All_N", "# L1 Tracks", "# Events", 100, 0, 399);
  phase2tkutil::add1DDesc(
      desc, "L1Track_All_NStubs", "L1Track_All_NStubs", "# L1 Stubs per L1 Track", "# L1 Tracks", 8, 0, 8);
  phase2tkutil::add1DDesc(
      desc, "L1Track_All_NLayersMissed", "L1Track_All_NLayersMissed", "# Layers missed", "# L1 Tracks", 8, 0, 8);
  phase2tkutil::add2DDesc(
      desc, "L1Track_All_Eta_NStubs", "L1Track_All_Eta_NStubs", "#eta", "# L1 Stubs", 15, -3.0, 3.0, 5, 3, 8);
  phase2tkutil::add1DDesc(desc, "L1Track_All_Pt", "L1Track_All_Pt", "p_{T} [GeV]", "# L1 Tracks", 50, 0, 100);
  phase2tkutil::add1DDesc(desc, "L1Track_All_Phi", "L1Track_All_Phi", "#phi", "# L1 Tracks", 60, -3.5, 3.5);
  phase2tkutil::add1DDesc(desc, "L1Track_All_D0", "L1Track_All_D0", "Track D0", "# L1 Tracks", 101, -0.15, 0.15);
  phase2tkutil::add1DDesc(desc, "L1Track_All_Eta", "L1Track_All_Eta", "#eta", "# L1 Tracks", 45, -3.0, 3.0);
  phase2tkutil::add1DDesc(
      desc, "L1Track_All_VtxZ", "L1Track_All_VtxZ", "L1 Track vertex position z [cm]", "# L1 Tracks", 41, -20, 20);
  phase2tkutil::add1DDesc(desc, "L1Track_All_Chi2", "L1Track_All_Chi2", "L1 Track #chi^{2}", "# L1 Tracks", 100, 0, 50);
  phase2tkutil::add1DDesc(
      desc, "L1Track_All_Chi2RZ", "L1Track_All_Chi2RZ", "L1 Track #chi^{2} r-z", "# L1 Tracks", 100, 0, 50);
  phase2tkutil::add1DDesc(
      desc, "L1Track_All_Chi2RPhi", "L1Track_All_Chi2RPhi", "L1 Track #chi^{2}", "# L1 Tracks", 100, 0, 50);
  phase2tkutil::add1DDesc(
      desc, "L1Track_All_BendChi2", "L1Track_All_BendChi2", "L1 Track Bend #chi^{2}", "# L1 Tracks", 100, 0, 10);
  phase2tkutil::add1DDesc(
      desc, "L1Track_All_Chi2Red", "L1Track_All_Chi2Red", "L1 Track #chi^{2}/ndf", "# L1 Tracks", 100, 0, 10);
  phase2tkutil::add1DDesc(desc,
                          "L1Track_All_Chi2_Probability",
                          "L1Track_All_Chi2_Probability",
                          "#chi^{2} probability",
                          "# L1 Tracks",
                          100,
                          0,
                          1);
  phase2tkutil::add1DDesc(desc, "L1Track_All_MVA1", "L1Track_All_MVA1", "MVA1", "# L1 Tracks", 100, 0, 1);
  phase2tkutil::add2DDesc(desc,
                          "L1Track_All_Chi2Red_NStubs",
                          "L1Track_All_Chi2Red_NStubs",
                          "# L1 Stubs",
                          "L1 Track #chi^{2}/ndf",
                          5,
                          3,
                          8,
                          15,
                          0,
                          10);
  phase2tkutil::add2DDesc(
      desc, "L1Track_All_Chi2Red_Eta", "L1Track_All_Chi2Red_Eta", "#eta", "L1 Track #chi^{2}/ndf", 15, -3.0, 3.0, 15, 0, 10);
  phase2tkutil::add2DDesc(desc,
                          "L1Track_All_Eta_BarrelStubs",
                          "L1Track_All_Eta_BarrelStubs",
                          "#eta",
                          "# L1 Barrel Stubs",
                          15,
                          -3.0,
                          3.0,
                          5,
                          3,
                          8);
  phase2tkutil::add2DDesc(
      desc, "L1Track_All_Eta_ECStubs", "L1Track_All_Eta_ECStubs", "#eta", "# L1 EC Stubs", 15, -3.0, 3.0, 5, 3, 8);

  // HQ tracks
  phase2tkutil::add1DDesc(desc, "L1Track_HQ_N", "L1Track_HQ_N", "# L1 Tracks", "# Events", 100, 0, 399);
  phase2tkutil::add1DDesc(
      desc, "L1Track_HQ_NStubs", "L1Track_HQ_NStubs", "# L1 Stubs per L1 Track", "# L1 Tracks", 8, 0, 8);
  phase2tkutil::add1DDesc(
      desc, "L1Track_HQ_NLayersMissed", "L1Track_HQ_NLayersMissed", "# Layers missed", "# L1 Tracks", 8, 0, 8);
  phase2tkutil::add2DDesc(
      desc, "L1Track_HQ_Eta_NStubs", "L1Track_HQ_Eta_NStubs", "#eta", "# L1 Stubs", 15, -3.0, 3.0, 5, 3, 8);
  phase2tkutil::add1DDesc(desc, "L1Track_HQ_Pt", "L1Track_HQ_Pt", "p_{T} [GeV]", "# L1 Tracks", 50, 0, 100);
  phase2tkutil::add1DDesc(desc, "L1Track_HQ_Phi", "L1Track_HQ_Phi", "#phi", "# L1 Tracks", 60, -3.5, 3.5);
  phase2tkutil::add1DDesc(desc, "L1Track_HQ_D0", "L1Track_HQ_D0", "Track D0", "# L1 Tracks", 101, -0.15, 0.15);
  phase2tkutil::add1DDesc(desc, "L1Track_HQ_Eta", "L1Track_HQ_Eta", "#eta", "# L1 Tracks", 45, -3.0, 3.0);
  phase2tkutil::add1DDesc(
      desc, "L1Track_HQ_VtxZ", "L1Track_HQ_VtxZ", "L1 Track vertex position z [cm]", "# L1 Tracks", 41, -20, 20);
  phase2tkutil::add1DDesc(desc, "L1Track_HQ_Chi2", "L1Track_HQ_Chi2", "L1 Track #chi^{2}", "# L1 Tracks", 100, 0, 50);
  phase2tkutil::add1DDesc(
      desc, "L1Track_HQ_BendChi2", "L1Track_HQ_BendChi2", "L1 Track Bend #chi^{2}", "# L1 Tracks", 100, 0, 10);
  phase2tkutil::add1DDesc(
      desc, "L1Track_HQ_Chi2RZ", "L1Track_HQ_Chi2RZ", "L1 Track #chi^{2} r-z", "# L1 Tracks", 100, 0, 50);
  phase2tkutil::add1DDesc(
      desc, "L1Track_HQ_Chi2RPhi", "L1Track_HQ_Chi2RPhi", "L1 Track #chi^{2} r-phi", "# L1 Tracks", 100, 0, 50);
  phase2tkutil::add1DDesc(
      desc, "L1Track_HQ_Chi2Red", "L1Track_HQ_Chi2Red", "L1 Track #chi^{2}/ndf", "# L1 Tracks", 100, 0, 10);
  phase2tkutil::add1DDesc(
      desc, "L1Track_HQ_Chi2_Probability", "L1Track_HQ_Chi2_Probability", "#chi^{2} probability", "# L1 Tracks", 100, 0, 1);
  phase2tkutil::add1DDesc(desc, "L1Track_HQ_MVA1", "L1Track_HQ_MVA1", "MVA1", "# L1 Tracks", 100, 0, 1);
  phase2tkutil::add2DDesc(desc,
                          "L1Track_HQ_Chi2Red_NStubs",
                          "L1Track_HQ_Chi2Red_NStubs",
                          "# L1 Stubs",
                          "L1 Track #chi^{2}/ndf",
                          5,
                          3,
                          8,
                          15,
                          0,
                          10);
  phase2tkutil::add2DDesc(
      desc, "L1Track_HQ_Chi2Red_Eta", "L1Track_HQ_Chi2Red_Eta", "#eta", "L1 Track #chi^{2}/ndf", 15, -3.0, 3.0, 15, 0, 10);
  phase2tkutil::add2DDesc(
      desc, "L1Track_HQ_Eta_BarrelStubs", "L1Track_HQ_Eta_BarrelStubs", "#eta", "# L1 Barrel Stubs", 15, -3.0, 3.0, 5, 3, 8);
  phase2tkutil::add2DDesc(
      desc, "L1Track_HQ_Eta_ECStubs", "L1Track_HQ_Eta_ECStubs", "#eta", "# L1 EC Stubs", 15, -3.0, 3.0, 5, 3, 8);

  desc.add<std::string>("TopFolderName", "TrackerPhase2OTL1Track");
  desc.add<edm::InputTag>("TTTracksTag", edm::InputTag("l1tTTTracksFromTrackletEmulation", "Level1TTTracks"));
  desc.add<int>("HQNStubs", 4);
  desc.add<double>("HQChi2dof", 10.0);
  desc.add<double>("HQBendChi2", 2.2);
  descriptions.add("Phase2OTMonitorTTTrack", desc);
}
DEFINE_FWK_MODULE(Phase2OTMonitorTTTrack);
