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
  MonitorElement *Track_All_Chi2RZ = nullptr;            // chi2 r-phi distrubtion for tracks
  MonitorElement *Track_All_Chi2RPhi = nullptr;          // chi2 r-z distrubtion for tracks
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
  MonitorElement *Track_HQ_Chi2RZ = nullptr;            // chi2 r-z distrubtion for tracks
  MonitorElement *Track_HQ_Chi2RPhi = nullptr;          // chi2 r-phi distrubtion for tracks
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

// constructors and destructor
OuterTrackerMonitorTTTrack::OuterTrackerMonitorTTTrack(const edm::ParameterSet &iConfig) : conf_(iConfig) {
  topFolderName_ = conf_.getParameter<std::string>("TopFolderName");
  ttTrackToken_ =
      consumes<std::vector<TTTrack<Ref_Phase2TrackerDigi_>>>(conf_.getParameter<edm::InputTag>("TTTracksTag"));
  HQNStubs_ = conf_.getParameter<int>("HQNStubs");
  HQChi2dof_ = conf_.getParameter<double>("HQChi2dof");
  HQBendChi2_ = conf_.getParameter<double>("HQBendChi2");
}

OuterTrackerMonitorTTTrack::~OuterTrackerMonitorTTTrack() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

// ------------ method called for each event  ------------
void OuterTrackerMonitorTTTrack::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
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
    float track_bendchi2 = tempTrackPtr->stubPtConsistency();
    float track_chi2 = tempTrackPtr->chi2();
    float track_chi2dof = tempTrackPtr->chi2Red();
    float track_chi2rz = tempTrackPtr->chi2Z();
    float track_chi2rphi = tempTrackPtr->chi2XY();
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

  }  // End of loop over TTTracks

  Track_HQ_N->Fill(numHQTracks);
  Track_All_N->Fill(numAllTracks);
}  // end of method

// ------------ method called once each job just before starting event loop
// ------------
// Creating all histograms for DQM file output
void OuterTrackerMonitorTTTrack::bookHistograms(DQMStore::IBooker &iBooker,
                                                edm::Run const &run,
                                                edm::EventSetup const &es) {
  std::string HistoName;

  /// Low-quality tracks (All tracks, including HQ tracks)
  iBooker.setCurrentFolder(topFolderName_ + "/Tracks/All");
  // Nb of L1Tracks
  HistoName = "Track_All_N";
  edm::ParameterSet psTrack_N = conf_.getParameter<edm::ParameterSet>("TH1_NTracks");
  Track_All_N = iBooker.book1D(HistoName,
                               HistoName,
                               psTrack_N.getParameter<int32_t>("Nbinsx"),
                               psTrack_N.getParameter<double>("xmin"),
                               psTrack_N.getParameter<double>("xmax"));
  Track_All_N->setAxisTitle("# L1 Tracks", 1);
  Track_All_N->setAxisTitle("# Events", 2);

  // Number of stubs
  edm::ParameterSet psTrack_NStubs = conf_.getParameter<edm::ParameterSet>("TH1_NStubs");
  HistoName = "Track_All_NStubs";
  Track_All_NStubs = iBooker.book1D(HistoName,
                                    HistoName,
                                    psTrack_NStubs.getParameter<int32_t>("Nbinsx"),
                                    psTrack_NStubs.getParameter<double>("xmin"),
                                    psTrack_NStubs.getParameter<double>("xmax"));
  Track_All_NStubs->setAxisTitle("# L1 Stubs per L1 Track", 1);
  Track_All_NStubs->setAxisTitle("# L1 Tracks", 2);

  // Number of layers missed
  HistoName = "Track_All_NLayersMissed";
  Track_All_NLayersMissed = iBooker.book1D(HistoName,
                                           HistoName,
                                           psTrack_NStubs.getParameter<int32_t>("Nbinsx"),
                                           psTrack_NStubs.getParameter<double>("xmin"),
                                           psTrack_NStubs.getParameter<double>("xmax"));
  Track_All_NLayersMissed->setAxisTitle("# Layers missed", 1);
  Track_All_NLayersMissed->setAxisTitle("# L1 Tracks", 2);

  edm::ParameterSet psTrack_Eta_NStubs = conf_.getParameter<edm::ParameterSet>("TH2_Track_Eta_NStubs");
  HistoName = "Track_All_Eta_NStubs";
  Track_All_Eta_NStubs = iBooker.book2D(HistoName,
                                        HistoName,
                                        psTrack_Eta_NStubs.getParameter<int32_t>("Nbinsx"),
                                        psTrack_Eta_NStubs.getParameter<double>("xmin"),
                                        psTrack_Eta_NStubs.getParameter<double>("xmax"),
                                        psTrack_Eta_NStubs.getParameter<int32_t>("Nbinsy"),
                                        psTrack_Eta_NStubs.getParameter<double>("ymin"),
                                        psTrack_Eta_NStubs.getParameter<double>("ymax"));
  Track_All_Eta_NStubs->setAxisTitle("#eta", 1);
  Track_All_Eta_NStubs->setAxisTitle("# L1 Stubs", 2);

  // Pt of the tracks
  edm::ParameterSet psTrack_Pt = conf_.getParameter<edm::ParameterSet>("TH1_Track_Pt");
  HistoName = "Track_All_Pt";
  Track_All_Pt = iBooker.book1D(HistoName,
                                HistoName,
                                psTrack_Pt.getParameter<int32_t>("Nbinsx"),
                                psTrack_Pt.getParameter<double>("xmin"),
                                psTrack_Pt.getParameter<double>("xmax"));
  Track_All_Pt->setAxisTitle("p_{T} [GeV]", 1);
  Track_All_Pt->setAxisTitle("# L1 Tracks", 2);

  // Phi
  edm::ParameterSet psTrack_Phi = conf_.getParameter<edm::ParameterSet>("TH1_Track_Phi");
  HistoName = "Track_All_Phi";
  Track_All_Phi = iBooker.book1D(HistoName,
                                 HistoName,
                                 psTrack_Phi.getParameter<int32_t>("Nbinsx"),
                                 psTrack_Phi.getParameter<double>("xmin"),
                                 psTrack_Phi.getParameter<double>("xmax"));
  Track_All_Phi->setAxisTitle("#phi", 1);
  Track_All_Phi->setAxisTitle("# L1 Tracks", 2);

  // D0
  edm::ParameterSet psTrack_D0 = conf_.getParameter<edm::ParameterSet>("TH1_Track_D0");
  HistoName = "Track_All_D0";
  Track_All_D0 = iBooker.book1D(HistoName,
                                HistoName,
                                psTrack_D0.getParameter<int32_t>("Nbinsx"),
                                psTrack_D0.getParameter<double>("xmin"),
                                psTrack_D0.getParameter<double>("xmax"));
  Track_All_D0->setAxisTitle("Track D0", 1);
  Track_All_D0->setAxisTitle("# L1 Tracks", 2);

  // Eta
  edm::ParameterSet psTrack_Eta = conf_.getParameter<edm::ParameterSet>("TH1_Track_Eta");
  HistoName = "Track_All_Eta";
  Track_All_Eta = iBooker.book1D(HistoName,
                                 HistoName,
                                 psTrack_Eta.getParameter<int32_t>("Nbinsx"),
                                 psTrack_Eta.getParameter<double>("xmin"),
                                 psTrack_Eta.getParameter<double>("xmax"));
  Track_All_Eta->setAxisTitle("#eta", 1);
  Track_All_Eta->setAxisTitle("# L1 Tracks", 2);

  // VtxZ
  edm::ParameterSet psTrack_VtxZ = conf_.getParameter<edm::ParameterSet>("TH1_Track_VtxZ");
  HistoName = "Track_All_VtxZ";
  Track_All_VtxZ = iBooker.book1D(HistoName,
                                  HistoName,
                                  psTrack_VtxZ.getParameter<int32_t>("Nbinsx"),
                                  psTrack_VtxZ.getParameter<double>("xmin"),
                                  psTrack_VtxZ.getParameter<double>("xmax"));
  Track_All_VtxZ->setAxisTitle("L1 Track vertex position z [cm]", 1);
  Track_All_VtxZ->setAxisTitle("# L1 Tracks", 2);

  // chi2
  edm::ParameterSet psTrack_Chi2 = conf_.getParameter<edm::ParameterSet>("TH1_Track_Chi2");
  HistoName = "Track_All_Chi2";
  Track_All_Chi2 = iBooker.book1D(HistoName,
                                  HistoName,
                                  psTrack_Chi2.getParameter<int32_t>("Nbinsx"),
                                  psTrack_Chi2.getParameter<double>("xmin"),
                                  psTrack_Chi2.getParameter<double>("xmax"));
  Track_All_Chi2->setAxisTitle("L1 Track #chi^{2}", 1);
  Track_All_Chi2->setAxisTitle("# L1 Tracks", 2);

  // chi2 r-z
  HistoName = "Track_All_Chi2RZ";
  Track_All_Chi2RZ = iBooker.book1D(HistoName,
                                    HistoName,
                                    psTrack_Chi2.getParameter<int32_t>("Nbinsx"),
                                    psTrack_Chi2.getParameter<double>("xmin"),
                                    psTrack_Chi2.getParameter<double>("xmax"));
  Track_All_Chi2RZ->setAxisTitle("L1 Track #chi^{2} r-z", 1);
  Track_All_Chi2RZ->setAxisTitle("# L1 Tracks", 2);

  // chi2 r-phi
  HistoName = "Track_All_Chi2RPhi";
  Track_All_Chi2RPhi = iBooker.book1D(HistoName,
                                      HistoName,
                                      psTrack_Chi2.getParameter<int32_t>("Nbinsx"),
                                      psTrack_Chi2.getParameter<double>("xmin"),
                                      psTrack_Chi2.getParameter<double>("xmax"));
  Track_All_Chi2RPhi->setAxisTitle("L1 Track #chi^{2}", 1);
  Track_All_Chi2RPhi->setAxisTitle("# L1 Tracks", 2);

  // Bendchi2
  edm::ParameterSet psTrack_Chi2R = conf_.getParameter<edm::ParameterSet>("TH1_Track_Chi2R");
  HistoName = "Track_All_BendChi2";
  Track_All_BendChi2 = iBooker.book1D(HistoName,
                                      HistoName,
                                      psTrack_Chi2R.getParameter<int32_t>("Nbinsx"),
                                      psTrack_Chi2R.getParameter<double>("xmin"),
                                      psTrack_Chi2R.getParameter<double>("xmax"));
  Track_All_BendChi2->setAxisTitle("L1 Track Bend #chi^{2}", 1);
  Track_All_BendChi2->setAxisTitle("# L1 Tracks", 2);

  // chi2Red
  edm::ParameterSet psTrack_Chi2Red = conf_.getParameter<edm::ParameterSet>("TH1_Track_Chi2R");
  HistoName = "Track_All_Chi2Red";
  Track_All_Chi2Red = iBooker.book1D(HistoName,
                                     HistoName,
                                     psTrack_Chi2R.getParameter<int32_t>("Nbinsx"),
                                     psTrack_Chi2R.getParameter<double>("xmin"),
                                     psTrack_Chi2R.getParameter<double>("xmax"));
  Track_All_Chi2Red->setAxisTitle("L1 Track #chi^{2}/ndf", 1);
  Track_All_Chi2Red->setAxisTitle("# L1 Tracks", 2);

  // Chi2 prob
  edm::ParameterSet psTrack_Chi2_Probability = conf_.getParameter<edm::ParameterSet>("TH1_Track_Chi2_Probability");
  HistoName = "Track_All_Chi2_Probability";
  Track_All_Chi2_Probability = iBooker.book1D(HistoName,
                                              HistoName,
                                              psTrack_Chi2_Probability.getParameter<int32_t>("Nbinsx"),
                                              psTrack_Chi2_Probability.getParameter<double>("xmin"),
                                              psTrack_Chi2_Probability.getParameter<double>("xmax"));
  Track_All_Chi2_Probability->setAxisTitle("#chi^{2} probability", 1);
  Track_All_Chi2_Probability->setAxisTitle("# L1 Tracks", 2);

  // Reduced chi2 vs #stubs
  edm::ParameterSet psTrack_Chi2R_NStubs = conf_.getParameter<edm::ParameterSet>("TH2_Track_Chi2R_NStubs");
  HistoName = "Track_All_Chi2Red_NStubs";
  Track_All_Chi2Red_NStubs = iBooker.book2D(HistoName,
                                            HistoName,
                                            psTrack_Chi2R_NStubs.getParameter<int32_t>("Nbinsx"),
                                            psTrack_Chi2R_NStubs.getParameter<double>("xmin"),
                                            psTrack_Chi2R_NStubs.getParameter<double>("xmax"),
                                            psTrack_Chi2R_NStubs.getParameter<int32_t>("Nbinsy"),
                                            psTrack_Chi2R_NStubs.getParameter<double>("ymin"),
                                            psTrack_Chi2R_NStubs.getParameter<double>("ymax"));
  Track_All_Chi2Red_NStubs->setAxisTitle("# L1 Stubs", 1);
  Track_All_Chi2Red_NStubs->setAxisTitle("L1 Track #chi^{2}/ndf", 2);

  // chi2/dof vs eta
  edm::ParameterSet psTrack_Chi2R_Eta = conf_.getParameter<edm::ParameterSet>("TH2_Track_Chi2R_Eta");
  HistoName = "Track_All_Chi2Red_Eta";
  Track_All_Chi2Red_Eta = iBooker.book2D(HistoName,
                                         HistoName,
                                         psTrack_Chi2R_Eta.getParameter<int32_t>("Nbinsx"),
                                         psTrack_Chi2R_Eta.getParameter<double>("xmin"),
                                         psTrack_Chi2R_Eta.getParameter<double>("xmax"),
                                         psTrack_Chi2R_Eta.getParameter<int32_t>("Nbinsy"),
                                         psTrack_Chi2R_Eta.getParameter<double>("ymin"),
                                         psTrack_Chi2R_Eta.getParameter<double>("ymax"));
  Track_All_Chi2Red_Eta->setAxisTitle("#eta", 1);
  Track_All_Chi2Red_Eta->setAxisTitle("L1 Track #chi^{2}/ndf", 2);

  // Eta vs #stubs in barrel
  HistoName = "Track_All_Eta_BarrelStubs";
  Track_All_Eta_BarrelStubs = iBooker.book2D(HistoName,
                                             HistoName,
                                             psTrack_Eta_NStubs.getParameter<int32_t>("Nbinsx"),
                                             psTrack_Eta_NStubs.getParameter<double>("xmin"),
                                             psTrack_Eta_NStubs.getParameter<double>("xmax"),
                                             psTrack_Eta_NStubs.getParameter<int32_t>("Nbinsy"),
                                             psTrack_Eta_NStubs.getParameter<double>("ymin"),
                                             psTrack_Eta_NStubs.getParameter<double>("ymax"));
  Track_All_Eta_BarrelStubs->setAxisTitle("#eta", 1);
  Track_All_Eta_BarrelStubs->setAxisTitle("# L1 Barrel Stubs", 2);

  // Eta vs #stubs in EC
  HistoName = "Track_LQ_Eta_ECStubs";
  Track_All_Eta_ECStubs = iBooker.book2D(HistoName,
                                         HistoName,
                                         psTrack_Eta_NStubs.getParameter<int32_t>("Nbinsx"),
                                         psTrack_Eta_NStubs.getParameter<double>("xmin"),
                                         psTrack_Eta_NStubs.getParameter<double>("xmax"),
                                         psTrack_Eta_NStubs.getParameter<int32_t>("Nbinsy"),
                                         psTrack_Eta_NStubs.getParameter<double>("ymin"),
                                         psTrack_Eta_NStubs.getParameter<double>("ymax"));
  Track_All_Eta_ECStubs->setAxisTitle("#eta", 1);
  Track_All_Eta_ECStubs->setAxisTitle("# L1 EC Stubs", 2);

  /// High-quality tracks (Bendchi2 < 2.2 and chi2/dof < 10)
  iBooker.setCurrentFolder(topFolderName_ + "/Tracks/HQ");
  // Nb of L1Tracks
  HistoName = "Track_HQ_N";
  Track_HQ_N = iBooker.book1D(HistoName,
                              HistoName,
                              psTrack_N.getParameter<int32_t>("Nbinsx"),
                              psTrack_N.getParameter<double>("xmin"),
                              psTrack_N.getParameter<double>("xmax"));
  Track_HQ_N->setAxisTitle("# L1 Tracks", 1);
  Track_HQ_N->setAxisTitle("# Events", 2);

  // Number of stubs
  HistoName = "Track_HQ_NStubs";
  Track_HQ_NStubs = iBooker.book1D(HistoName,
                                   HistoName,
                                   psTrack_NStubs.getParameter<int32_t>("Nbinsx"),
                                   psTrack_NStubs.getParameter<double>("xmin"),
                                   psTrack_NStubs.getParameter<double>("xmax"));
  Track_HQ_NStubs->setAxisTitle("# L1 Stubs per L1 Track", 1);
  Track_HQ_NStubs->setAxisTitle("# L1 Tracks", 2);

  // Number of layers missed
  HistoName = "Track_HQ_NLayersMissed";
  Track_HQ_NLayersMissed = iBooker.book1D(HistoName,
                                          HistoName,
                                          psTrack_NStubs.getParameter<int32_t>("Nbinsx"),
                                          psTrack_NStubs.getParameter<double>("xmin"),
                                          psTrack_NStubs.getParameter<double>("xmax"));
  Track_HQ_NLayersMissed->setAxisTitle("# Layers missed", 1);
  Track_HQ_NLayersMissed->setAxisTitle("# L1 Tracks", 2);

  // Track eta vs #stubs
  HistoName = "Track_HQ_Eta_NStubs";
  Track_HQ_Eta_NStubs = iBooker.book2D(HistoName,
                                       HistoName,
                                       psTrack_Eta_NStubs.getParameter<int32_t>("Nbinsx"),
                                       psTrack_Eta_NStubs.getParameter<double>("xmin"),
                                       psTrack_Eta_NStubs.getParameter<double>("xmax"),
                                       psTrack_Eta_NStubs.getParameter<int32_t>("Nbinsy"),
                                       psTrack_Eta_NStubs.getParameter<double>("ymin"),
                                       psTrack_Eta_NStubs.getParameter<double>("ymax"));
  Track_HQ_Eta_NStubs->setAxisTitle("#eta", 1);
  Track_HQ_Eta_NStubs->setAxisTitle("# L1 Stubs", 2);

  // Pt of the tracks
  HistoName = "Track_HQ_Pt";
  Track_HQ_Pt = iBooker.book1D(HistoName,
                               HistoName,
                               psTrack_Pt.getParameter<int32_t>("Nbinsx"),
                               psTrack_Pt.getParameter<double>("xmin"),
                               psTrack_Pt.getParameter<double>("xmax"));
  Track_HQ_Pt->setAxisTitle("p_{T} [GeV]", 1);
  Track_HQ_Pt->setAxisTitle("# L1 Tracks", 2);

  // Phi
  HistoName = "Track_HQ_Phi";
  Track_HQ_Phi = iBooker.book1D(HistoName,
                                HistoName,
                                psTrack_Phi.getParameter<int32_t>("Nbinsx"),
                                psTrack_Phi.getParameter<double>("xmin"),
                                psTrack_Phi.getParameter<double>("xmax"));
  Track_HQ_Phi->setAxisTitle("#phi", 1);
  Track_HQ_Phi->setAxisTitle("# L1 Tracks", 2);

  // D0
  HistoName = "Track_HQ_D0";
  Track_HQ_D0 = iBooker.book1D(HistoName,
                               HistoName,
                               psTrack_D0.getParameter<int32_t>("Nbinsx"),
                               psTrack_D0.getParameter<double>("xmin"),
                               psTrack_D0.getParameter<double>("xmax"));
  Track_HQ_D0->setAxisTitle("Track D0", 1);
  Track_HQ_D0->setAxisTitle("# L1 Tracks", 2);

  // Eta
  HistoName = "Track_HQ_Eta";
  Track_HQ_Eta = iBooker.book1D(HistoName,
                                HistoName,
                                psTrack_Eta.getParameter<int32_t>("Nbinsx"),
                                psTrack_Eta.getParameter<double>("xmin"),
                                psTrack_Eta.getParameter<double>("xmax"));
  Track_HQ_Eta->setAxisTitle("#eta", 1);
  Track_HQ_Eta->setAxisTitle("# L1 Tracks", 2);

  // VtxZ
  HistoName = "Track_HQ_VtxZ";
  Track_HQ_VtxZ = iBooker.book1D(HistoName,
                                 HistoName,
                                 psTrack_VtxZ.getParameter<int32_t>("Nbinsx"),
                                 psTrack_VtxZ.getParameter<double>("xmin"),
                                 psTrack_VtxZ.getParameter<double>("xmax"));
  Track_HQ_VtxZ->setAxisTitle("L1 Track vertex position z [cm]", 1);
  Track_HQ_VtxZ->setAxisTitle("# L1 Tracks", 2);

  // chi2
  HistoName = "Track_HQ_Chi2";
  Track_HQ_Chi2 = iBooker.book1D(HistoName,
                                 HistoName,
                                 psTrack_Chi2.getParameter<int32_t>("Nbinsx"),
                                 psTrack_Chi2.getParameter<double>("xmin"),
                                 psTrack_Chi2.getParameter<double>("xmax"));
  Track_HQ_Chi2->setAxisTitle("L1 Track #chi^{2}", 1);
  Track_HQ_Chi2->setAxisTitle("# L1 Tracks", 2);

  // Bendchi2
  HistoName = "Track_HQ_BendChi2";
  Track_HQ_BendChi2 = iBooker.book1D(HistoName,
                                     HistoName,
                                     psTrack_Chi2R.getParameter<int32_t>("Nbinsx"),
                                     psTrack_Chi2R.getParameter<double>("xmin"),
                                     psTrack_Chi2R.getParameter<double>("xmax"));
  Track_HQ_BendChi2->setAxisTitle("L1 Track Bend #chi^{2}", 1);
  Track_HQ_BendChi2->setAxisTitle("# L1 Tracks", 2);

  // chi2 r-z
  HistoName = "Track_HQ_Chi2RZ";
  Track_HQ_Chi2RZ = iBooker.book1D(HistoName,
                                   HistoName,
                                   psTrack_Chi2.getParameter<int32_t>("Nbinsx"),
                                   psTrack_Chi2.getParameter<double>("xmin"),
                                   psTrack_Chi2.getParameter<double>("xmax"));
  Track_HQ_Chi2RZ->setAxisTitle("L1 Track #chi^{2} r-z", 1);
  Track_HQ_Chi2RZ->setAxisTitle("# L1 Tracks", 2);

  HistoName = "Track_HQ_Chi2RPhi";
  Track_HQ_Chi2RPhi = iBooker.book1D(HistoName,
                                     HistoName,
                                     psTrack_Chi2.getParameter<int32_t>("Nbinsx"),
                                     psTrack_Chi2.getParameter<double>("xmin"),
                                     psTrack_Chi2.getParameter<double>("xmax"));
  Track_HQ_Chi2RPhi->setAxisTitle("L1 Track #chi^{2} r-phi", 1);
  Track_HQ_Chi2RPhi->setAxisTitle("# L1 Tracks", 2);

  // chi2Red
  HistoName = "Track_HQ_Chi2Red";
  Track_HQ_Chi2Red = iBooker.book1D(HistoName,
                                    HistoName,
                                    psTrack_Chi2R.getParameter<int32_t>("Nbinsx"),
                                    psTrack_Chi2R.getParameter<double>("xmin"),
                                    psTrack_Chi2R.getParameter<double>("xmax"));
  Track_HQ_Chi2Red->setAxisTitle("L1 Track #chi^{2}/ndf", 1);
  Track_HQ_Chi2Red->setAxisTitle("# L1 Tracks", 2);

  // Chi2 prob
  HistoName = "Track_HQ_Chi2_Probability";
  Track_HQ_Chi2_Probability = iBooker.book1D(HistoName,
                                             HistoName,
                                             psTrack_Chi2_Probability.getParameter<int32_t>("Nbinsx"),
                                             psTrack_Chi2_Probability.getParameter<double>("xmin"),
                                             psTrack_Chi2_Probability.getParameter<double>("xmax"));
  Track_HQ_Chi2_Probability->setAxisTitle("#chi^{2} probability", 1);
  Track_HQ_Chi2_Probability->setAxisTitle("# L1 Tracks", 2);

  // Reduced chi2 vs #stubs
  HistoName = "Track_HQ_Chi2Red_NStubs";
  Track_HQ_Chi2Red_NStubs = iBooker.book2D(HistoName,
                                           HistoName,
                                           psTrack_Chi2R_NStubs.getParameter<int32_t>("Nbinsx"),
                                           psTrack_Chi2R_NStubs.getParameter<double>("xmin"),
                                           psTrack_Chi2R_NStubs.getParameter<double>("xmax"),
                                           psTrack_Chi2R_NStubs.getParameter<int32_t>("Nbinsy"),
                                           psTrack_Chi2R_NStubs.getParameter<double>("ymin"),
                                           psTrack_Chi2R_NStubs.getParameter<double>("ymax"));
  Track_HQ_Chi2Red_NStubs->setAxisTitle("# L1 Stubs", 1);
  Track_HQ_Chi2Red_NStubs->setAxisTitle("L1 Track #chi^{2}/ndf", 2);

  // chi2/dof vs eta
  HistoName = "Track_HQ_Chi2Red_Eta";
  Track_HQ_Chi2Red_Eta = iBooker.book2D(HistoName,
                                        HistoName,
                                        psTrack_Chi2R_Eta.getParameter<int32_t>("Nbinsx"),
                                        psTrack_Chi2R_Eta.getParameter<double>("xmin"),
                                        psTrack_Chi2R_Eta.getParameter<double>("xmax"),
                                        psTrack_Chi2R_Eta.getParameter<int32_t>("Nbinsy"),
                                        psTrack_Chi2R_Eta.getParameter<double>("ymin"),
                                        psTrack_Chi2R_Eta.getParameter<double>("ymax"));
  Track_HQ_Chi2Red_Eta->setAxisTitle("#eta", 1);
  Track_HQ_Chi2Red_Eta->setAxisTitle("L1 Track #chi^{2}/ndf", 2);

  // eta vs #stubs in barrel
  HistoName = "Track_HQ_Eta_BarrelStubs";
  Track_HQ_Eta_BarrelStubs = iBooker.book2D(HistoName,
                                            HistoName,
                                            psTrack_Eta_NStubs.getParameter<int32_t>("Nbinsx"),
                                            psTrack_Eta_NStubs.getParameter<double>("xmin"),
                                            psTrack_Eta_NStubs.getParameter<double>("xmax"),
                                            psTrack_Eta_NStubs.getParameter<int32_t>("Nbinsy"),
                                            psTrack_Eta_NStubs.getParameter<double>("ymin"),
                                            psTrack_Eta_NStubs.getParameter<double>("ymax"));
  Track_HQ_Eta_BarrelStubs->setAxisTitle("#eta", 1);
  Track_HQ_Eta_BarrelStubs->setAxisTitle("# L1 Barrel Stubs", 2);

  // eta vs #stubs in EC
  HistoName = "Track_HQ_Eta_ECStubs";
  Track_HQ_Eta_ECStubs = iBooker.book2D(HistoName,
                                        HistoName,
                                        psTrack_Eta_NStubs.getParameter<int32_t>("Nbinsx"),
                                        psTrack_Eta_NStubs.getParameter<double>("xmin"),
                                        psTrack_Eta_NStubs.getParameter<double>("xmax"),
                                        psTrack_Eta_NStubs.getParameter<int32_t>("Nbinsy"),
                                        psTrack_Eta_NStubs.getParameter<double>("ymin"),
                                        psTrack_Eta_NStubs.getParameter<double>("ymax"));
  Track_HQ_Eta_ECStubs->setAxisTitle("#eta", 1);
  Track_HQ_Eta_ECStubs->setAxisTitle("# L1 EC Stubs", 2);

}  // end of method

DEFINE_FWK_MODULE(OuterTrackerMonitorTTTrack);
