// -*- C++ -*-
//
// Package:    RecoPPS/RPixEfficiencyTools
// Class:      NoiseAnalyzer
//
/**\class ShowerAnalyzer_2018 ShowerAnalyzer_2018.cc
 RecoPPS/RPixEfficiencyTools/plugins/ShowerAnalyzer_2018.cc

 Description: [one line class summary]

 Implementation:
                 [Notes on implementation]
*/
//
// Original Author:  Andrea Bellora
//         Created:  Wed, 07 Jul 2019 09:55:05 GMT
//
//

#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <exception>
#include <fstream>
#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/CTPPSDetId/interface/CTPPSPixelDetId.h"

#include "DataFormats/CTPPSReco/interface/CTPPSLocalTrackLite.h"
#include "DataFormats/CTPPSReco/interface/CTPPSPixelCluster.h"
#include "DataFormats/CTPPSReco/interface/CTPPSPixelLocalTrack.h"
#include "DataFormats/CTPPSReco/interface/CTPPSPixelRecHit.h"

#include "DataFormats/CTPPSReco/interface/CTPPSPixelLocalTrackRecoInfo.h"
#include "DataFormats/ProtonReco/interface/ForwardProton.h"
#include "DataFormats/ProtonReco/interface/ForwardProtonFwd.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <TEfficiency.h>
#include <TF1.h>
#include <TFile.h>
#include <TGraphErrors.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TMath.h>
#include <TObjArray.h>

using namespace std;

class ShowerAnalyzer_2018
    : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit ShowerAnalyzer_2018(const edm::ParameterSet &);
  ~ShowerAnalyzer_2018();
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  virtual void beginJob() override;
  virtual void analyze(const edm::Event &, const edm::EventSetup &) override;
  virtual void endJob() override;
  bool Cut(CTPPSLocalTrackLite track);
  bool CutWithNoNumberOfPoints(CTPPSLocalTrackLite track);

  bool debug_ = false;

  // Data to get
  edm::EDGetTokenT<reco::ForwardProtonCollection> protonsToken_;
  edm::EDGetTokenT<edm::DetSetVector<CTPPSPixelLocalTrack>>
      pixelLocalTrackToken_;
  edm::EDGetTokenT<edm::DetSetVector<CTPPSPixelRecHit>> pixelRecHitToken_;
  edm::EDGetTokenT<edm::DetSetVector<CTPPSPixelCluster>> pixelClusterToken_;

  // Parameter set
  string outputFileName_;
  double maxChi2Prob_;
  int minNumberOfPlanesForTrack_;
  int maxNumberOfPlanesForTrack_ = 6;
  int minTracksPerEvent;
  int maxTracksPerEvent;

  // Configs
  vector<uint32_t> listOfArms_ = {0, 1};
  vector<uint32_t> listOfStations_ = {0, 2};
  vector<uint32_t> listOfPlanes_ = {0, 1, 2, 3, 4, 5};

  vector<CTPPSPixelDetId> planeIdVector_;
  vector<CTPPSPixelDetId> romanPotIdVector_;

  int binGroupingX = 1;
  int binGroupingY = 1;

  int mapXbins_st2 = 200;
  float mapXmin_st2 = 0. * TMath::Cos(18.4 / 180. * TMath::Pi());
  float mapXmax_st2 = 30. * TMath::Cos(18.4 / 180. * TMath::Pi());
  int mapYbins_st2 = 240;
  float mapYmin_st2 = -16.;
  float mapYmax_st2 = 8.;

  int mapXbins_st0 = 200;
  float mapXmin_st0 = 0. * TMath::Cos(18.4 / 180. * TMath::Pi());
  float mapXmax_st0 = 30. * TMath::Cos(18.4 / 180. * TMath::Pi());
  int mapYbins_st0 = 240;
  float mapYmin_st0 = -16.;
  float mapYmax_st0 = 8.;

  int mapXbins = mapXbins_st0;
  float mapXmin = mapXmin_st0;
  float mapXmax = mapXmax_st0;
  int mapYbins = mapYbins_st0;
  float mapYmin = mapYmin_st0;
  float mapYmax = mapYmax_st0;

  double xiBins = 44;
  double xiMin = 0.0;
  double xiMax = 0.22;
  double angleBins = 100;
  double angleMin = -0.03;
  double angleMax = 0.03;

  double nRecHitsBins = 100;
  double nRecHitsMin = 0;
  double nRecHitsMax = 200;

  int mapRowBins = 160;
  float mapRowMin = 0;
  float mapRowMax = 160;
  int mapColBins = 156;
  float mapColMin = 0;
  float mapColMax = 156;

  // Matching parameters
  // Cuts for 2018 re-MINIAOD
  double xMatchWindow45 = 4. * 0.16008188;
  double xMatchMean45 = -0.065194856;
  double yMatchWindow45 = 4. * 0.1407986;
  double yMatchMean45 = +0.10973631;
  double xiMatchWindow45 = 4. * 0.0012403586;
  double xiMatchMean45 = +3.113062e-5;

  double xMatchWindow56 = 5. * 0.18126434;
  double xMatchMean56 = +0.073016431;
  double yMatchWindow56 = 5. * 0.14990802;
  double yMatchMean56 = +0.064261029;
  double xiMatchWindow56 = 5. * 0.002046409;
  double xiMatchMean56 = -1.1852528e-5;

  bool excludeMultipleMatches = true;

  // output histograms
  // RP hists
  map<CTPPSPixelDetId, TH1D *> h1ProtonMux_;
  map<CTPPSPixelDetId, TH1D *> h1NumberOfRecHitsInPot_;
  map<CTPPSPixelDetId, TH1D *> h1NumberOfRecHitsInPotWithNoProton_;
  map<CTPPSPixelDetId, TH1D *> h1NumberOfRecHitsInPotWithOneProton_;
  map<CTPPSPixelDetId, TH1D *>
      h1NumberOfRecHitsInPotWithOneProtonMinusTrackHits_;
  map<CTPPSPixelDetId, TH2D *> h2CorrelationNumberOfRecHitsInPotWithNoProton_;
  map<CTPPSPixelDetId, TH2D *> h2CorrelationNumberOfTracks_;

  // Plane hists
  map<CTPPSPixelDetId, TH1D *> h1NumberOfRecHits_;
  map<CTPPSPixelDetId, TH1D *> h1NumberOfRecHitsWithNoProton_;
  map<CTPPSPixelDetId, TH2D *> h2AdditionalRecHitsMap_;

  vector<double> fiducialXLowVector_;
  vector<double> fiducialXHighVector_;
  vector<double> fiducialYLowVector_;
  vector<double> fiducialYHighVector_;
  map<pair<int, int>, double> fiducialXLow_;
  map<pair<int, int>, double> fiducialXHigh_;
  map<pair<int, int>, double> fiducialYLow_;
  map<pair<int, int>, double> fiducialYHigh_;
  int recoInfoCut_;
  int maxProtonsInPixelRP_;
};

ShowerAnalyzer_2018::ShowerAnalyzer_2018(const edm::ParameterSet &iConfig) {
  usesResource("TFileService");
  protonsToken_ = consumes<reco::ForwardProtonCollection>(
      edm::InputTag("ctppsProtons", "singleRP"));
  pixelLocalTrackToken_ = consumes<edm::DetSetVector<CTPPSPixelLocalTrack>>(
      edm::InputTag("ctppsPixelLocalTracks", ""));
  pixelRecHitToken_ = consumes<edm::DetSetVector<CTPPSPixelRecHit>>(
      edm::InputTag("ctppsPixelRecHits", ""));
  pixelClusterToken_ = consumes<edm::DetSetVector<CTPPSPixelCluster>>(
      edm::InputTag("ctppsPixelClusters", ""));

  outputFileName_ = iConfig.getUntrackedParameter<string>("outputFileName");
  minNumberOfPlanesForTrack_ =
      iConfig.getParameter<int>("minNumberOfPlanesForTrack");
  maxChi2Prob_ = iConfig.getUntrackedParameter<double>("maxChi2Prob");
  minTracksPerEvent = iConfig.getParameter<int>("minTracksPerEvent"); // UNUSED!
  maxTracksPerEvent = iConfig.getParameter<int>("maxTracksPerEvent"); // UNUSED!
  binGroupingX = iConfig.getUntrackedParameter<int>("binGroupingX");  // UNUSED!
  binGroupingY = iConfig.getUntrackedParameter<int>("binGroupingY");  // UNUSED!
  fiducialXLowVector_ =
      iConfig.getUntrackedParameter<vector<double>>("fiducialXLow");
  fiducialXHighVector_ =
      iConfig.getUntrackedParameter<vector<double>>("fiducialXHigh");
  fiducialYLowVector_ =
      iConfig.getUntrackedParameter<vector<double>>("fiducialYLow");
  fiducialYHighVector_ =
      iConfig.getUntrackedParameter<vector<double>>("fiducialYHigh");
  fiducialXLow_ = {
      {pair<int, int>(0, 0), fiducialXLowVector_.at(0)},
      {pair<int, int>(0, 2), fiducialXLowVector_.at(1)},
      {pair<int, int>(1, 0), fiducialXLowVector_.at(2)},
      {pair<int, int>(1, 2), fiducialXLowVector_.at(3)},
  };
  fiducialXHigh_ = {
      {pair<int, int>(0, 0), fiducialXHighVector_.at(0)},
      {pair<int, int>(0, 2), fiducialXHighVector_.at(1)},
      {pair<int, int>(1, 0), fiducialXHighVector_.at(2)},
      {pair<int, int>(1, 2), fiducialXHighVector_.at(3)},
  };
  fiducialYLow_ = {
      {pair<int, int>(0, 0), fiducialYLowVector_.at(0)},
      {pair<int, int>(0, 2), fiducialYLowVector_.at(1)},
      {pair<int, int>(1, 0), fiducialYLowVector_.at(2)},
      {pair<int, int>(1, 2), fiducialYLowVector_.at(3)},
  };
  fiducialYHigh_ = {
      {pair<int, int>(0, 0), fiducialYHighVector_.at(0)},
      {pair<int, int>(0, 2), fiducialYHighVector_.at(1)},
      {pair<int, int>(1, 0), fiducialYHighVector_.at(2)},
      {pair<int, int>(1, 2), fiducialYHighVector_.at(3)},
  };
  recoInfoCut_ = iConfig.getUntrackedParameter<int>("recoInfo");
  maxProtonsInPixelRP_ =
      iConfig.getUntrackedParameter<int>("maxProtonsInPixelRP");
  debug_ = iConfig.getUntrackedParameter<bool>("debug");
}

ShowerAnalyzer_2018::~ShowerAnalyzer_2018() {
  for (auto &rpId : romanPotIdVector_) {
    delete h1ProtonMux_[rpId];
    delete h1NumberOfRecHitsInPot_[rpId];
    delete h1NumberOfRecHitsInPotWithNoProton_[rpId];
    delete h1NumberOfRecHitsInPotWithOneProton_[rpId];
    delete h1NumberOfRecHitsInPotWithOneProtonMinusTrackHits_[rpId];
    delete h2CorrelationNumberOfRecHitsInPotWithNoProton_[rpId];
    delete h2CorrelationNumberOfTracks_[rpId];
  }

  for (auto &planeId : planeIdVector_) {
    delete h1NumberOfRecHits_[planeId];
    delete h1NumberOfRecHitsWithNoProton_[planeId];
    delete h2AdditionalRecHitsMap_[planeId];
  }
}

void ShowerAnalyzer_2018::fillDescriptions(
    edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

void ShowerAnalyzer_2018::beginJob() {}

void ShowerAnalyzer_2018::analyze(const edm::Event &iEvent,
                                  const edm::EventSetup &iSetup) {
  using namespace edm;

  Handle<reco::ForwardProtonCollection> protons;
  iEvent.getByToken(protonsToken_, protons);
  Handle<edm::DetSetVector<CTPPSPixelLocalTrack>> pixelTracks;
  iEvent.getByToken(pixelLocalTrackToken_, pixelTracks);
  Handle<edm::DetSetVector<CTPPSPixelRecHit>> pixelRecHits;
  iEvent.getByToken(pixelRecHitToken_, pixelRecHits);
  Handle<edm::DetSetVector<CTPPSPixelCluster>> pixelClusters;
  iEvent.getByToken(pixelClusterToken_, pixelClusters);

  // Compute proton mux
  std::map<CTPPSPixelDetId, int> mux;
  std::map<CTPPSPixelDetId, int> recHitsInPot;

  for (auto &proton : *protons) {
    if (!proton.validFit() ||
        proton.method() != reco::ForwardProton::ReconstructionMethod::singleRP)
      continue;
    CTPPSLocalTrackLite track = *(proton.contributingLocalTracks().at(0));
    CTPPSDetId detId = CTPPSDetId(track.rpId());
    int arm = detId.arm();
    int station = detId.station();
    int rp = detId.rp();
    CTPPSPixelDetId pixelDetId(detId.rawId());
    if (h1ProtonMux_.find(pixelDetId) == h1ProtonMux_.end()) {

      if (find(romanPotIdVector_.begin(), romanPotIdVector_.end(),
               pixelDetId) == romanPotIdVector_.end())
        romanPotIdVector_.push_back(pixelDetId);

      h1ProtonMux_[pixelDetId] = new TH1D(
          Form("h1ProtonMux_arm%i_st%i_rp%i", arm, station, rp),
          Form("h1ProtonMux_arm%i_st%i_rp%i;Protons", arm, station, rp), 11,
          -0.5, 10.5);
    }
    mux[pixelDetId]++;
  }

  for (auto &pixelDetId : romanPotIdVector_) {
    h1ProtonMux_[pixelDetId]->Fill(mux[pixelDetId]);
  }

  // Analyze rechits
  for (auto &recHitDs : *pixelRecHits) {
    // Get the id of the plane
    CTPPSPixelDetId recHitId = CTPPSPixelDetId(recHitDs.id);

    // Make it the RP id
    CTPPSPixelDetId pixelDetId = recHitId;
    pixelDetId.setPlane(0);

    int arm = recHitId.arm();
    int station = recHitId.station();
    int rp = recHitId.rp();
    int plane = recHitId.plane();

    // Book missing RP hists
    if (h1NumberOfRecHitsInPot_.find(pixelDetId) ==
        h1NumberOfRecHitsInPot_.end()) {
      // If the id is not already in the vector, add it, and book the hist that
      // previously was not booked
      if (find(romanPotIdVector_.begin(), romanPotIdVector_.end(),
               pixelDetId) == romanPotIdVector_.end()) {
        romanPotIdVector_.push_back(pixelDetId);
        h1ProtonMux_[pixelDetId] = new TH1D(
            Form("h1ProtonMux_arm%i_st%i_rp%i", arm, station, rp),
            Form("h1ProtonMux_arm%i_st%i_rp%i;Protons", arm, station, rp), 11,
            -0.5, 10.5);
      }
      h1NumberOfRecHitsInPot_[pixelDetId] = new TH1D(
          Form("h1NumberOfRecHitsInPot_arm%i_st%i_rp%i", arm, station, rp),
          Form("h1NumberOfRecHitsInPot_arm%i_st%i_rp%i;# of RecHits", arm,
               station, rp),
          nRecHitsBins, nRecHitsMin, nRecHitsMax);
      h1NumberOfRecHitsInPotWithNoProton_[pixelDetId] = new TH1D(
          Form("h1NumberOfRecHitsInPotWithNoProton_arm%i_st%i_rp%i", arm,
               station, rp),
          Form(
              "h1NumberOfRecHitsInPotWithNoProton_arm%i_st%i_rp%i;# of RecHits",
              arm, station, rp),
          nRecHitsBins, nRecHitsMin, nRecHitsMax);
      h1NumberOfRecHitsInPotWithOneProton_[pixelDetId] =
          new TH1D(Form("h1NumberOfRecHitsInPotWithOneProton_arm%i_st%i_rp%i",
                        arm, station, rp),
                   Form("h1NumberOfRecHitsInPotWithOneProton_arm%i_st%i_rp%i;# "
                        "of RecHits",
                        arm, station, rp),
                   nRecHitsBins, nRecHitsMin, nRecHitsMax);
      h1NumberOfRecHitsInPotWithOneProtonMinusTrackHits_[pixelDetId] =
          new TH1D(Form("h1NumberOfRecHitsInPotWithOneProtonMinusTrackHits_arm%"
                        "i_st%i_rp%i",
                        arm, station, rp),
                   Form("h1NumberOfRecHitsInPotWithOneProtonMinusTrackHits_arm%"
                        "i_st%i_rp%i;# of RecHits",
                        arm, station, rp),
                   nRecHitsBins, nRecHitsMin, nRecHitsMax);
      h2CorrelationNumberOfRecHitsInPotWithNoProton_[pixelDetId] = new TH2D(
          Form("h2CorrelationNumberOfRecHitsInPotWithNoProton_arm%i_st%i_rp%i",
               arm, station, rp),
          Form("h2CorrelationNumberOfRecHitsInPotWithNoProton_arm%i_st%i_rp%i;#"
               " of RecHits arm%i_st%i;# of RecHits arm%i_st%i;",
               arm, station, rp, arm, station, arm, ((station == 0) ? 2 : 0)),
          nRecHitsBins, nRecHitsMin, nRecHitsMax, nRecHitsBins, nRecHitsMin,
          nRecHitsMax);
      h2CorrelationNumberOfTracks_[pixelDetId] = new TH2D(
          Form("h2CorrelationNumberOfTracks_arm%i_st%i_rp%i", arm, station, rp),
          Form("h2CorrelationNumberOfTracks_arm%i_st%i_rp%i;#"
               " of Tracks arm%i_st%i;# of Tracks arm%i_st%i;",
               arm, station, rp, arm, station, arm, ((station == 0) ? 2 : 0)),
          11, 0, 11, 11, 0, 11);
    }

    // Book missing plane hists
    if (h1NumberOfRecHits_.find(recHitId) == h1NumberOfRecHits_.end()) {
      planeIdVector_.push_back(recHitId);
      TString planeTag = Form("arm%i_st%i_rp%i_pl%i", arm, station, rp, plane);
      h1NumberOfRecHits_[recHitId] =
          new TH1D("h1NumberOfRecHits_" + planeTag,
                   "h1NumberOfRecHits_" + planeTag + ";Number of RecHits",
                   nRecHitsBins, nRecHitsMin, nRecHitsMax);
      h1NumberOfRecHitsWithNoProton_[recHitId] = new TH1D(
          "h1NumberOfRecHitsWithNoProton_" + planeTag,
          "h1NumberOfRecHitsWithNoProton_" + planeTag + ";Number of RecHits",
          nRecHitsBins, nRecHitsMin, nRecHitsMax);
      h2AdditionalRecHitsMap_[recHitId] =
          new TH2D("h2AdditionalRecHitsMap_" + planeTag,
                   "h2AdditionalRecHitsMap_" + planeTag + "Col;Row", mapColMax,
                   mapColMin, mapColMax, mapRowBins, mapRowMin, mapRowMax);
    }

    recHitsInPot[pixelDetId] += recHitDs.data.size();
    h1NumberOfRecHits_[recHitId]->Fill(recHitDs.data.size());

    if (mux[pixelDetId] == 0)
      h1NumberOfRecHitsWithNoProton_[recHitId]->Fill(recHitDs.data.size());

    if (mux[pixelDetId] == 1) {
      // Fill map with all hits positions
      for (auto &hit : recHitDs.data)
        h2AdditionalRecHitsMap_[recHitId]->Fill(hit.minPixelCol(),
                                                hit.minPixelRow());
    }
  }

  for (auto &pixelDetId : romanPotIdVector_) {
    CTPPSPixelDetId otherStationId = CTPPSPixelDetId(
        pixelDetId.arm(), ((pixelDetId.station() == 0) ? 2 : 0), 3);

    h1NumberOfRecHitsInPot_[pixelDetId]->Fill(recHitsInPot[pixelDetId]);
    h2CorrelationNumberOfTracks_[pixelDetId]->Fill(mux[pixelDetId],
                                                   mux[otherStationId]);
    if (mux[pixelDetId] == 0) {
      h1NumberOfRecHitsInPotWithNoProton_[pixelDetId]->Fill(
          recHitsInPot[pixelDetId]);

      // Fill correlation hists
      if (mux[otherStationId] == 0) {
        h2CorrelationNumberOfRecHitsInPotWithNoProton_[pixelDetId]->Fill(
            recHitsInPot[pixelDetId], recHitsInPot[otherStationId]);
      }
    }

    if (mux[pixelDetId] == 1) {
      h1NumberOfRecHitsInPotWithOneProton_[pixelDetId]->Fill(
          recHitsInPot[pixelDetId]);

      int numberOfHitsInTrack = 0;
      bool foundTrack = false;
      // Get the number of hits used for the track found
      for (auto &proton : *protons) {
        if (!proton.validFit() ||
            proton.method() !=
                reco::ForwardProton::ReconstructionMethod::singleRP)
          continue;
        CTPPSLocalTrackLite trackLite =
            *(proton.contributingLocalTracks().at(0));
        if (trackLite.rpId() != pixelDetId.rawId()) {
          continue;
        }
        foundTrack = true;
        numberOfHitsInTrack = trackLite.numberOfPointsUsedForFit();

        // Find fat track correspondent to trackLite
        auto rpPixelTracks = (*pixelTracks)[trackLite.rpId()];
        CTPPSPixelLocalTrack track;
        bool associatedTrackFound = false;
        for (auto &pixelTrack : rpPixelTracks) {
          if (TMath::Abs(pixelTrack.x0() - trackLite.x()) < 0.1 &&
              TMath::Abs(pixelTrack.y0() - trackLite.y()) < 0.1) {
            track = pixelTrack;
            associatedTrackFound = true;
            if (debug_)
              cout << "Found track to be associated" << endl;
            break;
          }
        } // for (auto &pixelTrack : rpPixelTracks)
        if (!associatedTrackFound) {
          cout << "WARNING: no track associated to trackLite found!" << endl;
          continue;
        }

        // Remove the hits used for the track fit from the map
        for (auto &fittedHitDs : track.hits()) {
          // There's actually only one hit per plane here, so this iteration is
          // just for being elegant
          if (fittedHitDs.data.size() > 1)
            cout << "WARNING: more than one fitted hit in the same plane for "
                    "the same track!!!"
                 << endl;
          for (auto &fittedHit : fittedHitDs.data) {
            if (fittedHit.isRealHit()) {
              h2AdditionalRecHitsMap_[CTPPSPixelDetId(fittedHitDs.id)]->Fill(
                  fittedHit.minPixelCol(), fittedHit.minPixelRow(), -1);
            }
          }
        }
        break;
      }
      if (!foundTrack) {
        cout << "WARNING: track not found in 1 track event" << endl;
        continue;
      }
      h1NumberOfRecHitsInPotWithOneProtonMinusTrackHits_[pixelDetId]->Fill(
          recHitsInPot[pixelDetId] - numberOfHitsInTrack);
    } // if (mux[pixelDetId] == 1)
  }
}

void ShowerAnalyzer_2018::endJob() {
  TFile *outputFile_ = new TFile(outputFileName_.data(), "RECREATE");
  for (auto &rpId : romanPotIdVector_) {
    uint32_t arm = rpId.arm();
    uint32_t station = rpId.station();
    string rpDirName = Form("Arm%i_st%i_rp3", arm, station);
    outputFile_->mkdir(rpDirName.data());
    outputFile_->cd(rpDirName.data());

    if (h1ProtonMux_.find(rpId) != h1ProtonMux_.end()) {
      h1ProtonMux_[rpId]->Write();
    }
    if (h1NumberOfRecHitsInPot_.find(rpId) != h1NumberOfRecHitsInPot_.end()) {
      h1NumberOfRecHitsInPot_[rpId]->Write();
      h1NumberOfRecHitsInPotWithNoProton_[rpId]->Write();
      h1NumberOfRecHitsInPotWithOneProton_[rpId]->Write();
      h1NumberOfRecHitsInPotWithOneProtonMinusTrackHits_[rpId]->Write();
      h2CorrelationNumberOfRecHitsInPotWithNoProton_[rpId]->Write();
      h2CorrelationNumberOfTracks_[rpId]->Write();
    }
  }

  for (auto &planeId : planeIdVector_) {
    uint32_t arm = planeId.arm();
    uint32_t station = planeId.station();
    uint32_t plane = planeId.plane();

    string planeDirName = Form("Arm%i_st%i_rp3/Arm%i_st%i_rp3_pl%i", arm,
                               station, arm, station, plane);
    outputFile_->mkdir(planeDirName.data());
    outputFile_->cd(planeDirName.data());

    if (h1NumberOfRecHits_.find(planeId) != h1NumberOfRecHits_.end()) {
      h1NumberOfRecHits_[planeId]->Write();
      h1NumberOfRecHitsWithNoProton_[planeId]->Write();
      h2AdditionalRecHitsMap_[planeId]->Write();
    }
  }
  outputFile_->Close();
  delete outputFile_;
}

bool ShowerAnalyzer_2018::Cut(CTPPSLocalTrackLite track) {
  CTPPSDetId detId = CTPPSDetId(track.rpId());
  uint32_t arm = detId.arm();
  uint32_t station = detId.station();
  uint32_t ndf = 2 * track.numberOfPointsUsedForFit() - 4;
  double x = track.x();
  double y = track.y();
  if (station == 0) {
    double pixelX0_rotated = x * TMath::Cos((-8. / 180.) * TMath::Pi()) -
                             y * TMath::Sin((-8. / 180.) * TMath::Pi());
    double pixelY0_rotated = x * TMath::Sin((-8. / 180.) * TMath::Pi()) +
                             y * TMath::Cos((-8. / 180.) * TMath::Pi());
    x = pixelX0_rotated;
    y = pixelY0_rotated;
  }

  double maxTx = 0.03;
  double maxTy = 0.04;
  double maxChi2 = TMath::ChisquareQuantile(maxChi2Prob_, ndf);
  if (station == 2) {
    if (debug_) {
      if (track.chiSquaredOverNDF() * ndf > maxChi2)
        cout << "Chi2 cut not passed" << endl;
      if (TMath::Abs(track.tx()) > maxTx)
        cout << "maxTx cut not passed" << endl;
      if (TMath::Abs(track.ty()) > maxTy)
        cout << "maxTy cut not passed" << endl;
      if (track.numberOfPointsUsedForFit() < minNumberOfPlanesForTrack_)
        cout << "Too few planes for track" << endl;
      if (track.numberOfPointsUsedForFit() > maxNumberOfPlanesForTrack_)
        cout << "Too many planes for track" << endl;
      if (y > fiducialYHigh_[pair<int, int>(arm, station)])
        cout << "fiducialYHigh cut not passed" << endl;
      if (y < fiducialYLow_[pair<int, int>(arm, station)])
        cout << "fiducialYLow cut not passed" << endl;
      if (x < fiducialXLow_[pair<int, int>(arm, station)])
        cout << "fiducialXLow cut not passed" << endl;
    }
    if (TMath::Abs(track.tx()) > maxTx ||
        TMath::Abs(track.ty()) > maxTy ||
        track.chiSquaredOverNDF() * ndf > maxChi2 ||
        track.numberOfPointsUsedForFit() < minNumberOfPlanesForTrack_ ||
        track.numberOfPointsUsedForFit() > maxNumberOfPlanesForTrack_ ||
        y > fiducialYHigh_[pair<int, int>(arm, station)] ||
        y < fiducialYLow_[pair<int, int>(arm, station)] ||
        x < fiducialXLow_[pair<int, int>(arm, station)])
      return true;
    else {
      if (recoInfoCut_ != 5) {
        if (recoInfoCut_ != -1) {
          if ((int)track.pixelTrackRecoInfo() != recoInfoCut_)
            return true;
          else
            return false;
        } else {
          if ((int)track.pixelTrackRecoInfo() != 0 &&
              (int)track.pixelTrackRecoInfo() != 2)
            return true;
          else
            return false;
        }
      } else
        return false;
    }
  } else {
    if (station == 0) {
      if (TMath::Abs(track.tx()) > maxTx ||
          TMath::Abs(track.ty()) > maxTy)
        return true;
      else
        return false;
    } else
      throw "Station is neither 0 or 2!!!";
  }
}

bool ShowerAnalyzer_2018::CutWithNoNumberOfPoints(CTPPSLocalTrackLite track) {
  CTPPSDetId detId = CTPPSDetId(track.rpId());
  uint32_t arm = detId.arm();
  uint32_t station = detId.station();
  uint32_t ndf = 2 * track.numberOfPointsUsedForFit() - 4;
  double x = track.x();
  double y = track.y();
  if (station == 0) {
    double pixelX0_rotated = x * TMath::Cos((-8. / 180.) * TMath::Pi()) -
                             y * TMath::Sin((-8. / 180.) * TMath::Pi());
    double pixelY0_rotated = x * TMath::Sin((-8. / 180.) * TMath::Pi()) +
                             y * TMath::Cos((-8. / 180.) * TMath::Pi());
    x = pixelX0_rotated;
    y = pixelY0_rotated;
  }

  double maxTx = 0.03;
  double maxTy = 0.04;
  double maxChi2 = TMath::ChisquareQuantile(maxChi2Prob_, ndf);
  if (station == 2) {
    if (debug_) {
      if (track.chiSquaredOverNDF() * ndf > maxChi2)
        cout << "Chi2 cut not passed" << endl;
      if (TMath::Abs(track.tx()) > maxTx)
        cout << "maxTx cut not passed" << endl;
      if (TMath::Abs(track.ty()) > maxTy)
        cout << "maxTy cut not passed" << endl;
      if (y > fiducialYHigh_[pair<int, int>(arm, station)])
        cout << "fiducialYHigh cut not passed" << endl;
      if (y < fiducialYLow_[pair<int, int>(arm, station)])
        cout << "fiducialYLow cut not passed" << endl;
      if (x < fiducialXLow_[pair<int, int>(arm, station)])
        cout << "fiducialXLow cut not passed" << endl;
    }
    if (TMath::Abs(track.tx()) > maxTx ||
        TMath::Abs(track.ty()) > maxTy ||
        track.chiSquaredOverNDF() * ndf > maxChi2 ||
        y > fiducialYHigh_[pair<int, int>(arm, station)] ||
        y < fiducialYLow_[pair<int, int>(arm, station)] ||
        x < fiducialXLow_[pair<int, int>(arm, station)])
      return true;
    else {
      if (recoInfoCut_ != 5) {
        if (recoInfoCut_ != -1) {
          if ((int)track.pixelTrackRecoInfo() != recoInfoCut_)
            return true;
          else
            return false;
        } else {
          if ((int)track.pixelTrackRecoInfo() != 0 &&
              (int)track.pixelTrackRecoInfo() != 2)
            return true;
          else
            return false;
        }
      } else
        return false;
    }
  } else {
    if (station == 0) {
      if (TMath::Abs(track.tx()) > maxTx ||
          TMath::Abs(track.ty()) > maxTy)
        return true;
      else
        return false;
    } else
      throw "Station is neither 0 or 2!!!";
  }
}

// define this as a plug-in
DEFINE_FWK_MODULE(ShowerAnalyzer_2018);