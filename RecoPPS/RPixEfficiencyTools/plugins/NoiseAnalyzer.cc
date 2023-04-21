// -*- C++ -*-
//
// Package:    RecoPPS/RPixEfficiencyTools
// Class:      NoiseAnalyzer
//
/**\class NoiseAnalyzer_2018 NoiseAnalyzer_2018.cc
 RecoPPS/RPixEfficiencyTools/plugins/NoiseAnalyzer_2018.cc

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

class NoiseAnalyzer_2018
    : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit NoiseAnalyzer_2018(const edm::ParameterSet &);
  ~NoiseAnalyzer_2018();
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

  double chargeBins = 200;
  double chargeMin = 0;
  double chargeMax = 50000;

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
  map<CTPPSPixelDetId, TH1D *> h1ProtonMux_;
  map<CTPPSPixelDetId, TH1D *> h1AllClustersCharge_;
  map<CTPPSPixelDetId, TH1D *> h1FittedClustersCharge_;
  map<CTPPSPixelDetId, TH1D *> h1MatchingTrackClustersCharge_;
  map<CTPPSPixelDetId, TH1D *> h1NotMatchingTrackClustersCharge_;

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

NoiseAnalyzer_2018::NoiseAnalyzer_2018(const edm::ParameterSet &iConfig) {
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

NoiseAnalyzer_2018::~NoiseAnalyzer_2018() {
  for (auto &rpId : romanPotIdVector_) {
    delete h1ProtonMux_[rpId];
  }

  for (auto &planeId : planeIdVector_) {
    delete h1AllClustersCharge_[planeId];
    delete h1FittedClustersCharge_[planeId];
    delete h1MatchingTrackClustersCharge_[planeId];
    delete h1NotMatchingTrackClustersCharge_[planeId];
  }
}

void NoiseAnalyzer_2018::fillDescriptions(
    edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

void NoiseAnalyzer_2018::beginJob() {}

void NoiseAnalyzer_2018::analyze(const edm::Event &iEvent,
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

  // Analyze clusters
  for (auto &clusterDs : *pixelClusters) {
    CTPPSPixelDetId clusterId = CTPPSPixelDetId(clusterDs.id);
    int arm = clusterId.arm();
    int station = clusterId.station();
    int rp = clusterId.rp();
    int plane = clusterId.plane();
    if (h1AllClustersCharge_.find(clusterId) == h1AllClustersCharge_.end()) {
      planeIdVector_.push_back(clusterId);
      TString planeTag = Form("arm%i_st%i_rp%i_pl%i", arm, station, rp, plane);
      h1AllClustersCharge_[clusterId] =
          new TH1D("h1AllClustersCharge_" + planeTag,
                   "h1AllClustersCharge_" + planeTag + ";Electrons", chargeBins,
                   chargeMin, chargeMax);
    }
    for (auto &cluster : clusterDs.data) {
      h1AllClustersCharge_[clusterId]->Fill(cluster.charge());
    }
  }

  // Analyze the event
  for (auto &proton : *protons) {
    if (!proton.validFit())
      continue;

    CTPPSLocalTrackLite trackLite = *(proton.contributingLocalTracks().at(0));
    CTPPSDetId detId = CTPPSDetId(trackLite.rpId());
    int arm = detId.arm();
    int station = detId.station();
    int rp = detId.rp();
    double trackX0 = trackLite.x();
    double trackY0 = trackLite.y();
    double xi = proton.xi();

    // Apply the usual general-purpose cut
    if (Cut(trackLite))
      continue;

    // Don't analyze protons when too many or too few protons are in the same RP
    if (mux[detId] > maxTracksPerEvent || mux[detId] < minTracksPerEvent) {
      if (debug_)
        cout << "Mux cut not passed" << endl;
      continue;
    }

    // Just a few printouts for debugging
    if (debug_) {
      cout << endl;
      cout << endl;
      cout << "Analyzing track in arm " << arm << " station " << station
           << endl;
      cout << "X = " << trackX0 << "\tY = " << trackY0 << "\tXi = " << xi
           << endl;
    }

    // Matching part
    int interPotMatches = 0;
    bool interPotMatch = false;
    for (auto &proton_Probe : *protons) {
      if (!proton.validFit())
        continue;
      CTPPSLocalTrackLite trackLite_Probe =
          *(proton_Probe.contributingLocalTracks().at(0));
      CTPPSDetId detId_Probe = CTPPSDetId(trackLite_Probe.rpId());

      // Apply the usual general-purpose cut
      if (CutWithNoNumberOfPoints(trackLite_Probe))
        continue;

      // Ask the singleRP proton to be in the same arm, different station
      if ((int)detId_Probe.arm() != arm ||
          (int)detId_Probe.station() == station)
        continue;

      bool xiMatchPass = false;
      bool yMatchPass = false;
      bool xMatchPass = false;

      double trackX0_Probe = trackLite_Probe.x();
      double trackY0_Probe = trackLite_Probe.y();
      double xi_Probe = proton_Probe.xi();

      if (arm == 0 &&
          TMath::Abs(xi_Probe - xi - xiMatchMean45) < xiMatchWindow45)
        xiMatchPass = true;
      if (arm == 0 &&
          TMath::Abs(trackY0 - trackY0 - yMatchMean45) < yMatchWindow45)
        yMatchPass = true;
      if (arm == 0 &&
          TMath::Abs(trackX0 - trackX0 - xMatchMean45) < xMatchWindow45)
        xMatchPass = true;
      if (arm == 1 &&
          TMath::Abs(xi_Probe - xi - xiMatchMean56) < xiMatchWindow56)
        xiMatchPass = true;
      if (arm == 1 &&
          TMath::Abs(trackY0_Probe - trackY0 - yMatchMean56) < yMatchWindow56)
        yMatchPass = true;
      if (arm == 1 &&
          TMath::Abs(trackX0_Probe - trackX0 - xMatchMean56) < xMatchWindow56)
        xMatchPass = true;

      if (xiMatchPass && xMatchPass && yMatchPass)
        interPotMatches++;
    }

    if (excludeMultipleMatches && interPotMatches == 1)
      interPotMatch = true;
    if (!excludeMultipleMatches && interPotMatches >= 1)
      interPotMatch = true;

    if (debug_ && interPotMatch)
      cout << "Found interPot match!" << endl;
    else if (debug_)
      cout << "InterPot match not found" << endl;

    // Find fat track correspondent to trackLite
    auto rpPixelTracks = (*pixelTracks)[trackLite.rpId()];
    CTPPSPixelLocalTrack track;
    bool associatedTrackFound = false;
    for (auto &pixelTrack : rpPixelTracks) {
      if (TMath::Abs(pixelTrack.x0() - trackX0) < 0.1 &&
          TMath::Abs(pixelTrack.y0() - trackY0) < 0.1 &&
          !associatedTrackFound) {
        track = pixelTrack;
        associatedTrackFound = true;
        if (debug_)
          cout << "Found track to be associated" << endl;
      }
    } // for (auto &pixelTrack : rpPixelTracks)
    if (!associatedTrackFound) {
      cout << "WARNING: no track associated to trackLite found!" << endl;
      continue;
    }

    // Get the hits in the track
    edm::DetSetVector<CTPPSPixelFittedRecHit> fittedHits = track.hits();
    for (auto &plane : listOfPlanes_) {
      CTPPSPixelDetId planeId(detId);
      planeId.setPlane(plane);

      if (h1FittedClustersCharge_.find(planeId) ==
          h1FittedClustersCharge_.end()) {
        TString planeTag =
            Form("arm%i_st%i_rp%i_pl%i", arm, station, rp, plane);
        h1FittedClustersCharge_[planeId] =
            new TH1D("h1FittedClustersCharge_" + planeTag,
                     "h1FittedClustersCharge_" + planeTag + ";Electrons",
                     chargeBins, chargeMin, chargeMax);
        h1MatchingTrackClustersCharge_[planeId] =
            new TH1D("h1MatchingTrackClustersCharge_" + planeTag,
                     "h1MatchingTrackClustersCharge_" + planeTag + ";Electrons",
                     chargeBins, chargeMin, chargeMax);
        h1NotMatchingTrackClustersCharge_[planeId] = new TH1D(
            "h1NotMatchingTrackClustersCharge_" + planeTag,
            "h1NotMatchingTrackClustersCharge_" + planeTag + ";Electrons",
            chargeBins, chargeMin, chargeMax);
      }

      auto planeFittedHit =
          fittedHits[planeId.rawId()][0]; // There is always only one fitted hit
      if (!planeFittedHit.isRealHit())
        continue;

      double hitX0 = planeFittedHit.globalCoordinates().x() +
                     planeFittedHit.xResidual();
      double hitY0 = planeFittedHit.globalCoordinates().y() +
                     planeFittedHit.yResidual();

      int minPixelRow = planeFittedHit.minPixelRow();
      int minPixelCol = planeFittedHit.minPixelCol();
      int clsRow = planeFittedHit.clusterSizeRow();
      int clsCol = planeFittedHit.clusterSizeCol();

      // Few printouts for debugging
      if (debug_) {
        cout << endl;
        cout << "Analyzing hit on plane: " << plane
             << "\tID: " << planeId.rawId() << endl;
        cout << "X = " << hitX0 << "\tY = " << hitY0
             << "\tminRow = " << minPixelRow << "\tminCol = " << minPixelCol
             << "\tclsRow = " << clsRow << "\tclsCol = " << clsCol << endl;
      }

      // Find the cluster associated with the hit
      CTPPSPixelCluster cluster;
      bool associatedClusterFound = false;
      for (auto &planeCluster : (*pixelClusters)[planeId.rawId()]) {

        if ((int)planeCluster.minPixelCol() == minPixelCol &&
            (int)planeCluster.minPixelRow() == minPixelRow) {
          if (debug_) {
            cout << "Found cluster!" << endl;
            cout << "Charge = " << planeCluster.charge() << endl;
          }
          associatedClusterFound = true;
          cluster = planeCluster;
        }
      }
      if (!associatedClusterFound) {
        cout << "WARNING: no cluster associated to real fittedRecHit found!"
             << endl;
        continue;
      }

      h1FittedClustersCharge_[planeId]->Fill(cluster.charge());

      if (interPotMatch)
        h1MatchingTrackClustersCharge_[planeId]->Fill(cluster.charge());
      else
        h1NotMatchingTrackClustersCharge_[planeId]->Fill(cluster.charge());
    } // for (auto &plane : listOfPlanes_)
  }   // for (auto &proton : *protons)
}

void NoiseAnalyzer_2018::endJob() {
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
  }

  for (auto &planeId : planeIdVector_) {
    uint32_t arm = planeId.arm();
    uint32_t station = planeId.station();
    uint32_t plane = planeId.plane();

    string planeDirName = Form("Arm%i_st%i_rp3/Arm%i_st%i_rp3_pl%i", arm,
                               station, arm, station, plane);
    outputFile_->mkdir(planeDirName.data());
    outputFile_->cd(planeDirName.data());

    if (h1AllClustersCharge_.find(planeId) != h1AllClustersCharge_.end()) {
      h1AllClustersCharge_[planeId]->Write();
      h1FittedClustersCharge_[planeId]->Write();
      h1MatchingTrackClustersCharge_[planeId]->Write();
      h1NotMatchingTrackClustersCharge_[planeId]->Write();
    }
  }
  outputFile_->Close();
  delete outputFile_;
}

bool NoiseAnalyzer_2018::Cut(CTPPSLocalTrackLite track) {
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

bool NoiseAnalyzer_2018::CutWithNoNumberOfPoints(CTPPSLocalTrackLite track) {
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
DEFINE_FWK_MODULE(NoiseAnalyzer_2018);