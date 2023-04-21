// -*- C++ -*-
//
// Package:    RecoPPS/RPixEfficiencyTools
// Class:      InterpotEfficiency_2018
//
/**\class InterpotEfficiency_2018 InterpotEfficiency_2018.cc
 RecoPPS/RPixEfficiencyTools/plugins/InterpotEfficiency_2018.cc

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
#include "DataFormats/CTPPSReco/interface/CTPPSPixelLocalTrack.h"
#include "DataFormats/CTPPSReco/interface/CTPPSPixelRecHit.h"

#include "DataFormats/ProtonReco/interface/ForwardProton.h"
#include "DataFormats/ProtonReco/interface/ForwardProtonFwd.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

// LHC Info
#include "CondFormats/DataRecord/interface/LHCInfoRcd.h"
#include "CondFormats/RunInfo/interface/LHCInfo.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include <TEfficiency.h>
#include <TF1.h>
#include <TFile.h>
#include <TGraphErrors.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TMath.h>
#include <TObjArray.h>

float Aperture(Float_t xangle, Int_t arm, TString era);

class InterpotEfficiency_2018
    : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit InterpotEfficiency_2018(const edm::ParameterSet &);
  ~InterpotEfficiency_2018();
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  virtual void beginJob() override;
  virtual void analyze(const edm::Event &, const edm::EventSetup &) override;
  virtual void endJob() override;
  bool Cut(CTPPSLocalTrackLite track);

  bool debug_ = false;
  bool fancyBinning_ = true;

  // z position of the pots (mm)
  std::map<CTPPSDetId, double> Z = {
      {CTPPSDetId(3, 0, 0, 3), -212550}, // strips, arm0, station0, rp3
      {CTPPSDetId(3, 1, 0, 3), 212550},  // strips, arm1, station0, rp3
      {CTPPSDetId(4, 0, 2, 3), -219550}, // pixels, arm0, station2, rp3
      {CTPPSDetId(4, 1, 2, 3), 219550}}; // pixels, arm1, station2, rp3

  // Data to get
  edm::EDGetTokenT<reco::ForwardProtonCollection> protonsToken_;
  edm::EDGetTokenT<reco::ForwardProtonCollection> multiRP_protonsToken_;

  // Parameter set
  std::string outputFileName_;
  int minNumberOfPlanesForTrack_;
  int maxNumberOfPlanesForTrack_ = 6;
  uint32_t maxTracksInTagPot = 99;
  uint32_t minTracksInTagPot = 0;
  uint32_t maxTracksInProbePot = 99;
  uint32_t minTracksInProbePot = 0;
  double maxChi2Prob_;
  std::string producerTag;

  // Configs
  std::vector<uint32_t> listOfArms_ = {0, 1};
  std::vector<uint32_t> listOfStations_ = {2};
  std::vector<uint32_t> listOfPlanes_ = {0, 1, 2, 3, 4, 5};

  std::vector<CTPPSPixelDetId> detectorIdVector_;
  std::vector<CTPPSPixelDetId> romanPotIdVector_;

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

  double mapXbinSize_small = (mapXmax - mapXmin) / mapXbins;
  double mapXbinSize_large = (mapXmax - mapXmin) / mapXbins * 2;

  std::map<CTPPSPixelDetId, double> mapXbin_changeCoordinate = {
      {CTPPSPixelDetId(0, 0, 3), 13},
      {CTPPSPixelDetId(0, 2, 3), 13},
      {CTPPSPixelDetId(1, 0, 3), 13},
      {CTPPSPixelDetId(1, 2, 3), 13}};

  std::map<CTPPSPixelDetId, int> nBinsX_total;
  std::map<CTPPSPixelDetId, std::vector<double>> xBinEdges;

  double xiBins = 41;
  double xiMin = 0;
  double xiMax = 0.22;
  double angleBins = 100;
  double angleMin = -0.03;
  double angleMax = 0.03;

  double xiMatchMean45 = +3.113062e-5;
  double xiMatchMean56 = -1.1852528e-5;
  double yMatchMean45 = +0.10973631;
  double yMatchMean56 = +0.064261029;
  double xMatchMean45 = -0.065194856;
  double xMatchMean56 = +0.073016431;
  double xiMatchWindow45 = 4. * 0.0012403586;
  double xiMatchWindow56 = 5. * 0.002046409;
  double yMatchWindow45 = 4. * 0.1407986;
  double yMatchWindow56 = 5. * 0.14990802;
  double xMatchWindow45 = 4. * 0.16008188;
  double xMatchWindow56 = 5. * 0.18126434;
  bool excludeMultipleMatches = false;

  // Number of times that a Tag proton matched more than one Probe
  std::map<CTPPSPixelDetId, uint32_t> overmatches;
  std::map<CTPPSPixelDetId, uint32_t> tries;

  // std::map<CTPPSPixelDetId, int> binAlignmentParameters = {
  //     {CTPPSPixelDetId(0, 0, 3), 0},
  //     {CTPPSPixelDetId(0, 2, 3), 0},
  //     {CTPPSPixelDetId(1, 0, 3), 0},
  //     {CTPPSPixelDetId(1, 2, 3), 0}};

  // output histograms
  std::map<CTPPSPixelDetId, TH2D *> h2AuxProtonHitDistribution_;
  std::map<CTPPSPixelDetId, TH2D *> h2AuxProtonHitDistributionWithNoMultiRP_;
  std::map<CTPPSPixelDetId, TH2D *> h2InterPotEfficiencyMap_;
  std::map<CTPPSPixelDetId, TH2D *> h2InterPotEfficiencyMapMultiRP_;
  std::map<CTPPSPixelDetId, TH1D *> h1AuxXi_;
  std::map<CTPPSPixelDetId, TH1D *> h1InterPotEfficiencyVsXi_;
  std::map<CTPPSPixelDetId, TH1D *> h1DeltaXiMatch_;
  std::map<CTPPSPixelDetId, TH1D *> h1DeltaYMatch_;
  std::map<CTPPSPixelDetId, TH1D *> h1TxMatch_;
  std::map<CTPPSPixelDetId, TH1D *> h1TyMatch_;
  std::map<CTPPSPixelDetId, TH1D *> h1ProtonsInProbePotWhenNoMatchFound_;
  std::map<CTPPSPixelDetId, TH2D *> h2TxCorrelationMatch_;
  std::map<CTPPSPixelDetId, TH2D *> h2TyCorrelationMatch_;
  std::map<CTPPSPixelDetId, TH2D *> h2XCorrelationMatch_;
  std::map<CTPPSPixelDetId, TH2D *> h2YCorrelationMatch_;

  std::map<CTPPSPixelDetId, TH1D *> h1TrackMux_;
  std::map<CTPPSPixelDetId, uint32_t> trackMux_;

  // file to insert the output hists in
  TFile *efficiencyFile_;

  std::vector<double> fiducialXLowVector_;
  std::vector<double> fiducialYLowVector_;
  std::vector<double> fiducialYHighVector_;
  std::map<std::pair<int, int>, double> fiducialXLow_;
  std::map<std::pair<int, int>, double> fiducialYLow_;
  std::map<std::pair<int, int>, double> fiducialYHigh_;

  int recoInfoCut_;
};

InterpotEfficiency_2018::InterpotEfficiency_2018(
    const edm::ParameterSet &iConfig) {
  usesResource("TFileService");

  producerTag = iConfig.getUntrackedParameter<std::string>("producerTag");

  protonsToken_ = consumes<reco::ForwardProtonCollection>(
      edm::InputTag("ctppsProtons", "singleRP", producerTag));
  multiRP_protonsToken_ = consumes<reco::ForwardProtonCollection>(
      edm::InputTag("ctppsProtons", "multiRP", producerTag));

  outputFileName_ =
      iConfig.getUntrackedParameter<std::string>("outputFileName");
  minNumberOfPlanesForTrack_ =
      iConfig.getUntrackedParameter<int>("minNumberOfPlanesForTrack");
  maxTracksInTagPot = iConfig.getUntrackedParameter<int>("maxTracksInTagPot");
  minTracksInTagPot = iConfig.getUntrackedParameter<int>("minTracksInTagPot");
  maxTracksInProbePot =
      iConfig.getUntrackedParameter<int>("maxTracksInProbePot");
  minTracksInProbePot =
      iConfig.getUntrackedParameter<int>("minTracksInProbePot");
  maxChi2Prob_ = iConfig.getUntrackedParameter<double>("maxChi2Prob");

  binGroupingX = iConfig.getUntrackedParameter<int>("binGroupingX"); // UNUSED!
  binGroupingY = iConfig.getUntrackedParameter<int>("binGroupingY"); // UNUSED!
  fiducialXLowVector_ =
      iConfig.getUntrackedParameter<std::vector<double>>("fiducialXLow");
  fiducialYLowVector_ =
      iConfig.getUntrackedParameter<std::vector<double>>("fiducialYLow");
  fiducialYHighVector_ =
      iConfig.getUntrackedParameter<std::vector<double>>("fiducialYHigh");
  fiducialXLow_ = {
      {std::pair<int, int>(0, 0), fiducialXLowVector_.at(0)},
      {std::pair<int, int>(0, 2), fiducialXLowVector_.at(1)},
      {std::pair<int, int>(1, 0), fiducialXLowVector_.at(2)},
      {std::pair<int, int>(1, 2), fiducialXLowVector_.at(3)},
  };
  fiducialYLow_ = {
      {std::pair<int, int>(0, 0), fiducialYLowVector_.at(0)},
      {std::pair<int, int>(0, 2), fiducialYLowVector_.at(1)},
      {std::pair<int, int>(1, 0), fiducialYLowVector_.at(2)},
      {std::pair<int, int>(1, 2), fiducialYLowVector_.at(3)},
  };
  fiducialYHigh_ = {
      {std::pair<int, int>(0, 0), fiducialYHighVector_.at(0)},
      {std::pair<int, int>(0, 2), fiducialYHighVector_.at(1)},
      {std::pair<int, int>(1, 0), fiducialYHighVector_.at(2)},
      {std::pair<int, int>(1, 2), fiducialYHighVector_.at(3)},
  };
  recoInfoCut_ = iConfig.getUntrackedParameter<int>("recoInfo");
  debug_ = iConfig.getUntrackedParameter<bool>("debug");

  // Compute binning arrays
  for (auto detID_and_coordinate : mapXbin_changeCoordinate) {
    CTPPSPixelDetId detId = detID_and_coordinate.first;
    int nBinsX_small =
        (int)((detID_and_coordinate.second - mapXmin) / mapXbinSize_small);
    mapXbin_changeCoordinate[detId] =
        mapXmin + nBinsX_small * mapXbinSize_small;
    int nBinsX_large =
        (int)((mapXmax - detID_and_coordinate.second) / mapXbinSize_large);
    nBinsX_total[detId] = nBinsX_small + nBinsX_large;
    for (int i = 0; i <= nBinsX_total[detId]; i++) {
      if (i <= nBinsX_small)
        xBinEdges[detId].push_back(i * mapXbinSize_small);
      else
        xBinEdges[detId].push_back(nBinsX_small * mapXbinSize_small +
                                   (i - nBinsX_small) * mapXbinSize_large);
    }
  }
}

InterpotEfficiency_2018::~InterpotEfficiency_2018() {
  for (auto &rpId : romanPotIdVector_) {
    delete h2AuxProtonHitDistribution_[rpId];
    delete h2AuxProtonHitDistributionWithNoMultiRP_[rpId];
    delete h2InterPotEfficiencyMap_[rpId];
    delete h2InterPotEfficiencyMapMultiRP_[rpId];
    delete h1AuxXi_[rpId];
    delete h1InterPotEfficiencyVsXi_[rpId];
    delete h1DeltaXiMatch_[rpId];
    delete h1DeltaYMatch_[rpId];
    delete h1TxMatch_[rpId];
    delete h1TyMatch_[rpId];
    delete h1ProtonsInProbePotWhenNoMatchFound_[rpId];
    delete h2TxCorrelationMatch_[rpId];
    delete h2TyCorrelationMatch_[rpId];
    delete h2XCorrelationMatch_[rpId];
    delete h2YCorrelationMatch_[rpId];
    delete h1TrackMux_[rpId];
  }
}

void InterpotEfficiency_2018::analyze(const edm::Event &iEvent,
                                      const edm::EventSetup &iSetup) {
  using namespace edm;

  Handle<reco::ForwardProtonCollection> protons;
  iEvent.getByToken(protonsToken_, protons);

  Handle<reco::ForwardProtonCollection> multiRP_protons;
  iEvent.getByToken(multiRP_protonsToken_, multiRP_protons);

  // Get LHCInfo handle
  edm::ESHandle<LHCInfo> lhcInfo;
  iSetup.get<LHCInfoRcd>().get("", lhcInfo);
  const LHCInfo *pLhcInfo = lhcInfo.product();
  double xangle = pLhcInfo->crossingAngle();

  trackMux_.clear();
  for (auto &proton_Tag : *protons) {
    if (!proton_Tag.validFit())
      continue;
    CTPPSLocalTrackLite track_Tag =
        *(proton_Tag.contributingLocalTracks().at(0));
    CTPPSDetId detId_Tag = CTPPSDetId(track_Tag.rpId());
    if (h1TrackMux_.find(detId_Tag) == h1TrackMux_.end()) {
      h1TrackMux_[detId_Tag] =
          new TH1D(Form("h1TrackMux_arm%i_st%i_rp%i", detId_Tag.arm(),
                        detId_Tag.station(), detId_Tag.rp()),
                   Form("h1TrackMux_arm%i_st%i_rp%i", detId_Tag.arm(),
                        detId_Tag.station(), detId_Tag.rp()),
                   11, 0, 11);
    }
    trackMux_[detId_Tag]++;
  }

  for (auto const &idAndHist : h1TrackMux_) {
    idAndHist.second->Fill(trackMux_[idAndHist.first]);
  }

  // Inter-Pot efficiency
  for (auto &proton_Tag : *protons) {
    if (!proton_Tag.validFit())
      continue;
    CTPPSLocalTrackLite track_Tag =
        *(proton_Tag.contributingLocalTracks().at(0));
    CTPPSDetId detId_Tag = CTPPSDetId(track_Tag.rpId());
    int arm_Tag = detId_Tag.arm();
    int station_Tag = detId_Tag.station();
    double trackX0_Tag = track_Tag.x();
    double trackY0_Tag = track_Tag.y();
    double trackTx_Tag = track_Tag.tx();
    double trackTy_Tag = track_Tag.ty();
    double xi_Tag = proton_Tag.xi();
    int matches = 0;

    if (trackMux_[detId_Tag] > maxTracksInTagPot ||
        trackMux_[detId_Tag] < minTracksInTagPot)
      continue;
    // Start only from strips
    // if (detId_Tag.station() != 0) // use as Tag only the strips RPs
    //   continue;

    // Apply aperture cut
    if (debug_)
      std::cout << "Aperture cut for arm " << arm_Tag << ": xangle = " << xangle
                << " xiMax = " << Aperture(xangle, arm_Tag, "2018")
                << std::endl;

    // if (xi_Tag > Aperture(xangle, arm_Tag, "2018"))
    // continue;

    // Apply the cuts
    if (Cut(track_Tag))
      continue;

    uint32_t arm_Probe = detId_Tag.arm();
    uint32_t station_Probe = (detId_Tag.station() == 0) ? 2 : 0;
    uint32_t rp_Probe = detId_Tag.rp();

    // CTPPSPixelDetId that the probe proton must have
    CTPPSPixelDetId pixelDetId(arm_Probe, station_Probe, rp_Probe);
    CTPPSDetId detId_Probe(pixelDetId.rawId());

    if (trackMux_[detId_Probe] > maxTracksInProbePot ||
        trackMux_[detId_Probe] < minTracksInProbePot)
      continue;

    double deltaZ = Z[detId_Probe] - Z[detId_Tag];
    // double expectedTrackX0_Probe = 0;
    // double expectedTrackY0_Probe = 0;
    double expectedTrackX0_Probe = trackX0_Tag; //+ trackTx_Tag * deltaZ;
    double expectedTrackY0_Probe = trackY0_Tag; //+ trackTy_Tag * deltaZ;
    int protonsInProbePot = 0;

    // Booking histograms
    if (h2InterPotEfficiencyMap_.find(pixelDetId) ==
        h2InterPotEfficiencyMap_.end()) {

      romanPotIdVector_.push_back(pixelDetId);
      if (station_Probe == 2) {
        mapXbins = mapXbins_st2;
        mapXmin = mapXmin_st2;
        mapXmax = mapXmax_st2;
        mapYbins = mapYbins_st2;
        mapYmin = mapYmin_st2;
        mapYmax = mapYmax_st2;
      } else {
        mapXbins = mapXbins_st0;
        mapXmin = mapXmin_st0;
        mapXmax = mapXmax_st0;
        mapYbins = mapYbins_st0;
        mapYmin = mapYmin_st0;
        mapYmax = mapYmax_st0;
      }

      if (fancyBinning_) {
        h2AuxProtonHitDistribution_[pixelDetId] = new TH2D(
            Form("h2ProtonHitExpectedDistribution_arm%i_st%i_rp%i", arm_Probe,
                 station_Probe, rp_Probe),
            Form(
                "h2ProtonHitExpectedDistribution_arm%i_st%i_rp%i;x (mm);y (mm)",
                arm_Probe, station_Probe, rp_Probe),
            nBinsX_total[pixelDetId], &xBinEdges[pixelDetId][0], mapYbins,
            mapYmin, mapYmax);
        h2AuxProtonHitDistributionWithNoMultiRP_[pixelDetId] = new TH2D(
            Form("h2ProtonHitExpectedDistributionWithNoMultiRP_arm%i_st%i_rp%i",
                 arm_Probe, station_Probe, rp_Probe),
            Form("h2ProtonHitExpectedDistributionWithNoMultiRP_arm%i_st%i_rp%i;"
                 "x (mm);y (mm)",
                 arm_Probe, station_Probe, rp_Probe),
            nBinsX_total[pixelDetId], &xBinEdges[pixelDetId][0], mapYbins,
            mapYmin, mapYmax);
        h2InterPotEfficiencyMap_[pixelDetId] = new TH2D(
            Form("h2InterPotEfficiencyMap_arm%i_st%i_rp%i", arm_Probe,
                 station_Probe, rp_Probe),
            Form("h2InterPotEfficiencyMap_arm%i_st%i_rp%i;x (mm);y (mm)",
                 arm_Probe, station_Probe, rp_Probe),
            nBinsX_total[pixelDetId], &xBinEdges[pixelDetId][0], mapYbins,
            mapYmin, mapYmax);
        h2InterPotEfficiencyMapMultiRP_[pixelDetId] = new TH2D(
            Form("h2InterPotEfficiencyMapMultiRP_arm%i_st%i_rp%i", arm_Probe,
                 station_Probe, rp_Probe),
            Form("h2InterPotEfficiencyMapMultiRP_arm%i_st%i_rp%i;x (mm);y (mm)",
                 arm_Probe, station_Probe, rp_Probe),
            nBinsX_total[pixelDetId], &xBinEdges[pixelDetId][0], mapYbins,
            mapYmin, mapYmax);
      } else {
        h2AuxProtonHitDistribution_[pixelDetId] = new TH2D(
            Form("h2ProtonHitExpectedDistribution_arm%i_st%i_rp%i", arm_Probe,
                 station_Probe, rp_Probe),
            Form(
                "h2ProtonHitExpectedDistribution_arm%i_st%i_rp%i;x (mm);y (mm)",
                arm_Probe, station_Probe, rp_Probe),
            mapXbins, mapXmin, mapXmax, mapYbins, mapYmin, mapYmax);
        h2AuxProtonHitDistributionWithNoMultiRP_[pixelDetId] = new TH2D(
            Form("h2ProtonHitExpectedDistributionWithNoMultiRP_arm%i_st%i_rp%i",
                 arm_Probe, station_Probe, rp_Probe),
            Form("h2ProtonHitExpectedDistributionWithNoMultiRP_arm%i_st%i_rp%i;"
                 "x (mm);y (mm)",
                 arm_Probe, station_Probe, rp_Probe),
            mapXbins, mapXmin, mapXmax, mapYbins, mapYmin, mapYmax);
        h2InterPotEfficiencyMap_[pixelDetId] = new TH2D(
            Form("h2InterPotEfficiencyMap_arm%i_st%i_rp%i", arm_Probe,
                 station_Probe, rp_Probe),
            Form("h2InterPotEfficiencyMap_arm%i_st%i_rp%i;x (mm);y (mm)",
                 arm_Probe, station_Probe, rp_Probe),
            mapXbins, mapXmin, mapXmax, mapYbins, mapYmin, mapYmax);
        h2InterPotEfficiencyMapMultiRP_[pixelDetId] = new TH2D(
            Form("h2InterPotEfficiencyMapMultiRP_arm%i_st%i_rp%i", arm_Probe,
                 station_Probe, rp_Probe),
            Form("h2InterPotEfficiencyMapMultiRP_arm%i_st%i_rp%i;x (mm);y (mm)",
                 arm_Probe, station_Probe, rp_Probe),
            mapXbins, mapXmin, mapXmax, mapYbins, mapYmin, mapYmax);
      }
      h1InterPotEfficiencyVsXi_[pixelDetId] = new TH1D(
          Form("h1InterPotEfficiencyVsXi_arm%i_st%i_rp%i", arm_Probe,
               station_Probe, rp_Probe),
          Form("h1InterPotEfficiencyVsXi_arm%i_st%i_rp%i;#xi;Efficiency",
               arm_Probe, station_Probe, rp_Probe),
          xiBins, xiMin, xiMax);
      h1AuxXi_[pixelDetId] = new TH1D(
          Form("h1AuxXi_arm%i_st%i_rp%i", arm_Probe, station_Probe, rp_Probe),
          Form("h1AuxXi_arm%i_st%i_rp%i;#xi;Efficiency", arm_Probe,
               station_Probe, rp_Probe),
          xiBins, xiMin, xiMax);
      h1DeltaXiMatch_[pixelDetId] =
          new TH1D(Form("h1DeltaXiMatch_arm%i_st%i_rp%i", arm_Probe,
                        station_Probe, rp_Probe),
                   Form("h1DeltaXiMatch_arm%i_st%i_rp%i;#Delta_{#xi}",
                        arm_Probe, station_Probe, rp_Probe),
                   100, -0.02, 0.02);
      h1DeltaYMatch_[pixelDetId] =
          new TH1D(Form("h1DeltaYMatch_arm%i_st%i_rp%i", arm_Probe,
                        station_Probe, rp_Probe),
                   Form("h1DeltaYMatch_arm%i_st%i_rp%i;#Delta_{#xi}", arm_Probe,
                        station_Probe, rp_Probe),
                   100, -5, 5);
      h1TxMatch_[pixelDetId] = new TH1D(
          Form("h1TxMatch_arm%i_st%i_rp%i", arm_Probe, station_Probe, rp_Probe),
          Form("h1TxMatch_%i_st%i_rp%i;Tx", arm_Probe, station_Probe, rp_Probe),
          100, -0.02, 0.02);
      h1TyMatch_[pixelDetId] = new TH1D(
          Form("h1TyMatch_arm%i_st%i_rp%i", arm_Probe, station_Probe, rp_Probe),
          Form("h1TyMatch_%i_st%i_rp%i;Ty", arm_Probe, station_Probe, rp_Probe),
          100, -0.02, 0.02);
      h1ProtonsInProbePotWhenNoMatchFound_[pixelDetId] =
          new TH1D(Form("h1ProtonsInProbePotWhenNoMatchFound_arm%i_st%i_rp%i",
                        arm_Probe, station_Probe, rp_Probe),
                   Form("h1ProtonsInProbePotWhenNoMatchFound_arm%i_st%i_rp%i",
                        arm_Probe, station_Probe, rp_Probe),
                   11, 0, 11);
      h2XCorrelationMatch_[pixelDetId] =
          new TH2D(Form("h2XCorrelationMatch_arm%i_st%i_rp%i", arm_Probe,
                        station_Probe, rp_Probe),
                   Form("h2XCorrelationMatch_arm%i_st%i_rp%i;x pixel (mm);x "
                        "strips (mm)",
                        arm_Probe, station_Probe, rp_Probe),
                   mapXbins, mapXmin, mapXmax, mapXbins, mapXmin, mapXmax);
      h2YCorrelationMatch_[pixelDetId] = new TH2D(
          Form("h2YCorrelationMatch_arm%i_st%i_rp%i", arm_Probe, station_Probe,
               rp_Probe),
          Form("h2YCorrelationMatch_arm%i_st%i_rp%i;y pixel (mm);y strips (mm)",
               arm_Probe, station_Probe, rp_Probe),
          mapYbins, mapYmin, mapYmax, mapYbins, mapYmin, mapYmax);
      h2TxCorrelationMatch_[pixelDetId] =
          new TH2D(Form("h2TxCorrelationMatch_arm%i_st%i_rp%i", arm_Probe,
                        station_Probe, rp_Probe),
                   Form("h2TxCorrelationMatch_arm%i_st%i_rp%i;Tx pixel (mm);Ty "
                        "pixel (mm)",
                        arm_Probe, station_Probe, rp_Probe),
                   100, -0.01, 0.01, 100, -0.01, 0.01);
      h2TyCorrelationMatch_[pixelDetId] =
          new TH2D(Form("h2TyCorrelationMatch_arm%i_st%i_rp%i", arm_Probe,
                        station_Probe, rp_Probe),
                   Form("h2TyCorrelationMatch_arm%i_st%i_rp%i;Tx pixel (mm);Ty "
                        "pixel (mm)",
                        arm_Probe, station_Probe, rp_Probe),
                   100, -0.01, 0.01, 100, -0.01, 0.01);
    }

    for (auto &proton_Probe : *protons) { // Probe -> Roman Pot Under Test
      if (!proton_Probe.validFit())
        continue;
      CTPPSLocalTrackLite track_Probe =
          *(proton_Probe.contributingLocalTracks().at(0));
      // CTPPSDetId detId_Probe = CTPPSDetId(track_Probe.rpId());
      double trackX0_Probe = track_Probe.x();
      double trackY0_Probe = track_Probe.y();
      double trackTx_Probe = track_Probe.tx();
      double trackTy_Probe = track_Probe.ty();
      double xi_Probe = proton_Probe.xi();
      // Require the proton_Probe to be in the same arm, different station
      // This means that the CTPPSPixelDetId is the same as above
      // if (detId_Tag.station() == detId_Probe.station() ||
      //     detId_Tag.arm() != detId_Probe.arm())

      if (detId_Probe != track_Probe.rpId())
        continue;
      protonsInProbePot++;

      // Apply the cuts
      if (Cut(track_Probe))
        continue;

      bool xiMatchPass = false;
      bool yMatchPass = false;
      bool xMatchPass = false;
      
      //NEAR - Tag, FAR - Probe 
      
      // Make it so that the difference is always NEAR - FAR
      double xiDiff =
          (station_Tag == 0) ? xi_Tag - xi_Probe : xi_Probe - xi_Tag;
      double xDiff = (station_Tag == 0) ? trackX0_Tag - trackX0_Probe
                                        : trackX0_Probe - trackX0_Tag;
      double yDiff = (station_Tag == 0) ? trackY0_Tag - trackY0_Probe
                                        : trackY0_Probe - trackY0_Tag;
      if (arm_Tag == 0 && TMath::Abs(xiDiff - xiMatchMean45) < xiMatchWindow45)
        xiMatchPass = true;
      if (arm_Tag == 0 && TMath::Abs(yDiff - yMatchMean45) < yMatchWindow45)
        yMatchPass = true;
      if (arm_Tag == 0 && TMath::Abs(xDiff - xMatchMean45) < xMatchWindow45)
        xMatchPass = true;
      if (arm_Tag == 1 && TMath::Abs(xiDiff - xiMatchMean56) < xiMatchWindow56)
        xiMatchPass = true;
      if (arm_Tag == 1 && TMath::Abs(yDiff - yMatchMean56) < yMatchWindow56)
        yMatchPass = true;
      if (arm_Tag == 1 && TMath::Abs(xDiff - xMatchMean56) < xMatchWindow56)
        xMatchPass = true;

      h1DeltaXiMatch_[pixelDetId]->Fill(xi_Tag - xi_Probe);

      if (xiMatchPass) {
        h1DeltaYMatch_[pixelDetId]->Fill(trackY0_Tag - trackY0_Probe);
        if (xMatchPass && yMatchPass) {
          matches++;
          if (debug_) {
            std::cout << "********MATCH FOUND********" << std::endl;
            std::cout << "Tag track:\n"
                      << "Arm: " << detId_Tag.arm()
                      << " Station: " << detId_Tag.station()
                      << " X: " << trackX0_Tag << " Y: " << trackY0_Tag
                      << " Tx: " << trackTx_Tag << " Ty: " << trackTy_Tag
                      << " Xi: " << xi_Tag << std::endl;
            std::cout << "Probe track:\n"
                      << "Arm: " << detId_Probe.arm()
                      << " Station: " << detId_Probe.station()
                      << " X: " << trackX0_Probe << " Y: " << trackY0_Probe
                      << " Tx: " << trackTx_Probe << " Ty: " << trackTy_Probe
                      << " Xi: " << xi_Probe << "\nDeltaZ: " << deltaZ
                      << " Expected X: " << expectedTrackX0_Probe
                      << " Expected Y: " << expectedTrackY0_Probe
                      << " RecoInfo: "
                      << (int)track_Probe.pixelTrackRecoInfo() << std::endl;
            std::cout << "**************************" << std::endl;
          }
          if (matches == 1) {
            h2InterPotEfficiencyMap_[pixelDetId]->Fill(expectedTrackX0_Probe,
                                                       expectedTrackY0_Probe);
            h1InterPotEfficiencyVsXi_[pixelDetId]->Fill(
                xi_Tag); // xi_Tag and xi_Probe are expected to be the same
            h1TxMatch_[pixelDetId]->Fill(trackTx_Tag);
            h1TyMatch_[pixelDetId]->Fill(trackTy_Tag);
            h2XCorrelationMatch_[pixelDetId]->Fill(trackX0_Probe, trackX0_Tag);
            h2YCorrelationMatch_[pixelDetId]->Fill(trackY0_Probe, trackY0_Tag);
            h2TxCorrelationMatch_[pixelDetId]->Fill(trackTx_Probe, trackTx_Tag);
            h2TyCorrelationMatch_[pixelDetId]->Fill(trackTy_Probe, trackTy_Tag);
          }
          if (excludeMultipleMatches && matches == 2) {
            h2InterPotEfficiencyMap_[pixelDetId]->Fill(
                expectedTrackX0_Probe, expectedTrackY0_Probe, -1);
            h1InterPotEfficiencyVsXi_[pixelDetId]->Fill(
                xi_Tag, -1); // xi_Tag and xi_Probe are expected to be the same
            h1TxMatch_[pixelDetId]->Fill(trackTx_Tag, -1);
            h1TyMatch_[pixelDetId]->Fill(trackTy_Tag, -1);
            h2XCorrelationMatch_[pixelDetId]->Fill(trackX0_Probe, trackX0_Tag,
                                                   -1);
            h2YCorrelationMatch_[pixelDetId]->Fill(trackY0_Probe, trackY0_Tag,
                                                   -1);
            h2TxCorrelationMatch_[pixelDetId]->Fill(trackTx_Probe, trackTx_Tag,
                                                    -1);
            h2TyCorrelationMatch_[pixelDetId]->Fill(trackTy_Probe, trackTy_Tag,
                                                    -1);
          }
        }
      }
    }

    // MultiRP efficiency
    uint32_t multiRPmatchFound = 0;
    for (auto &multiRP_proton : *multiRP_protons) {
      if (!multiRP_proton.validFit() ||
          multiRP_proton.method() !=
              reco::ForwardProton::ReconstructionMethod::multiRP) {
        if (debug_)
          std::cout << "Found INVALID multiRP proton!" << std::endl;
        continue;
      }

      if (debug_) {
        std::cout << "***Analyzing multiRP proton***" << std::endl;
        std::cout << "Xi = " << multiRP_proton.xi() << std::endl;
        std::cout << "Composed by track: " << std::endl;
      }

      for (auto &track_ptr : multiRP_proton.contributingLocalTracks()) {
        CTPPSLocalTrackLite track = *track_ptr;
        CTPPSDetId detId = CTPPSDetId(track.rpId());
        int arm = detId.arm();
        int station = detId.station();
        double trackX0 = track.x();
        double trackY0 = track.y();
        double trackTx = track.tx();
        double trackTy = track.ty();

        if (debug_) {
          std::cout << "Arm: " << arm << " Station: " << station << std::endl
                    << " X: " << trackX0 << " Y: " << trackY0
                    << " Tx: " << trackTx << " Ty: " << trackTy
                    << " recoInfo: " << (int)track.pixelTrackRecoInfo()
                    << std::endl;
        }

        if (arm == arm_Tag && station == station_Tag && station != 1 &&
            TMath::Abs(trackX0_Tag - trackX0) < 0.01 &&
            TMath::Abs(trackY0_Tag - trackY0) < 0.01) {
          if (debug_)
            std::cout << "**MultiRP proton matched to the tag track!**"
                      << std::endl;
          multiRPmatchFound++;
          h2InterPotEfficiencyMapMultiRP_[pixelDetId]->Fill(
              expectedTrackX0_Probe, expectedTrackY0_Probe);
        }
      }
    }
    if (multiRPmatchFound > 1) {
      std::cout << "WARNING: More than one multiRP matched!" << std::endl;
    }

    h2AuxProtonHitDistribution_[pixelDetId]->Fill(expectedTrackX0_Probe,
                                                  expectedTrackY0_Probe);
    if (multiRPmatchFound == 0)
      h2AuxProtonHitDistributionWithNoMultiRP_[pixelDetId]->Fill(
          expectedTrackX0_Probe, expectedTrackY0_Probe);
    h1AuxXi_[pixelDetId]->Fill(xi_Tag);

    if (matches > 1) {
      overmatches[pixelDetId]++;
      if (debug_)
        std::cout << "***WARNING: Overmatching!***" << std::endl;
    }
    tries[pixelDetId]++;

    bool goodInterPotMatch =
        (excludeMultipleMatches) ? matches == 1 : matches >= 1;

    if (!goodInterPotMatch) {
      h1ProtonsInProbePotWhenNoMatchFound_[pixelDetId]->Fill(protonsInProbePot);
    }
  }

  if (debug_)
    std::cout << std::endl;
}

void InterpotEfficiency_2018::fillDescriptions(
    edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

void InterpotEfficiency_2018::beginJob() {}

void InterpotEfficiency_2018::endJob() {
  TFile *outputFile_ = new TFile(outputFileName_.data(), "RECREATE");
  for (auto &rpId : romanPotIdVector_) {
    uint32_t arm = rpId.arm();
    uint32_t station = rpId.station();
    std::string rpDirName = Form("Arm%i_st%i_rp3", arm, station);
    outputFile_->mkdir(rpDirName.data());
    outputFile_->cd(rpDirName.data());

    if (h1InterPotEfficiencyVsXi_.find(rpId) !=
        h1InterPotEfficiencyVsXi_.end()) {
      h1TrackMux_[rpId]->Write();
      h1InterPotEfficiencyVsXi_[rpId]->Divide(h1InterPotEfficiencyVsXi_[rpId],
                                              h1AuxXi_[rpId], 1., 1.);
      h1InterPotEfficiencyVsXi_[rpId]->SetMaximum(1.1);
      h1InterPotEfficiencyVsXi_[rpId]->SetMinimum(0);

      h1InterPotEfficiencyVsXi_[rpId]->Write();
      h1AuxXi_[rpId]->Write();
      h1DeltaXiMatch_[rpId]->Write();
      h1DeltaYMatch_[rpId]->Write();
      h1ProtonsInProbePotWhenNoMatchFound_[rpId]->Write();

      h2InterPotEfficiencyMap_[rpId]->Divide(h2InterPotEfficiencyMap_[rpId],
                                             h2AuxProtonHitDistribution_[rpId],
                                             1., 1., "B");
      for (auto i = 1; i < h2InterPotEfficiencyMap_[rpId]->GetNbinsX(); i++) {
        for (auto j = 1; j < h2InterPotEfficiencyMap_[rpId]->GetNbinsY(); j++) {
          double efficiency =
              h2InterPotEfficiencyMap_[rpId]->GetBinContent(i, j);
          double tries = h2AuxProtonHitDistribution_[rpId]->GetBinContent(i, j);
          if (tries != 0) {
            double error = TMath::Sqrt((efficiency * (1 - efficiency)) / tries);
            h2InterPotEfficiencyMap_[rpId]->SetBinError(i, j, error);
          } else
            h2InterPotEfficiencyMap_[rpId]->SetBinError(i, j, 0);
        }
      }
      h2InterPotEfficiencyMap_[rpId]->SetMinimum(0);
      h2InterPotEfficiencyMap_[rpId]->SetMaximum(1);
      h2InterPotEfficiencyMap_[rpId]->Write();

      TEfficiency TEInterPotEfficiencyMapMultiRP =
          TEfficiency(*h2InterPotEfficiencyMapMultiRP_[rpId],
                      *h2AuxProtonHitDistribution_[rpId]);
      TEInterPotEfficiencyMapMultiRP.SetNameTitle(
          Form("TEInterPotEfficiencyMapMultiRP_arm%i_st%i_rp%i", arm, station,
               3),
          Form("TEInterPotEfficiencyMapMultiRP_arm%i_st%i_rp%i", arm, station,
               3));
      TEInterPotEfficiencyMapMultiRP.Write();

      h2InterPotEfficiencyMapMultiRP_[rpId]->Divide(
          h2InterPotEfficiencyMapMultiRP_[rpId],
          h2AuxProtonHitDistribution_[rpId], 1., 1., "B");
      h2InterPotEfficiencyMapMultiRP_[rpId]->SetMinimum(0);
      h2InterPotEfficiencyMapMultiRP_[rpId]->SetMaximum(1);
      for (auto i = 1; i < h2InterPotEfficiencyMapMultiRP_[rpId]->GetNbinsX();
           i++) {
        for (auto j = 1; j < h2InterPotEfficiencyMapMultiRP_[rpId]->GetNbinsY();
             j++) {
          double efficiency =
              h2InterPotEfficiencyMapMultiRP_[rpId]->GetBinContent(i, j);
          double tries = h2AuxProtonHitDistribution_[rpId]->GetBinContent(i, j);
          if (tries != 0) {
            double error = TMath::Sqrt((efficiency * (1 - efficiency)) / tries);
            h2InterPotEfficiencyMapMultiRP_[rpId]->SetBinError(i, j, error);
          } else
            h2InterPotEfficiencyMapMultiRP_[rpId]->SetBinError(i, j, 0);
        }
      }
      h2InterPotEfficiencyMapMultiRP_[rpId]->Write();
      h2AuxProtonHitDistribution_[rpId]->Write();
      h2AuxProtonHitDistributionWithNoMultiRP_[rpId]->Write(); 
      h1TxMatch_[rpId]->Write();
      h1TyMatch_[rpId]->Write();
      h2XCorrelationMatch_[rpId]->Write();
      h2YCorrelationMatch_[rpId]->Write();
      h2TxCorrelationMatch_[rpId]->Write();
      h2TyCorrelationMatch_[rpId]->Write();
    }
  }
  outputFile_->Close();
  delete outputFile_;

  for (auto pixelDetId : romanPotIdVector_) {
    std::cout << "*** Arm: " << pixelDetId.arm()
              << " Station: " << pixelDetId.station() << std::endl;
    std::cout << "The overmatch rate is: "
              << ((double)overmatches[pixelDetId] / tries[pixelDetId]) * 100
              << " %" << std::endl;
  }
}

bool InterpotEfficiency_2018::Cut(CTPPSLocalTrackLite track) {
  CTPPSDetId detId = CTPPSDetId(track.rpId());
  uint32_t arm = detId.arm();
  uint32_t station = detId.station();
  uint32_t ndf = 2 * track.numberOfPointsUsedForFit() - 4;
  double x = track.x();
  double y = track.y();
  float pixelX0_rotated = 0;
  float pixelY0_rotated = 0;
  if (station == 0) {
    pixelX0_rotated = x * TMath::Cos((-8. / 180.) * TMath::Pi()) -
                      y * TMath::Sin((-8. / 180.) * TMath::Pi());
    pixelY0_rotated = x * TMath::Sin((-8. / 180.) * TMath::Pi()) +
                      y * TMath::Cos((-8. / 180.) * TMath::Pi());
    x = pixelX0_rotated;
    y = pixelY0_rotated;
  }

  double maxTx = 0.02;
  double maxTy = 0.02;
  double maxChi2 = TMath::ChisquareQuantile(maxChi2Prob_, ndf);

  if (debug_) {
    if (track.chiSquaredOverNDF() * ndf > maxChi2)
      std::cout << "Chi2 cut not passed" << std::endl;
    if (TMath::Abs(track.tx()) > maxTx)
      std::cout << "maxTx cut not passed" << std::endl;
    if (TMath::Abs(track.ty()) > maxTy)
      std::cout << "maxTy cut not passed" << std::endl;
    if (track.numberOfPointsUsedForFit() < minNumberOfPlanesForTrack_)
      std::cout << "Too few planes for track" << std::endl;
    if (track.numberOfPointsUsedForFit() > maxNumberOfPlanesForTrack_)
      std::cout << "Too many planes for track" << std::endl;
    if (y > fiducialYHigh_[std::pair<int, int>(arm, station)])
      std::cout << "fiducialYHigh cut not passed" << std::endl;
    if (y < fiducialYLow_[std::pair<int, int>(arm, station)])
      std::cout << "fiducialYLow cut not passed" << std::endl;
    if (x < fiducialXLow_[std::pair<int, int>(arm, station)])
      std::cout << "fiducialXLow cut not passed" << std::endl;
  }
  if (TMath::Abs(track.tx()) > maxTx || TMath::Abs(track.ty()) > maxTy ||
      track.chiSquaredOverNDF() * ndf > maxChi2 ||
      track.numberOfPointsUsedForFit() < minNumberOfPlanesForTrack_ ||
      track.numberOfPointsUsedForFit() > maxNumberOfPlanesForTrack_ ||
      y > fiducialYHigh_[std::pair<int, int>(arm, station)] ||
      y < fiducialYLow_[std::pair<int, int>(arm, station)] ||
      x < fiducialXLow_[std::pair<int, int>(arm, station)])
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
}

float Aperture(Float_t xangle, Int_t arm, TString era) {
  float aperturelimit = 0.0;
  if (era == "2016preTS2") {
    if (arm == 0)
      aperturelimit = 0.111;
    if (arm == 1)
      aperturelimit = 0.138;
  }
  if (era == "2016postTS2") {
    if (arm == 0)
      aperturelimit = 0.104;
    if (arm == 1) // Note - 1 strip RP was not in, so no aperture cuts derived
      aperturelimit = 999.9;
  }
  if (era == "2017preTS2") {
    if (arm == 0)
      aperturelimit = 0.066 + (3.54E-4 * xangle);
    if (arm == 1)
      aperturelimit = 0.062 + (5.96E-4 * xangle);
  }
  if (era == "2017postTS2") {
    if (arm == 0)
      aperturelimit = 0.073 + (4.11E-4 * xangle);
    if (arm == 1)
      aperturelimit = 0.067 + (6.87E-4 * xangle);
  }
  if (era == "2018") {
    if (arm == 0)
      aperturelimit = 0.079 + (4.21E-4 * xangle);
    if (arm == 1)
      aperturelimit = 0.074 + (6.6E-4 * xangle);
  }

  return aperturelimit;
}

// define this as a plug-in
DEFINE_FWK_MODULE(InterpotEfficiency_2018);