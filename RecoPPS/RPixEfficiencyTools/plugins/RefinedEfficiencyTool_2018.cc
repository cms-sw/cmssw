// -*- C++ -*-
//
// Package:    RecoPPS/RPixEfficiencyTools
// Class:      RefinedEfficiencyTool_2018
//
/**\class RefinedEfficiencyTool_2018 RefinedEfficiencyTool_2018.cc
 RecoPPS/RPixEfficiencyTools/plugins/RefinedEfficiencyTool_2018.cc

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
#include "DataFormats/CTPPSReco/interface/CTPPSPixelLocalTrack.h"
#include "DataFormats/CTPPSReco/interface/CTPPSPixelRecHit.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <TEfficiency.h>
#include <TF1.h>
#include <TFile.h>
#include <TGraphErrors.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TMath.h>
#include <TObjArray.h>

class RefinedEfficiencyTool_2018
    : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit RefinedEfficiencyTool_2018(const edm::ParameterSet &);
  ~RefinedEfficiencyTool_2018();
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  virtual void beginJob() override;
  virtual void analyze(const edm::Event &, const edm::EventSetup &) override;
  virtual void endJob() override;
  float
  probabilityNplanesBlind(const std::vector<uint32_t> &inputPlaneList,
                          int numberToExtract,
                          const std::map<unsigned, float> &planeEfficiency);
  void getPlaneCombinations(
      const std::vector<uint32_t> &inputPlaneList, uint32_t numberToExtract,
      std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>>
          &planesExtractedAndNot);
  float
  probabilityCalculation(const std::map<uint32_t, float> &planeEfficiency);
  bool Cut(CTPPSPixelLocalTrack track, int arm, int station);

  bool debug_ = false;

  // Data to get
  edm::EDGetTokenT<edm::DetSetVector<CTPPSPixelLocalTrack>>
      pixelLocalTrackToken_;

  // Parameter set
  TFile *outputFile_;
  std::string efficiencyFileName_;
  std::string outputFileName_;
  int minNumberOfPlanesForEfficiency_;
  int minNumberOfPlanesForTrack_;
  int maxNumberOfPlanesForTrack_ = 6;
  int minTracksPerEvent;
  int maxTracksPerEvent;
  std::string producerTag;

  // Configs
  std::vector<uint32_t> listOfArms_ = {0, 1};
  std::vector<uint32_t> listOfStations_ = {0, 2};
  std::vector<uint32_t> listOfPlanes_ = {0, 1, 2, 3, 4, 5};

  std::vector<CTPPSPixelDetId> detectorIdVector_;
  std::vector<CTPPSPixelDetId> romanPotIdVector_;

  int binGroupingX = 1;
  int binGroupingY = 1;

  int mapXbins_st2 = 200 / binGroupingX;
  float mapXmin_st2 = 0. * TMath::Cos(18.4 / 180. * TMath::Pi());
  float mapXmax_st2 = 30. * TMath::Cos(18.4 / 180. * TMath::Pi());
  int mapYbins_st2 = 240 / binGroupingY;
  float mapYmin_st2 = -16.;
  float mapYmax_st2 = 8.;

  int mapXbins_st0 = 200 / binGroupingX;
  float mapXmin_st0 = 0. * TMath::Cos(18.4 / 180. * TMath::Pi());
  float mapXmax_st0 = 30. * TMath::Cos(18.4 / 180. * TMath::Pi());
  int mapYbins_st0 = 240 / binGroupingY;
  float mapYmin_st0 = -16.;
  float mapYmax_st0 = 8.;

  int mapXbins = mapXbins_st0;
  float mapXmin = mapXmin_st0;
  float mapXmax = mapXmax_st0;
  int mapYbins = mapYbins_st0;
  float mapYmin = mapYmin_st0;
  float mapYmax = mapYmax_st0;

  std::map<CTPPSPixelDetId, int> binAlignmentParameters = {
      {CTPPSPixelDetId(0, 0, 3), 0},
      {CTPPSPixelDetId(0, 2, 3), 0},
      {CTPPSPixelDetId(1, 0, 3), 0},
      {CTPPSPixelDetId(1, 2, 3), 0}};

  // output histograms
  std::map<CTPPSPixelDetId, TH2D *> h2PlaneEfficiencyMap_;
  std::map<CTPPSPixelDetId, TH2D *> h2RefinedTrackEfficiency_;
  std::map<CTPPSPixelDetId, TH2D *> h2TrackHitDistribution_;
  std::map<CTPPSPixelDetId, TH2D *> h2RefinedTrackEfficiency_rotated;
  std::map<CTPPSPixelDetId, TH2D *> h2TrackHitDistribution_rotated;

  // file to insert the output hists in
  TFile *efficiencyFile_;

  std::vector<double> fiducialXLowVector_;
  std::vector<double> fiducialYLowVector_;
  std::vector<double> fiducialYHighVector_;
  std::map<std::pair<int, int>, double> fiducialXLow_;
  std::map<std::pair<int, int>, double> fiducialYLow_;
  std::map<std::pair<int, int>, double> fiducialYHigh_;
};

RefinedEfficiencyTool_2018::RefinedEfficiencyTool_2018(
    const edm::ParameterSet &iConfig) {
  usesResource("TFileService");

  producerTag = iConfig.getUntrackedParameter<std::string>("producerTag");

  pixelLocalTrackToken_ = consumes<edm::DetSetVector<CTPPSPixelLocalTrack>>(
      edm::InputTag("ctppsPixelLocalTracks", "", producerTag));
  efficiencyFileName_ =
      iConfig.getUntrackedParameter<std::string>("efficiencyFileName");
outputFileName_ =
      iConfig.getUntrackedParameter<std::string>("outputFileName");
  minNumberOfPlanesForEfficiency_ =
      iConfig.getParameter<int>("minNumberOfPlanesForEfficiency");
  minNumberOfPlanesForTrack_ =
      iConfig.getParameter<int>("minNumberOfPlanesForTrack");
  minTracksPerEvent = iConfig.getParameter<int>("minTracksPerEvent");
  maxTracksPerEvent = iConfig.getParameter<int>("maxTracksPerEvent");
  binGroupingX = iConfig.getUntrackedParameter<int>("binGroupingX");
  binGroupingY = iConfig.getUntrackedParameter<int>("binGroupingY");
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
}

RefinedEfficiencyTool_2018::~RefinedEfficiencyTool_2018() {
  for (auto &rpId : romanPotIdVector_) {
    delete h2RefinedTrackEfficiency_[rpId];
    delete h2TrackHitDistribution_[rpId];
    if (rpId.station() == 0) {
      delete h2RefinedTrackEfficiency_rotated[rpId];
      delete h2TrackHitDistribution_rotated[rpId];
    }
  }

  for (auto &planeId : detectorIdVector_) {
    delete h2PlaneEfficiencyMap_[planeId];
  }
}

void RefinedEfficiencyTool_2018::analyze(const edm::Event &iEvent,
                                         const edm::EventSetup &iSetup) {
  using namespace edm;

  Handle<edm::DetSetVector<CTPPSPixelLocalTrack>> pixelLocalTracks;
  iEvent.getByToken(pixelLocalTrackToken_, pixelLocalTracks);

  for (const auto &rpPixeltrack : *pixelLocalTracks) {
    if ((uint32_t)minTracksPerEvent > rpPixeltrack.size() ||
        rpPixeltrack.size() > (uint32_t)maxTracksPerEvent)
      continue;
    CTPPSPixelDetId rpId = CTPPSPixelDetId(rpPixeltrack.detId());
    uint32_t arm = rpId.arm();
    uint32_t station = rpId.station();
    uint32_t rp = rpId.rp();

    if (station == 2) {
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

    // Shift Xmin and Xmax to align bins with sensor edge
    double binSize = (mapXmax - mapXmin) / mapXbins;
    mapXmin += binAlignmentParameters[rpId] * binSize / 150.;
    mapXmax += binAlignmentParameters[rpId] * binSize / 150.;

    if (h2TrackHitDistribution_.find(rpId) == h2TrackHitDistribution_.end()) {
      romanPotIdVector_.push_back(rpId);
      h2TrackHitDistribution_[rpId] = new TH2D(
          Form("h2TrackHitDistribution_arm%i_st%i_rp%i", arm, station, rp),
          Form("h2TrackHitDistribution_arm%i_st%i_rp%i;x (mm);y (mm)", arm,
               station, rp),
          mapXbins, mapXmin, mapXmax, mapYbins, mapYmin, mapYmax);
      h2RefinedTrackEfficiency_[rpId] = new TH2D(
          Form("h2RefinedTrackEfficiency_arm%i_st%i_rp%i", arm, station, rp),
          Form("h2RefinedTrackEfficiency_arm%i_st%i_rp%i;x (mm);y (mm)", arm,
               station, rp),
          mapXbins, mapXmin, mapXmax, mapYbins, mapYmin, mapYmax);
      if (station == 0) {
        h2TrackHitDistribution_rotated[rpId] = new TH2D(
            Form("h2TrackHitDistribution_rotated_arm%i_st%i_rp%i", arm, station,
                 rp),
            Form("h2TrackHitDistribution_rotated_arm%i_st%i_rp%i;x (mm);y (mm)",
                 arm, station, rp),
            mapXbins, mapXmin, mapXmax, mapYbins, mapYmin, mapYmax);
        h2RefinedTrackEfficiency_rotated[rpId] =
            new TH2D(Form("h2RefinedTrackEfficiency_rotated_arm%i_st%i_rp%i",
                          arm, station, rp),
                     Form("h2RefinedTrackEfficiency_rotated_arm%i_st%i_rp%i;x "
                          "(mm);y (mm)",
                          arm, station, rp),
                     mapXbins, mapXmin, mapXmax, mapYbins, mapYmin, mapYmax);
      }
    }
    for (const auto &pixeltrack : rpPixeltrack) {

      if (Cut(pixeltrack, arm, station) || !pixeltrack.isValid())
        continue;
      float pixelX0 = pixeltrack.x0();
      float pixelY0 = pixeltrack.y0();
      // Rotating St0 tracks
      float pixelX0_rotated = 0;
      float pixelY0_rotated = 0;
      if (station == 0) {
        pixelX0_rotated =
            pixeltrack.x0() * TMath::Cos((-8. / 180.) * TMath::Pi()) -
            pixeltrack.y0() * TMath::Sin((-8. / 180.) * TMath::Pi());
        pixelY0_rotated =
            pixeltrack.x0() * TMath::Sin((-8. / 180.) * TMath::Pi()) +
            pixeltrack.y0() * TMath::Cos((-8. / 180.) * TMath::Pi());
      }

      edm::DetSetVector<CTPPSPixelFittedRecHit> fittedHits =
          pixeltrack.hits();

      h2TrackHitDistribution_[rpId]->Fill(pixelX0, pixelY0);
      if (station == 0)
        h2TrackHitDistribution_rotated[rpId]->Fill(pixelX0_rotated,
                                                   pixelY0_rotated);
      std::map<uint32_t, float> planeEfficiency;
      for (const auto &planeHits : fittedHits) {
        CTPPSPixelDetId planeId = CTPPSPixelDetId(planeHits.detId());
        uint32_t plane = planeId.plane();
        for (const auto &hit : planeHits) {
          double hitX0 = hit.globalCoordinates().x() + hit.xResidual();
          double hitY0 = hit.globalCoordinates().y() + hit.yResidual();
          uint32_t xBin =
              h2PlaneEfficiencyMap_[planeId]->GetXaxis()->FindBin(hitX0);
          uint32_t yBin =
              h2PlaneEfficiencyMap_[planeId]->GetYaxis()->FindBin(hitY0);
          planeEfficiency[plane] =
              h2PlaneEfficiencyMap_[planeId]->GetBinContent(xBin, yBin);
          if (debug_)
            std::cout << "Hit coordinates: (" << hitX0 << "," << hitY0
                      << ")\n Hit bins (" << xBin << "." << yBin << ")"
                      << std::endl;
        } // for each hit
      }   // for each hit collection
      for (const auto &plane : listOfPlanes_) {
        if (planeEfficiency.find(plane) == planeEfficiency.end()) {
          planeEfficiency[plane] = 0.;
        }
        if (debug_)
          std::cout << "Plane " << plane
                    << " efficiency: " << planeEfficiency[plane] << std::endl;
      }
      float efficiency = probabilityCalculation(planeEfficiency);
      h2RefinedTrackEfficiency_[rpId]->Fill(pixelX0, pixelY0, efficiency);
      if (station == 0)
        h2RefinedTrackEfficiency_rotated[rpId]->Fill(
            pixelX0_rotated, pixelY0_rotated, efficiency);
      if (debug_)
        std::cout << "Track passing through: (" << pixelX0 << "," << pixelY0
                  << ")\n Efficiency: " << efficiency << std::endl;
    } // for each track in the collection
  }   // for each track collection
}

void RefinedEfficiencyTool_2018::fillDescriptions(
    edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

void RefinedEfficiencyTool_2018::beginJob() {
  efficiencyFile_ = new TFile(efficiencyFileName_.data(), "READ");
  if (!efficiencyFile_->IsOpen()) {
    std::cout << "No efficiency file available!" << std::endl;
    throw 1;
  }
  for (auto &arm : listOfArms_) {
    for (auto &station : listOfStations_) {
      for (auto &plane : listOfPlanes_) {
        CTPPSPixelDetId planeId = CTPPSPixelDetId(arm, station, 3, plane);
        std::string h2planeEfficiencyMapName =
            Form("Arm%i_st%i_rp3/Arm%i_st%i_rp3_pl%i/"
                 "h2EfficiencyMap_arm%i_st%i_rp3_pl%i",
                 arm, station, arm, station, plane, arm, station, plane);
        if (efficiencyFile_->Get(h2planeEfficiencyMapName.data())) {
          h2PlaneEfficiencyMap_[planeId] = new TH2D(
              *((TH2D *)efficiencyFile_->Get(h2planeEfficiencyMapName.data())));
          detectorIdVector_.push_back(planeId);
        }
      }
    }
  }
  efficiencyFile_->Close();
  delete efficiencyFile_;
}

void RefinedEfficiencyTool_2018::endJob() {
  outputFile_ = new TFile(outputFileName_.data(), "RECREATE");
  std::cout << "Saving output in: " << outputFile_->GetName() << std::endl;

  for (auto &rpId : romanPotIdVector_) {
    uint32_t arm = rpId.arm();
    uint32_t station = rpId.station();
    std::string rpDirName = Form("Arm%i_st%i_rp3", arm, station);
    outputFile_->mkdir(rpDirName.data());
    outputFile_->cd(rpDirName.data());
    h2RefinedTrackEfficiency_[rpId]->Divide(h2RefinedTrackEfficiency_[rpId],
                                            h2TrackHitDistribution_[rpId]);
    h2RefinedTrackEfficiency_[rpId]->SetMaximum(1.);
    h2RefinedTrackEfficiency_[rpId]->Write();
    if (station == 0) {
      h2RefinedTrackEfficiency_rotated[rpId]->Divide(
          h2RefinedTrackEfficiency_rotated[rpId],
          h2TrackHitDistribution_rotated[rpId]);
      h2RefinedTrackEfficiency_rotated[rpId]->SetMaximum(1.);
      h2RefinedTrackEfficiency_rotated[rpId]->Write();
    }
  }
  outputFile_->Close();
  delete outputFile_;
}

// This function produces all the possible plane combinations extracting
// numberToExtract planes over numberOfPlanes planes
void RefinedEfficiencyTool_2018::getPlaneCombinations(
    const std::vector<uint32_t> &inputPlaneList, uint32_t numberToExtract,
    std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>>
        &planesExtractedAndNot) {
  uint32_t numberOfPlanes = inputPlaneList.size();
  std::string bitmask(numberToExtract, 1); // numberToExtract leading 1's
  bitmask.resize(numberOfPlanes,
                 0); // numberOfPlanes-numberToExtract trailing 0's
  planesExtractedAndNot.clear();

  // store the combination and permute bitmask
  do {
    planesExtractedAndNot.push_back(
        std::pair<std::vector<uint32_t>, std::vector<uint32_t>>(
            std::vector<uint32_t>(), std::vector<uint32_t>()));
    for (uint32_t i = 0; i < numberOfPlanes;
         ++i) { // [0..numberOfPlanes-1] integers
      if (bitmask[i])
        planesExtractedAndNot.back().second.push_back(inputPlaneList.at(i));
      else
        planesExtractedAndNot.back().first.push_back(inputPlaneList.at(i));
    }
  } while (std::prev_permutation(bitmask.begin(), bitmask.end()));

  return;
}

float RefinedEfficiencyTool_2018::probabilityCalculation(
    const std::map<uint32_t, float> &planeEfficiency) {

  int minNumberOfBlindPlanes = 3;
  int maxNumberOfBlindPlanes = listOfPlanes_.size();
  float rpEfficiency = 1.;

  for (uint32_t i = (uint32_t)minNumberOfBlindPlanes;
       i <= (uint32_t)maxNumberOfBlindPlanes; i++) {
    rpEfficiency -= probabilityNplanesBlind(listOfPlanes_, i, planeEfficiency);
  }
  return rpEfficiency;
}

float RefinedEfficiencyTool_2018::probabilityNplanesBlind(
    const std::vector<uint32_t> &inputPlaneList, int numberToExtract,
    const std::map<unsigned, float> &planeEfficiency) {
  std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>>
      planesExtractedAndNot;
  getPlaneCombinations(inputPlaneList, numberToExtract, planesExtractedAndNot);

  float probability = 0.;

  for (const auto &combination : planesExtractedAndNot) {
    float combinationProbability = 1.;
    for (const auto &efficientPlane : combination.first) {
      combinationProbability *= planeEfficiency.at(efficientPlane);
    }
    for (const auto &notEfficientPlane : combination.second) {
      combinationProbability *= (1. - planeEfficiency.at(notEfficientPlane));
    }
    probability += combinationProbability;
  }
  return probability;
}

bool RefinedEfficiencyTool_2018::Cut(CTPPSPixelLocalTrack track, int arm,
                                     int station) {
  double maxTx = 0.02;
  double maxTy = 0.02;
  double maxChi2 = TMath::ChisquareQuantile(0.95, track.ndf());
  double x = track.x0();
  double y = track.y0();
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

  if (TMath::Abs(track.tx()) > maxTx || TMath::Abs(track.ty()) > maxTy ||
      track.numberOfPointsUsedForFit() < minNumberOfPlanesForTrack_ ||
      track.numberOfPointsUsedForFit() > maxNumberOfPlanesForTrack_ ||
      track.chiSquaredOverNDF() * track.ndf() > maxChi2 ||
      y > fiducialYHigh_[std::pair<int, int>(arm, station)] ||
      y < fiducialYLow_[std::pair<int, int>(arm, station)] ||
      x < fiducialXLow_[std::pair<int, int>(arm, station)])
    return true;
  else
    return false;
}

// define this as a plug-in
DEFINE_FWK_MODULE(RefinedEfficiencyTool_2018);