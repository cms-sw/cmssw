// -*- C++ -*-
//
// Package:    RecoPPS/RPixEfficiencyTools
// Class:      EfficiencyTool_2018
//
/**\class EfficiencyTool_2018 EfficiencyTool_2018.cc
 RecoPPS/RPixEfficiencyTools/plugins/EfficiencyTool_2018.cc

 Description: [one line class summary]

 Implementation:
                 [Notes on implementation]
*/
//
// Original Author:  Andrea Bellora
//         Created:  Wed, 22 Aug 2017 09:55:05 GMT
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
#include "FWCore/Framework/interface/EventSetup.h" 
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/CTPPSDetId/interface/CTPPSPixelDetId.h"
#include "DataFormats/CTPPSReco/interface/CTPPSPixelLocalTrack.h"
#include "DataFormats/CTPPSReco/interface/CTPPSPixelRecHit.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/RunInfo/interface/LHCInfo.h"
#include "CondFormats/DataRecord/interface/LHCInfoRcd.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <TF1.h>
#include <TFile.h>
#include <TGraphErrors.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TMath.h>
#include <TObjArray.h>

class EfficiencyTool_2018
    : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit EfficiencyTool_2018(const edm::ParameterSet &);
  ~EfficiencyTool_2018();
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  virtual void beginJob() override;
  virtual void analyze(const edm::Event &, const edm::EventSetup &) override;
  virtual void endJob() override;

  // Computes the probability of having numberToExtract inefficient planes
  float
  probabilityNplanesBlind(const std::vector<uint32_t> &inputPlaneList,
                          int numberToExtract,
                          const std::map<unsigned, float> &planeEfficiency);

  // Computes all the combination of planes with numberToExtract planes
  // extracted
  void getPlaneCombinations(
      const std::vector<uint32_t> &inputPlaneList, uint32_t numberToExtract,
      std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>>
          &planesExtractedAndNot);

  // Computes the efficiency, given certain plane efficiency values
  float
  probabilityCalculation(const std::map<uint32_t, float> &planeEfficiency);

  // Obsolete and not correct error propagation
  float errorCalculation(const std::map<uint32_t, float> &planeEfficiency,
                         const std::map<uint32_t, float> &planeEfficiencyError);
  float efficiencyPartialDerivativewrtPlane(
      uint32_t plane, const std::vector<uint32_t> &inputPlaneList,
      int numberToExtract, const std::map<unsigned, float> &planeEfficiency);

  // Return true if a track should be discarded
  bool Cut(CTPPSPixelLocalTrack track, int arm, int station);

  edm::EDGetTokenT<edm::DetSetVector<CTPPSPixelLocalTrack>>
      pixelLocalTrackToken_;
  edm::EDGetTokenT<edm::DetSetVector<CTPPSPixelRecHit>> pixelRecHitToken_;

  TFile *outputFile_;
  std::string outputFileName_;
  bool isCorrelationPlotEnabled;
  bool supplementaryPlots;
  int minNumberOfPlanesForEfficiency_;
  int minNumberOfPlanesForTrack_;
  int maxNumberOfPlanesForTrack_;
  int minTracksPerEvent;
  int maxTracksPerEvent;
  std::string producerTag;

  static const unsigned int totalNumberOfBunches_ = 3564;
  std::string bunchSelection_;
  std::string bunchListFileName_;
  bool validBunchArray_[totalNumberOfBunches_];

  TH1D *h1BunchCrossing_;
  TH1D *h1CrossingAngle_;
  std::map<CTPPSPixelDetId, TH2D *> h2ModuleHitMap_;
  std::map<CTPPSPixelDetId, TH2D *> h2EfficiencyMap_;
  std::map<CTPPSPixelDetId, TH2D *> h2AuxEfficiencyMap_;
  std::map<CTPPSPixelDetId, TH2D *> h2EfficiencyNormalizationMap_;
  std::map<CTPPSPixelDetId, TH2D *> h2TrackHitDistribution_;
  std::map<int, std::map<CTPPSPixelDetId, TH2D *>>
      h2TrackHitDistributionBinShift_;
  std::map<CTPPSPixelDetId, TH2D *> h23PointsTrackHitDistribution_;
  std::map<CTPPSPixelDetId, TH2D *> h2TrackEfficiencyMap_;
  std::map<CTPPSPixelDetId, TH2D *> h2TrackEfficiencyErrorMap_;
  std::map<CTPPSPixelDetId, TH1D *> h1NumberOfTracks_;
  std::map<CTPPSPixelDetId, TGraph *> g1X0Correlation_;
  std::map<CTPPSPixelDetId, TGraph *> g1Y0Correlation_;
  std::map<CTPPSPixelDetId, TH2D *> h2X0Correlation_;
  std::map<CTPPSPixelDetId, TH2D *> h2Y0Correlation_;
  std::map<CTPPSPixelDetId, TH2D *> h2AvgPlanesUsed_;
  std::map<CTPPSPixelDetId, TH1D *> h1PlanesUsed_;
  std::map<CTPPSPixelDetId, TH1D *> h1ChiSquaredOverNDF_;
  std::map<CTPPSPixelDetId, TH1D *> h1ConsecutivePlanes_;

  std::map<CTPPSPixelDetId, TH2D *> h2ModuleHitMap_rotated;
  std::map<CTPPSPixelDetId, TH2D *> h2EfficiencyMap_rotated;
  std::map<CTPPSPixelDetId, TH2D *> h2AuxEfficiencyMap_rotated;
  std::map<CTPPSPixelDetId, TH2D *> h2EfficiencyNormalizationMap_rotated;
  std::map<CTPPSPixelDetId, TH2D *> h2TrackHitDistribution_rotated;
  std::map<int, std::map<CTPPSPixelDetId, TH2D *>>
      h2TrackHitDistributionBinShift_rotated;
  std::map<CTPPSPixelDetId, TH2D *> h2TrackEfficiencyMap_rotated;
  std::map<CTPPSPixelDetId, TH2D *> h2TrackEfficiencyErrorMap_rotated;
  std::map<CTPPSPixelDetId, TH2D *> h2AvgPlanesUsed_rotated;

  // Resolution histograms
  std::map<CTPPSPixelDetId, std::map<std::pair<int, int>, TH1D *>>
      h1X0Sigma; // map<DetId,map< pair< PlanesUsedForFit, PlanesWithColCls >,
                 // Sigma > >
  std::map<CTPPSPixelDetId, std::map<std::pair<int, int>, TH1D *>>
      h1Y0Sigma; // map<DetId,map< pair< PlanesUsedForFit, PlanesWithColCls >,
                 // Sigma > >
  std::map<CTPPSPixelDetId, std::map<std::pair<int, int>, TH1D *>>
      h1TxSigma; // map<DetId,map< pair< PlanesUsedForFit, PlanesWithColCls >,
                 // Sigma > >
  std::map<CTPPSPixelDetId, std::map<std::pair<int, int>, TH1D *>>
      h1TySigma; // map<DetId,map< pair< PlanesUsedForFit, PlanesWithColCls >,
                 // Sigma > >

  std::vector<CTPPSPixelDetId> detectorIdVector_;
  std::vector<CTPPSPixelDetId> romanPotIdVector_;

  std::vector<uint32_t> listOfPlanes_ = {0, 1, 2, 3, 4, 5};
  std::vector<int> binShifts_ = {0,   5,   10,  15,  20,  25,  30,  35,  40, 45,
                                 50,  55,  60,  65,  70,  75,  80,  85,  90, 95,
                                 100, 105, 110, 115, 120, 125, 130, 135, 140};

  int binGroupingX = 1;
  int binGroupingY = 1;

  int mapXbins_st2 = 200 / binGroupingX;
  float mapXmin_st2 = 0. * TMath::Cos(18.4 / 180. * TMath::Pi());
  float mapXmax_st2 = 30. * TMath::Cos(18.4 / 180. * TMath::Pi());
  int mapYbins_st2 = 240 / binGroupingY;
  float mapYmin_st2 = -16.;
  float mapYmax_st2 = 8.;
  float fitXmin_st2 = 6.;
  float fitXmax_st2 = 19.;

  int mapXbins_st0 = 200 / binGroupingX;
  float mapXmin_st0 = 0. * TMath::Cos(18.4 / 180. * TMath::Pi());
  float mapXmax_st0 = 30. * TMath::Cos(18.4 / 180. * TMath::Pi());
  int mapYbins_st0 = 240 / binGroupingY;
  float mapYmin_st0 = -16.;
  float mapYmax_st0 = 8.;
  float fitXmin_st0 = 45.;
  float fitXmax_st0 = 58.;

  int mapXbins = mapXbins_st0;
  float mapXmin = mapXmin_st0;
  float mapXmax = mapXmax_st0;
  int mapYbins = mapYbins_st0;
  float mapYmin = mapYmin_st0;
  float mapYmax = mapYmax_st0;
  float fitXmin = fitXmin_st0;
  float fitXmax = fitXmax_st0;

  std::map<CTPPSPixelDetId, int> binAlignmentParameters = {
      {CTPPSPixelDetId(0, 0, 3), 0},
      {CTPPSPixelDetId(0, 2, 3), 0},
      {CTPPSPixelDetId(1, 0, 3), 0},
      {CTPPSPixelDetId(1, 2, 3), 0}};

  int numberOfAttempts = 0;
  std::vector<double> fiducialXLowVector_;
  std::vector<double> fiducialYLowVector_;
  std::vector<double> fiducialYHighVector_;
  std::map<std::pair<int, int>, double> fiducialXLow_;
  std::map<std::pair<int, int>, double> fiducialYLow_;
  std::map<std::pair<int, int>, double> fiducialYHigh_;
};

EfficiencyTool_2018::EfficiencyTool_2018(const edm::ParameterSet &iConfig) {
  usesResource("TFileService");

  producerTag = iConfig.getUntrackedParameter<std::string>("producerTag");

  pixelLocalTrackToken_ = consumes<edm::DetSetVector<CTPPSPixelLocalTrack>>(
      edm::InputTag("ctppsPixelLocalTracks", "",producerTag));
  pixelRecHitToken_ = consumes<edm::DetSetVector<CTPPSPixelRecHit>>(
      edm::InputTag("ctppsPixelRecHits", "",producerTag));
  outputFileName_ =
      iConfig.getUntrackedParameter<std::string>("outputFileName");
  minNumberOfPlanesForEfficiency_ =
      iConfig.getParameter<int>("minNumberOfPlanesForEfficiency");
  minNumberOfPlanesForTrack_ =
      iConfig.getParameter<int>("minNumberOfPlanesForTrack");
  maxNumberOfPlanesForTrack_ =
      iConfig.getParameter<int>("maxNumberOfPlanesForTrack");
  isCorrelationPlotEnabled =
      iConfig.getParameter<bool>("isCorrelationPlotEnabled");
  minTracksPerEvent = iConfig.getParameter<int>("minTracksPerEvent");
  maxTracksPerEvent = iConfig.getParameter<int>("maxTracksPerEvent");
  supplementaryPlots = iConfig.getParameter<bool>("supplementaryPlots");
  bunchSelection_ =
      iConfig.getUntrackedParameter<std::string>("bunchSelection");
  bunchListFileName_ =
      iConfig.getUntrackedParameter<std::string>("bunchListFileName");
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
edm::LogInfo   ("category") <<"\n\n\n\n TESTING \n\n\n\n";
std::cout<<"CUSTOM PARAM: "<<iConfig.getParameter<std::string>("customParameterTest");
}
EfficiencyTool_2018::~EfficiencyTool_2018() {
  delete h1BunchCrossing_;
  delete h1CrossingAngle_;
  for (const auto &rpId : romanPotIdVector_) {
    delete h2TrackHitDistribution_[rpId];
    if (supplementaryPlots) {
      for (auto binShift : binShifts_)
        delete h2TrackHitDistributionBinShift_[binShift][rpId];
    }
    delete h23PointsTrackHitDistribution_[rpId];
    delete h2TrackEfficiencyMap_[rpId];
    delete h2TrackEfficiencyErrorMap_[rpId];
    delete h1NumberOfTracks_[rpId];
    delete g1X0Correlation_[rpId];
    delete g1Y0Correlation_[rpId];
    delete h2X0Correlation_[rpId];
    delete h2Y0Correlation_[rpId];

    if (supplementaryPlots) {
      for (int nPlanes = 3; nPlanes <= 6; nPlanes++) {
        for (int numberOfCls = 0; numberOfCls <= nPlanes; numberOfCls++) {
          // std::cout << "Deleting hist " << nPlanes << " " << numberOfCls <<
          // std::endl;
          delete h1X0Sigma[rpId][std::pair(nPlanes, numberOfCls)];
          delete h1Y0Sigma[rpId][std::pair(nPlanes, numberOfCls)];
          delete h1TxSigma[rpId][std::pair(nPlanes, numberOfCls)];
          delete h1TySigma[rpId][std::pair(nPlanes, numberOfCls)];
        }
      }
      delete h2AvgPlanesUsed_[rpId];
      delete h1PlanesUsed_[rpId];
      delete h1ChiSquaredOverNDF_[rpId];
      delete h1ConsecutivePlanes_[rpId];
    }

    if (rpId.station() == 0) {
      delete h2TrackEfficiencyMap_rotated[rpId];
      delete h2TrackEfficiencyErrorMap_rotated[rpId];
      if (supplementaryPlots) {
        delete h2TrackHitDistribution_rotated[rpId];
        for (auto binShift : binShifts_)
          delete h2TrackHitDistributionBinShift_rotated[binShift][rpId];
        delete h2AvgPlanesUsed_rotated[rpId];
      }
    }
  }

  for (const auto &detId : detectorIdVector_) {
    delete h2ModuleHitMap_[detId];
    delete h2EfficiencyNormalizationMap_[detId];
    delete h2EfficiencyMap_[detId];
    delete h2AuxEfficiencyMap_[detId];

    if (detId.station() == 0) {
      delete h2EfficiencyNormalizationMap_rotated[detId];
      delete h2EfficiencyMap_rotated[detId];
      delete h2AuxEfficiencyMap_rotated[detId];
      if (supplementaryPlots) {
        delete h2ModuleHitMap_rotated[detId];
      }
    }
  }
}

void EfficiencyTool_2018::analyze(const edm::Event &iEvent,
                                  const edm::EventSetup &iSetup) {
  using namespace edm;
  Handle<edm::DetSetVector<CTPPSPixelRecHit>> pixelRecHits;
  iEvent.getByToken(pixelRecHitToken_, pixelRecHits);

  Handle<edm::DetSetVector<CTPPSPixelLocalTrack>> pixelLocalTracks;
  iEvent.getByToken(pixelLocalTrackToken_, pixelLocalTracks);

  if (!validBunchArray_[iEvent.eventAuxiliary().bunchCrossing()])
    return;
  h1BunchCrossing_->Fill(iEvent.eventAuxiliary().bunchCrossing());

  edm::ESHandle<LHCInfo> pSetup;
  const std::string label = "";
  iSetup.get<LHCInfoRcd>().get(label, pSetup);

  // re-initialise algorithm upon crossing-angle change
  const LHCInfo *pInfo = pSetup.product();
  h1CrossingAngle_->Fill(pInfo->crossingAngle());

  for (const auto &rpPixeltrack : *pixelLocalTracks) {
    if ((uint32_t)minTracksPerEvent > rpPixeltrack.size() ||
        rpPixeltrack.size() > (uint32_t)maxTracksPerEvent)
      continue;
    CTPPSPixelDetId rpId = CTPPSPixelDetId(rpPixeltrack.detId());
    uint32_t arm = rpId.arm();
    uint32_t rp = rpId.rp();
    uint32_t station = rpId.station();

    if (station == 2) {
      mapXbins = mapXbins_st2;
      mapXmin = mapXmin_st2;
      mapXmax = mapXmax_st2;
      mapYbins = mapYbins_st2;
      mapYmin = mapYmin_st2;
      mapYmax = mapYmax_st2;
      fitXmin = fitXmin_st2;
      fitXmax = fitXmax_st2;
    } else {
      mapXbins = mapXbins_st0;
      mapXmin = mapXmin_st0;
      mapXmax = mapXmax_st0;
      mapYbins = mapYbins_st0;
      mapYmin = mapYmin_st0;
      mapYmax = mapYmax_st0;
      fitXmin = fitXmin_st0;
      fitXmax = fitXmax_st0;
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
      h23PointsTrackHitDistribution_[rpId] = new TH2D(
          Form("h23PointsTrackHitDistribution_arm%i_st%i_rp%i", arm, station,
               rp),
          Form("h23PointsTrackHitDistribution_arm%i_st%i_rp%i;x (mm);y (mm)",
               arm, station, rp),
          mapXbins, mapXmin, mapXmax, mapYbins, mapYmin, mapYmax);
      h2TrackEfficiencyMap_[rpId] = new TH2D(
          Form("h2TrackEfficiencyMap_arm%i_st%i_rp%i", arm, station, rp),
          Form("h2TrackEfficiencyMap_arm%i_st%i_rp%i; x (mm); y (mm)", arm,
               station, rp),
          mapXbins, mapXmin, mapXmax, mapYbins, mapYmin, mapYmax);
      h2TrackEfficiencyErrorMap_[rpId] = new TH2D(
          Form("h2TrackEfficiencyErrorMap_arm%i_st%i_rp%i", arm, station, rp),
          Form("h2TrackEfficiencyErrorMap_arm%i_st%i_rp%i; x (mm); y (mm)", arm,
               station, rp),
          mapXbins, mapXmin, mapXmax, mapYbins, mapYmin, mapYmax);
      h1NumberOfTracks_[rpId] = new TH1D(
          Form("h1NumberOfTracks_arm%i_st%i_rp%i", arm, station, rp),
          Form("h1NumberOfTracks_arm%i_st%i_rp%i; Tracks;", arm, station, rp),
          16, -0.5, 15.5);
      if (supplementaryPlots) {
        for (auto binShift : binShifts_) {
          h2TrackHitDistributionBinShift_[binShift][rpId] = new TH2D(
              Form("h2TrackHitDistributionBinShift_%i_arm%i_st%i_rp%i",
                   binShift, arm, station, rp),
              Form("h2TrackHitDistributionBinShift_%i_arm%i_st%i_rp%i;x (mm);y "
                   "(mm)",
                   binShift, arm, station, rp),
              mapXbins, mapXmin + binShift * binSize / 150.,
              mapXmax + binShift * binSize / 150., mapYbins, mapYmin, mapYmax);
        }
        h2AvgPlanesUsed_[rpId] =
            new TH2D(Form("h2AvgPlanesUsed_arm%i_st%i_rp%i", arm, station, rp),
                     Form("h2AvgPlanesUsed_arm%i_st%i_rp%i; x (mm); y (mm)",
                          arm, station, rp),
                     mapXbins, mapXmin, mapXmax, mapYbins, mapYmin, mapYmax);
        h1PlanesUsed_[rpId] = new TH1D(
            Form("h1PlanesUsed_arm%i_st%i_rp%i", arm, station, rp),
            Form("h1PlanesUsed_arm%i_st%i_rp%i; Planes", arm, station, rp), 7,
            -0.5, 6.5);
        h1ChiSquaredOverNDF_[rpId] = new TH1D(
            Form("h1ChiSquaredOverNDF_arm%i_st%i_rp%i", arm, station, rp),
            Form("h1ChiSquaredOverNDF_arm%i_st%i_rp%i; Planes", arm, station,
                 rp),
            100, 0, 5);
        for (int nPlanes = 3; nPlanes <= 6; nPlanes++) {
          for (int numberOfCls = 0; numberOfCls <= nPlanes; numberOfCls++) {
            // std::cout << "Creating hist " << nPlanes << " " << numberOfCls <<
            // "Arm " << arm << "Station " << station <<std::endl;
            h1X0Sigma[rpId][std::pair(nPlanes, numberOfCls)] =
                new TH1D(Form("h1X0Sigma_arm%i_st%i_rp%i_nPlanes%i_nCls%i", arm,
                              station, rp, nPlanes, numberOfCls),
                         Form("h1X0Sigma_arm%i_st%i_rp%i_nPlanes%i_nCls%i; "
                              "#sigma_{x} (mm);",
                              arm, station, rp, nPlanes, numberOfCls),
                         100, 0, 0.1);
            h1Y0Sigma[rpId][std::pair(nPlanes, numberOfCls)] =
                new TH1D(Form("h1Y0Sigma_arm%i_st%i_rp%i_nPlanes%i_nCls%i", arm,
                              station, rp, nPlanes, numberOfCls),
                         Form("h1Y0Sigma_arm%i_st%i_rp%i_nPlanes%i_nCls%i; "
                              "#sigma_{y} (mm);",
                              arm, station, rp, nPlanes, numberOfCls),
                         100, 0, 0.1);
            h1TxSigma[rpId][std::pair(nPlanes, numberOfCls)] = new TH1D(
                Form("h1TxSigma_arm%i_st%i_rp%i_nPlanes%i_nCls%i", arm, station,
                     rp, nPlanes, numberOfCls),
                Form("h1TxSigma_arm%i_st%i_rp%i_nPlanes%i_nCls%i; #sigma_{Tx};",
                     arm, station, rp, nPlanes, numberOfCls),
                100, 0, 0.02);
            h1TySigma[rpId][std::pair(nPlanes, numberOfCls)] = new TH1D(
                Form("h1TySigma_arm%i_st%i_rp%i_nPlanes%i_nCls%i", arm, station,
                     rp, nPlanes, numberOfCls),
                Form("h1TySigma_arm%i_st%i_rp%i_nPlanes%i_nCls%i; #sigma_{Ty};",
                     arm, station, rp, nPlanes, numberOfCls),
                100, 0, 0.02);
          }
        }
        h1ConsecutivePlanes_[rpId] = new TH1D(
            Form("h1ConsecutivePlanes_arm%i_st%i_rp%i", arm, station, rp),
            Form("h1ConsecutivePlanes_arm%i_st%i_rp%i; #sigma_{Ty};", arm,
                 station, rp),
            2, 0, 2);
        h1ConsecutivePlanes_[rpId]->GetXaxis()->SetBinLabel(1,
                                                            "Non-consecutive");
        h1ConsecutivePlanes_[rpId]->GetXaxis()->SetBinLabel(2, "Consecutive");
      }
      if (station == 0) {
        h2TrackEfficiencyMap_rotated[rpId] = new TH2D(
            Form("h2TrackEfficiencyMap_rotated_arm%i_st%i_rp%i", arm, station,
                 rp),
            Form("h2TrackEfficiencyMap_rotated_arm%i_st%i_rp%i; x (mm); y (mm)",
                 arm, station, rp),
            mapXbins, mapXmin, mapXmax, mapYbins, mapYmin, mapYmax);
        h2TrackEfficiencyErrorMap_rotated[rpId] =
            new TH2D(Form("h2TrackEfficiencyErrorMap_rotated_arm%i_st%i_rp%i",
                          arm, station, rp),
                     Form("h2TrackEfficiencyErrorMap_rotated_arm%i_st%i_rp%i; "
                          "x (mm); y (mm)",
                          arm, station, rp),
                     mapXbins, mapXmin, mapXmax, mapYbins, mapYmin, mapYmax);
        if (supplementaryPlots) {
          h2TrackHitDistribution_rotated[rpId] =
              new TH2D(Form("h2TrackHitDistribution_rotated_arm%i_st%i_rp%i",
                            arm, station, rp),
                       Form("h2TrackHitDistribution_rotated_arm%i_st%i_rp%i;x "
                            "(mm);y (mm)",
                            arm, station, rp),
                       mapXbins, mapXmin, mapXmax, mapYbins, mapYmin, mapYmax);
          for (auto binShift : binShifts_) {
            h2TrackHitDistributionBinShift_rotated[binShift][rpId] = new TH2D(
                Form(
                    "h2TrackHitDistributionBinShift_rotated_%i_arm%i_st%i_rp%i",
                    binShift, arm, station, rp),
                Form("h2TrackHitDistributionBinShift_rotated_%i_arm%i_st%i_rp%"
                     "i;x (mm);y (mm)",
                     binShift, arm, station, rp),
                mapXbins, mapXmin + binShift * binSize / 150,
                mapXmax + binShift * binSize / 150, mapYbins, mapYmin, mapYmax);
          }
          h2AvgPlanesUsed_rotated[rpId] = new TH2D(
              Form("h2AvgPlanesUsed_rotated_arm%i_st%i_rp%i", arm, station, rp),
              Form("h2AvgPlanesUsed_rotated_arm%i_st%i_rp%i; x (mm); y (mm)",
                   arm, station, rp),
              mapXbins, mapXmin, mapXmax, mapYbins, mapYmin, mapYmax);
        }
      }
    }
    for (const auto &pixeltrack : rpPixeltrack) {
      // if ((pixeltrack.ndf() +4 )/2 < minNumberOfPlanesForTrack_ ||
      // (pixeltrack.ndf() +4 )/2 > maxNumberOfPlanesForTrack_ ) continue;
      if (Cut(pixeltrack, arm, station))
        continue;
      if (!pixeltrack.isValid())
        continue;
      h1NumberOfTracks_[rpId]->Fill(rpPixeltrack.size());

      float pixelX0 = pixeltrack.x0();
      float pixelY0 = pixeltrack.y0();
      int numberOfRowCls2 = 0;
      int numberOfColCls2 = 0;
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

      int numberOfFittedPoints = 0;
      std::vector<int> planesContributingToTrack;
      edm::DetSetVector<CTPPSPixelFittedRecHit> fittedHits =
          pixeltrack.hits();

      std::map<uint32_t, int> numberOfPointPerPlaneEff;
      for (const auto pln : listOfPlanes_) {
        numberOfPointPerPlaneEff[pln] = 0;
      }

      for (const auto &planeHits : fittedHits) {
        CTPPSPixelDetId planeId = CTPPSPixelDetId(planeHits.detId());
        uint32_t plane = planeId.plane();
        if (h2ModuleHitMap_.find(planeId) == h2ModuleHitMap_.end()) {
          detectorIdVector_.push_back(planeId);

          h2ModuleHitMap_[planeId] = new TH2D(
              Form("h2ModuleHitMap_arm%i_st%i_rp%i_pl%i", arm, station, rp,
                   plane),
              Form("h2ModuleHitMap_arm%i_st%i_rp%i_pl%i; x (mm); y (mm)", arm,
                   station, rp, plane),
              mapXbins, mapXmin, mapXmax, mapYbins, mapYmin, mapYmax);
          h2EfficiencyMap_[planeId] = new TH2D(
              Form("h2EfficiencyMap_arm%i_st%i_rp%i_pl%i", arm, station, rp,
                   plane),
              Form("h2EfficiencyMap_arm%i_st%i_rp%i_pl%i; x (mm); y (mm)", arm,
                   station, rp, plane),
              mapXbins, mapXmin, mapXmax, mapYbins, mapYmin, mapYmax);
          h2AuxEfficiencyMap_[planeId] =
              new TH2D(Form("h2AuxEfficiencyMap_arm%i_st%i_rp%i_pl%i", arm,
                            station, rp, plane),
                       Form("h2AuxEfficiencyMap_arm%i_st%i_rp%i_pl%i", arm,
                            station, rp, plane),
                       mapXbins, mapXmin, mapXmax, mapYbins, mapYmin, mapYmax);
          h2EfficiencyNormalizationMap_[planeId] =
              new TH2D(Form("h2EfficiencyNormalizationMap_arm%i_st%i_rp%i_pl%i",
                            arm, station, rp, plane),
                       Form("h2EfficiencyNormalizationMap_arm%i_st%i_rp%i_pl%i",
                            arm, station, rp, plane),
                       mapXbins, mapXmin, mapXmax, mapYbins, mapYmin, mapYmax);
          if (station == 0) {
            h2EfficiencyMap_rotated[planeId] = new TH2D(
                Form("h2EfficiencyMap_rotated_arm%i_st%i_rp%i_pl%i", arm,
                     station, rp, plane),
                Form("h2EfficiencyMap_rotated_arm%i_st%i_rp%i_pl%i; x (mm); y "
                     "(mm)",
                     arm, station, rp, plane),
                mapXbins, mapXmin, mapXmax, mapYbins, mapYmin, mapYmax);
            h2AuxEfficiencyMap_rotated[planeId] = new TH2D(
                Form("h2AuxEfficiencyMap_rotated_arm%i_st%i_rp%i_pl%i", arm,
                     station, rp, plane),
                Form("h2AuxEfficiencyMap_rotated_arm%i_st%i_rp%i_pl%i", arm,
                     station, rp, plane),
                mapXbins, mapXmin, mapXmax, mapYbins, mapYmin, mapYmax);
            h2EfficiencyNormalizationMap_rotated[planeId] = new TH2D(
                Form(
                    "h2EfficiencyNormalizationMap_rotated_arm%i_st%i_rp%i_pl%i",
                    arm, station, rp, plane),
                Form(
                    "h2EfficiencyNormalizationMap_rotated_arm%i_st%i_rp%i_pl%i",
                    arm, station, rp, plane),
                mapXbins, mapXmin, mapXmax, mapYbins, mapYmin, mapYmax);

            if (supplementaryPlots) {
              h2ModuleHitMap_rotated[planeId] = new TH2D(
                  Form("h2ModuleHitMap_rotated_arm%i_st%i_rp%i_pl%i", arm,
                       station, rp, plane),
                  Form("h2ModuleHitMap_rotated_arm%i_st%i_rp%i_pl%i; x (mm); y "
                       "(mm)",
                       arm, station, rp, plane),
                  mapXbins, mapXmin, mapXmax, mapYbins, mapYmin, mapYmax);
            }
          }
        }

        for (const auto &hit : planeHits) {
          if (hit.isUsedForFit()) {
            ++numberOfFittedPoints;
            planesContributingToTrack.push_back(plane);
            // Counting, for each plane, how many of the others were hit, to
            // establish if its efficiency is computable in the event
            for (auto pln : numberOfPointPerPlaneEff) {
              if (pln.first == planeId.plane())
                continue;
              numberOfPointPerPlaneEff[pln.first] = pln.second + 1;
            }
            double hitX0 = hit.globalCoordinates().x() + hit.xResidual();
            double hitY0 = hit.globalCoordinates().y() + hit.yResidual();
            double hitX0_rotated = 0;
            double hitY0_rotated = 0;
            if (station == 0) {
              hitX0_rotated = hitX0 * TMath::Cos((-8. / 180.) * TMath::Pi()) -
                              hitY0 * TMath::Sin((-8. / 180.) * TMath::Pi());
              hitY0_rotated = hitX0 * TMath::Sin((-8. / 180.) * TMath::Pi()) +
                              hitY0 * TMath::Cos((-8. / 180.) * TMath::Pi());
            }
            h2ModuleHitMap_[planeId]->Fill(hitX0, hitY0);

            if (supplementaryPlots && station == 0) {
              h2ModuleHitMap_rotated[planeId]->Fill(hitX0_rotated,
                                                    hitY0_rotated);
            }
            if (hit.clusterSizeRow() == 2)
              ++numberOfRowCls2;
            if (hit.clusterSizeCol() == 2)
              ++numberOfColCls2;
          }
        }
      }

      h2TrackHitDistribution_[rpId]->Fill(pixelX0, pixelY0);
      if (supplementaryPlots) {
        for (auto binShift : binShifts_) {
          h2TrackHitDistributionBinShift_[binShift][rpId]->Fill(pixelX0,
                                                                pixelY0);
        }
      }

      if (supplementaryPlots) {
        h2AvgPlanesUsed_[rpId]->Fill(pixelX0, pixelY0, numberOfFittedPoints);
        h1PlanesUsed_[rpId]->Fill(numberOfFittedPoints);
        h1ChiSquaredOverNDF_[rpId]->Fill(pixeltrack.chiSquaredOverNDF());

        // Sort the vector of planes and require them to be consecutive and fill
        // hist
        std::sort(planesContributingToTrack.begin(),
                  planesContributingToTrack.end());
        bool areConsecutive = true;
        for (auto it = planesContributingToTrack.begin();
             it != (planesContributingToTrack.end() - 1); it++) {
          if (*(it + 1) - *it != 1) {
            areConsecutive = false;
            break;
          }
        }
        h1ConsecutivePlanes_[rpId]->Fill(areConsecutive);

        if (station == 0) {
          h2AvgPlanesUsed_rotated[rpId]->Fill(pixelX0_rotated, pixelY0_rotated,
                                              numberOfFittedPoints);
          h2TrackHitDistribution_rotated[rpId]->Fill(pixelX0_rotated,
                                                     pixelY0_rotated);
          for (auto binShift : binShifts_) {
            h2TrackHitDistributionBinShift_rotated[binShift][rpId]->Fill(
                pixelX0_rotated, pixelY0_rotated);
          }
        }
      }
      if (numberOfFittedPoints == 3) {

        // For Run 324841, to estimate 3 planes tracks in the damaged region
        // if(arm == 0 && station == 2){
        // 	if(pixelX0 > 43.90 && pixelX0 < 45.48 && pixelY0 > 3.39 &&
        // pixelY0 < 4.59)
        // h23PointsTrackHitDistribution_[rpId]->Fill(pixelX0,pixelY0);
        // }
        // else if(arm == 0 && station == 0){
        // 	if(pixelX0 > 6.18 && pixelX0 < 7.46 && pixelY0 > 4.36 && pixelY0
        // < 5.19) h23PointsTrackHitDistribution_[rpId]->Fill(pixelX0,pixelY0);
        // }
        // else if(arm == 1 && station ==2){
        // 	if(pixelX0 > 44.15 && pixelX0 < 45.29 && pixelY0 > 2.28 &&
        // pixelY0 < 4.89)
        // h23PointsTrackHitDistribution_[rpId]->Fill(pixelX0,pixelY0);
        // }
        // else if(arm == 1 && station == 0){
        // 	if(pixelX0 > 5.31 && pixelX0 < 6.31 && pixelY0 > 3.7 && pixelY0
        // < 5.40) h23PointsTrackHitDistribution_[rpId]->Fill(pixelX0,pixelY0);
        // }
        h23PointsTrackHitDistribution_[rpId]->Fill(pixelX0, pixelY0);
      }

      if (supplementaryPlots && pixeltrack.chiSquaredOverNDF() < 2.) {
        h1X0Sigma[rpId][std::pair(numberOfFittedPoints, numberOfColCls2)]->Fill(
            pixeltrack.x0Sigma());
        h1Y0Sigma[rpId][std::pair(numberOfFittedPoints, numberOfRowCls2)]->Fill(
            pixeltrack.y0Sigma());
        h1TxSigma[rpId][std::pair(numberOfFittedPoints, numberOfColCls2)]->Fill(
            pixeltrack.txSigma());
        h1TySigma[rpId][std::pair(numberOfFittedPoints, numberOfRowCls2)]->Fill(
            pixeltrack.tySigma());
      }
      // Efficiency calculation
      for (const auto pln : listOfPlanes_) {
        CTPPSPixelDetId planeId = rpId;
        planeId.setPlane(pln);
        edm::DetSet<CTPPSPixelFittedRecHit> hitOnPlane = fittedHits[planeId];
        float hitX0 = hitOnPlane[0].globalCoordinates().x() +
                      hitOnPlane[0].xResidual();
        ;
        float hitY0 = hitOnPlane[0].globalCoordinates().y() +
                      hitOnPlane[0].yResidual();
        ;
        float hitX0_rotated = 0;
        float hitY0_rotated = 0;
        if (station == 0) {
          hitX0_rotated = hitX0 * TMath::Cos((-8. / 180.) * TMath::Pi()) -
                          hitY0 * TMath::Sin((-8. / 180.) * TMath::Pi());
          hitY0_rotated = hitX0 * TMath::Sin((-8. / 180.) * TMath::Pi()) +
                          hitY0 * TMath::Cos((-8. / 180.) * TMath::Pi());
        }
        if (numberOfPointPerPlaneEff[pln] >= minNumberOfPlanesForEfficiency_) {
          h2EfficiencyNormalizationMap_[planeId]->Fill(hitX0, hitY0);

          if (station == 0) {
            h2EfficiencyNormalizationMap_rotated[planeId]->Fill(hitX0_rotated,
                                                                hitY0_rotated);
          }
          if (hitOnPlane[0].isRealHit()) {
            h2AuxEfficiencyMap_[planeId]->Fill(hitX0, hitY0);
            if (station == 0) {
              h2AuxEfficiencyMap_rotated[planeId]->Fill(hitX0_rotated,
                                                        hitY0_rotated);
            }
          }
        }
      }
    }
  }
}

void EfficiencyTool_2018::beginJob() {

  // Applying bunch selection
  h1BunchCrossing_ = new TH1D("h1BunchCrossing", "h1BunchCrossing",
                              totalNumberOfBunches_, 0., totalNumberOfBunches_);
  h1CrossingAngle_ =
      new TH1D("h1CrossingAngle", "h1CrossingAngle", 70, 100., 170);
  std::ifstream bunchListFile(bunchListFileName_.data());
  if (bunchSelection_ == "NoSelection") {
    std::fill_n(validBunchArray_, totalNumberOfBunches_, true);
    return;
  }
  if (!bunchListFile.good()) {
    std::cout << "BunchList file not good. Skipping buch selection..."
              << std::endl;
    return;
  }

  bool filledBunchArray[totalNumberOfBunches_];
  std::fill_n(filledBunchArray, totalNumberOfBunches_, false);
  std::fill_n(validBunchArray_, totalNumberOfBunches_, false);

  bool startReading = false;
  while (bunchListFile.good()) {
    std::string line;
    getline(bunchListFile, line);
    if (line == "" || line == "\r")
      continue;
    std::vector<std::string> elements;
    boost::split(elements, line, boost::is_any_of(","));
    if (elements.at(0) == "B1 bucket number") {
      startReading = true;
      continue;
    }
    if (line.find("HEAD ON COLLISIONS FOR B2") != std::string::npos)
      break;
    if (!startReading)
      continue;
    if (elements.at(3) != "-")
      filledBunchArray[(std::atoi(elements.at(3).data()) - 1) / 10 + 1] = true;
  }
  for (unsigned int i = 0; i < totalNumberOfBunches_; ++i) {
    if (bunchSelection_ == "CentralBunchesInTrain") {
      if (i == 0)
        validBunchArray_[i] = filledBunchArray[totalNumberOfBunches_ - 1] &&
                              filledBunchArray[i] && filledBunchArray[i + 1];
      else if (i == totalNumberOfBunches_ - 1)
        validBunchArray_[i] = filledBunchArray[i - 1] && filledBunchArray[i] &&
                              filledBunchArray[0];
      else
        validBunchArray_[i] = filledBunchArray[i - 1] && filledBunchArray[i] &&
                              filledBunchArray[i + 1];
    } else if (bunchSelection_ == "FirstBunchInTrain") {
      if (i == 0)
        validBunchArray_[i] =
            filledBunchArray[i] && !filledBunchArray[totalNumberOfBunches_ - 1];
      else
        validBunchArray_[i] = filledBunchArray[i] && !filledBunchArray[i - 1];
    } else if (bunchSelection_ == "LastBunchInTrain") {
      if (i == totalNumberOfBunches_ - 1)
        validBunchArray_[i] = filledBunchArray[i] && !filledBunchArray[0];
      else
        validBunchArray_[i] = filledBunchArray[i] && !filledBunchArray[i + 1];
    } else if (bunchSelection_ == "FilledBunches")
      validBunchArray_[i] = filledBunchArray[i];
  }
}

void EfficiencyTool_2018::endJob() {
  outputFile_ = new TFile(outputFileName_.data(), "RECREATE");
  std::cout << "Saving output in: " << outputFile_->GetName() << std::endl;
  h1BunchCrossing_->Write();
  h1CrossingAngle_->Write();
  for (const auto &rpId : romanPotIdVector_) {
    uint32_t arm = rpId.arm();
    uint32_t rp = rpId.rp();
    uint32_t station = rpId.station();
    if (station == 2) {
      mapXbins = mapXbins_st2;
      mapXmin = mapXmin_st2;
      mapXmax = mapXmax_st2;
      mapYbins = mapYbins_st2;
      mapYmin = mapYmin_st2;
      mapYmax = mapYmax_st2;
      fitXmin = fitXmin_st2;
      fitXmax = fitXmax_st2;
    } else {
      mapXbins = mapXbins_st0;
      mapXmin = mapXmin_st0;
      mapXmax = mapXmax_st0;
      mapYbins = mapYbins_st0;
      mapYmin = mapYmin_st0;
      mapYmax = mapYmax_st0;
      fitXmin = fitXmin_st0;
      fitXmax = fitXmax_st0;
    }

    std::string romanPotFolderName = Form("Arm%i_st%i_rp%i", arm, station, rp);
    std::string romanPotBinShiftFolderName =
        Form("Arm%i_st%i_rp%i/BinShift", arm, station, rp);
    // std::cout << "Creating directory for: " << romanPotFolderName <<
    // std::endl;

    outputFile_->mkdir(romanPotFolderName.data());
    outputFile_->cd(romanPotFolderName.data());
    h2TrackHitDistribution_[rpId]->Write();
    if (supplementaryPlots) {
      outputFile_->mkdir(romanPotBinShiftFolderName.data());
      outputFile_->cd(romanPotBinShiftFolderName.data());
      for (auto binShift : binShifts_) {
        h2TrackHitDistributionBinShift_[binShift][rpId]->Write();
        if (station == 0) {
          h2TrackHitDistributionBinShift_rotated[binShift][rpId]->Write();
        }
      }
      outputFile_->cd(romanPotFolderName.data());
    }

    h23PointsTrackHitDistribution_[rpId]->Write();
    // h2X0Correlation_[rpId]->Write();
    // h2Y0Correlation_[rpId]->Write();
    if (isCorrelationPlotEnabled) {
      // g1X0Correlation_[rpId]->Write();
      // g1Y0Correlation_[rpId]->Write();
    }
  }

  for (const auto &detId : detectorIdVector_) {
    uint32_t arm = detId.arm();
    uint32_t rp = detId.rp();
    uint32_t station = detId.station();
    uint32_t plane = detId.plane();
    std::string planeFolderName =
        Form("Arm%i_st%i_rp%i/Arm%i_st%i_rp%i_pl%i", arm, station, rp, arm,
             station, rp, plane);
    // std::cout << "Creating directory for: " << planeFolderName << std::endl;
    outputFile_->mkdir(planeFolderName.data());
    // std::cout << "Created directory for: " << planeFolderName << std::endl;
    outputFile_->cd(planeFolderName.data());

    h2ModuleHitMap_[detId]->Write();
    h2EfficiencyMap_[detId]->Divide(h2AuxEfficiencyMap_[detId],
                                    h2EfficiencyNormalizationMap_[detId], 1.,
                                    1., "B");
    h2EfficiencyMap_[detId]->SetMaximum(1.);
    h2EfficiencyMap_[detId]->Write();

    if (station == 0) {
      h2EfficiencyMap_rotated[detId]->Divide(
          h2AuxEfficiencyMap_rotated[detId],
          h2EfficiencyNormalizationMap_rotated[detId], 1., 1., "B");
      h2EfficiencyMap_rotated[detId]->SetMaximum(1.);
      h2EfficiencyMap_rotated[detId]->Write();
      if (supplementaryPlots)
        h2ModuleHitMap_rotated[detId]->Write();
    }
  }

  for (const auto &rpId : romanPotIdVector_) {
    uint32_t arm = rpId.arm();
    uint32_t rp = rpId.rp();
    uint32_t station = rpId.station();
    if (station == 2) {
      mapXbins = mapXbins_st2;
      mapXmin = mapXmin_st2;
      mapXmax = mapXmax_st2;
      mapYbins = mapYbins_st2;
      mapYmin = mapYmin_st2;
      mapYmax = mapYmax_st2;
      fitXmin = fitXmin_st2;
      fitXmax = fitXmax_st2;
    } else {
      mapXbins = mapXbins_st0;
      mapXmin = mapXmin_st0;
      mapXmax = mapXmax_st0;
      mapYbins = mapYbins_st0;
      mapYmin = mapYmin_st0;
      mapYmax = mapYmax_st0;
      fitXmin = fitXmin_st0;
      fitXmax = fitXmax_st0;
    }

    std::string romanPotFolderName = Form("Arm%i_st%i_rp%i", arm, station, rp);
    outputFile_->cd(romanPotFolderName.data());
    for (int xBin = 1; xBin <= mapXbins; ++xBin) {
      for (int yBin = 1; yBin <= mapYbins; ++yBin) {
        std::map<uint32_t, float> planeEfficiency;
        std::map<uint32_t, float> planeEfficiencyError;
        std::map<uint32_t, float> planeEfficiency_rotated;
        std::map<uint32_t, float> planeEfficiencyError_rotated;

        for (const auto &plane : listOfPlanes_) {
          CTPPSPixelDetId planeId = CTPPSPixelDetId(rpId);
          planeId.setPlane(plane);
          planeEfficiency[plane] =
              h2EfficiencyMap_[planeId]->GetBinContent(xBin, yBin);
          planeEfficiencyError[plane] =
              h2EfficiencyMap_[planeId]->GetBinError(xBin, yBin);
          if (station == 0) {
            planeEfficiency_rotated[plane] =
                h2EfficiencyMap_rotated[planeId]->GetBinContent(xBin, yBin);
            planeEfficiencyError_rotated[plane] =
                h2EfficiencyMap_rotated[planeId]->GetBinError(xBin, yBin);
          }

          // if(plane == 0){
          // 	planeEfficiency[plane] = 1.;
          // 	if(station==0) planeEfficiency_rotated[plane] = 1.;
          // }
          // if(plane == 1){
          // 	planeEfficiency[plane] = 1.;
          // 	if(station==0) planeEfficiency_rotated[plane] = 1.;
          // }
          // if(plane == 2){
          // 	planeEfficiency[plane] = 1.;
          // 	if(station==0) planeEfficiency_rotated[plane] = 1.;
          // }
          // if(plane == 3){
          // 	planeEfficiency[plane] = 1.;
          // 	if(station==0) planeEfficiency_rotated[plane] = 1.;
          // }
          // if(plane == 4){
          // 	planeEfficiency[plane] = 1.;
          // 	if(station==0) planeEfficiency_rotated[plane] = 1.;
          // }
          // if(plane == 5){
          // 	planeEfficiency[plane] = 0.;
          // 	if(station==0) planeEfficiency_rotated[plane] = 0.;
          // }
        }
        float efficiency = probabilityCalculation(planeEfficiency);
        float efficiencyError =
            errorCalculation(planeEfficiency, planeEfficiencyError);
        h2TrackEfficiencyMap_[rpId]->SetBinContent(xBin, yBin, efficiency);
        h2TrackEfficiencyMap_[rpId]->SetBinError(xBin, yBin, efficiencyError);
        h2TrackEfficiencyErrorMap_[rpId]->SetBinContent(xBin, yBin,
                                                        efficiencyError);
        if (station == 0) {
          float efficiency_rotated =
              probabilityCalculation(planeEfficiency_rotated);
          float efficiencyError_rotated = errorCalculation(
              planeEfficiency_rotated, planeEfficiencyError_rotated);
          h2TrackEfficiencyMap_rotated[rpId]->SetBinContent(xBin, yBin,
                                                            efficiency_rotated);
          h2TrackEfficiencyMap_rotated[rpId]->SetBinError(
              xBin, yBin, efficiencyError_rotated);
          h2TrackEfficiencyErrorMap_rotated[rpId]->SetBinContent(
              xBin, yBin, efficiencyError_rotated);
        }
      }
    }
    h2TrackEfficiencyMap_[rpId]->Write();
    h2TrackEfficiencyErrorMap_[rpId]->Write();
    if (supplementaryPlots) {
      outputFile_->mkdir((romanPotFolderName + "/ResolutionHistograms").data());
      // std::cout << (romanPotFolderName+"/ResolutionHistograms").data() <<
      // std::endl;
      outputFile_->cd((romanPotFolderName + "/ResolutionHistograms").data());
      for (int nPlanes = 3; nPlanes <= 6; nPlanes++) {
        for (int numberOfCls = 0; numberOfCls <= nPlanes; numberOfCls++) {
          h1X0Sigma[rpId][std::pair(nPlanes, numberOfCls)]->Write();
          h1Y0Sigma[rpId][std::pair(nPlanes, numberOfCls)]->Write();
          h1TxSigma[rpId][std::pair(nPlanes, numberOfCls)]->Write();
          h1TySigma[rpId][std::pair(nPlanes, numberOfCls)]->Write();
        }
      }

      outputFile_->cd(romanPotFolderName.data());

      h1NumberOfTracks_[rpId]->Write();
      h2AvgPlanesUsed_[rpId]->Divide(h2TrackHitDistribution_[rpId]);
      h2AvgPlanesUsed_[rpId]->Write();
      h1PlanesUsed_[rpId]->Write();
      h1ChiSquaredOverNDF_[rpId]->Write();
      h1ConsecutivePlanes_[rpId]->Write();
    }
    if (station == 0) {
      h2TrackEfficiencyMap_rotated[rpId]->Write();
      h2TrackEfficiencyErrorMap_rotated[rpId]->Write();
      if (supplementaryPlots) {
        h2TrackHitDistribution_rotated[rpId]->Write();
        h2AvgPlanesUsed_rotated[rpId]->Divide(
            h2TrackHitDistribution_rotated[rpId]);
        h2AvgPlanesUsed_rotated[rpId]->Write();
      }
    }
    if (supplementaryPlots && station != 0) {
      TObjArray slices;
      TF1 *fGaus = new TF1("fGaus", "gaus", mapYmin, mapYmax);
      h2TrackHitDistribution_[rpId]->FitSlicesY(fGaus, 1, mapXbins, 0, "QNG3",
                                                &slices);
      delete fGaus;
      ((TH1D *)slices[0])->Write();
      ((TH1D *)slices[1])->Write();
      ((TH1D *)slices[2])->Write();
      ((TH1D *)slices[3])->Write();
    }
    if (supplementaryPlots && station == 0) {
      TObjArray slices;
      TF1 *fGaus = new TF1("fGaus", "gaus", mapYmin, mapYmax);
      h2TrackHitDistribution_rotated[rpId]->FitSlicesY(fGaus, 1, mapXbins, 0,
                                                       "QNG3", &slices);
      delete fGaus;
      ((TH1D *)slices[0])->Write();
      ((TH1D *)slices[1])->Write();
      ((TH1D *)slices[2])->Write();
      ((TH1D *)slices[3])->Write();
    }
  }

  if (isCorrelationPlotEnabled)
    std::cout << "ATTENTION: Remember to insert the fitting parameters in the "
                 "python configuration"
              << std::endl;
  outputFile_->Close();
}

void EfficiencyTool_2018::fillDescriptions(
    edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

// This function produces all the possible plane combinations extracting
// numberToExtract planes over numberOfPlanes planes
void EfficiencyTool_2018::getPlaneCombinations(
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

float EfficiencyTool_2018::probabilityNplanesBlind(
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

// Calculates the partial derivative of the rp efficiency with respect to the
// efficiency of a certain plane when extracting numberToExtract planes
float EfficiencyTool_2018::efficiencyPartialDerivativewrtPlane(
    uint32_t plane, const std::vector<uint32_t> &inputPlaneList,
    int numberToExtract, const std::map<unsigned, float> &planeEfficiency) {
  std::vector<uint32_t> modifiedInputPlaneList = inputPlaneList;
  modifiedInputPlaneList.erase(std::find(modifiedInputPlaneList.begin(),
                                         modifiedInputPlaneList.end(), plane));
  float partialDerivative = 0.;
  if (numberToExtract > 0 && numberToExtract < 6) {
    partialDerivative =
        -probabilityNplanesBlind(modifiedInputPlaneList, numberToExtract,
                                 planeEfficiency) +
        probabilityNplanesBlind(modifiedInputPlaneList, numberToExtract - 1,
                                planeEfficiency);
  } else {
    if (numberToExtract == 6) {
      partialDerivative = probabilityNplanesBlind(
          modifiedInputPlaneList, numberToExtract - 1, planeEfficiency);
    } else {
      partialDerivative = -probabilityNplanesBlind(
          modifiedInputPlaneList, numberToExtract, planeEfficiency);
    }
  }
  return partialDerivative;
}

float EfficiencyTool_2018::errorCalculation(
    const std::map<uint32_t, float> &planeEfficiency,
    const std::map<uint32_t, float> &planeEfficiencyError) {
  int minNumberOfBlindPlanes = 3;
  int maxNumberOfBlindPlanes = listOfPlanes_.size();
  float rpEfficiencySquareError = 0.;
  for (const auto &plane : listOfPlanes_) {
    float partialDerivative = 0.;
    for (uint32_t i = (uint32_t)minNumberOfBlindPlanes;
         i <= (uint32_t)maxNumberOfBlindPlanes; i++) {
      partialDerivative += efficiencyPartialDerivativewrtPlane(
          plane, listOfPlanes_, i, planeEfficiency);
    }
    rpEfficiencySquareError += partialDerivative * partialDerivative *
                               planeEfficiencyError.at(plane) *
                               planeEfficiencyError.at(plane);
  }
  return TMath::Sqrt(rpEfficiencySquareError);
}

float EfficiencyTool_2018::probabilityCalculation(
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

bool EfficiencyTool_2018::Cut(CTPPSPixelLocalTrack track, int arm,
                              int station) {
  uint32_t ndf = 2 * track.numberOfPointsUsedForFit() - 4;
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

  double maxTx = 0.02;
  double maxTy = 0.02;
  double maxChi2 = TMath::ChisquareQuantile(0.95, ndf);

  if (TMath::Abs(track.tx()) > maxTx || TMath::Abs(track.ty()) > maxTy ||
      track.chiSquaredOverNDF() * ndf > maxChi2 ||
      track.numberOfPointsUsedForFit() < minNumberOfPlanesForTrack_ ||
      track.numberOfPointsUsedForFit() > maxNumberOfPlanesForTrack_ ||
      y > fiducialYHigh_[std::pair<int, int>(arm, station)] ||
      y < fiducialYLow_[std::pair<int, int>(arm, station)] ||
      x < fiducialXLow_[std::pair<int, int>(arm, station)])
    return true;
  else
    return false;
}

// define this as a plug-in
DEFINE_FWK_MODULE(EfficiencyTool_2018);