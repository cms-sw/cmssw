// -*- C++ -*-
//
// Package:    RecoPPS/RPixEfficiencyTools
// Class:      EfficiencyTool_2018DQMWorker
//
/**\class EfficiencyTool_2018DQMWorker EfficiencyTool_2018DQMWorker.cc
 RecoPPS/RPixEfficiencyTools/plugins/EfficiencyTool_2018DQMWorker.cc

 Description: [one line class summary]

 Implementation:
                 [Notes on implementation]
*/
//
// Original Author:  Andrea Bellora
//         Created:  Wed, 22 Aug 2017 09:55:05 GMT
//
//
#include <string>
#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <exception>
#include <fstream>
#include <memory>
#include <set>

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/RegexMatch.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "CondTools/RunInfo/interface/LHCInfoCombined.h"
#include "HLTrigger/HLTcore/interface/HLTPrescaleProvider.h"

#include "DataFormats/Common/interface/DetSetVector.h"

#include "DataFormats/CTPPSDetId/interface/CTPPSPixelDetId.h"
#include "DataFormats/CTPPSReco/interface/CTPPSPixelLocalTrack.h"

#include "Geometry/VeryForwardGeometryBuilder/interface/CTPPSGeometry.h"
#include "Geometry/Records/interface/VeryForwardRealGeometryRecord.h"


//INTERPOT
#include "DataFormats/CTPPSReco/interface/CTPPSLocalTrackLite.h"
#include "DataFormats/CTPPSReco/interface/CTPPSPixelLocalTrack.h"
#include "DataFormats/CTPPSReco/interface/CTPPSPixelRecHit.h"

#include "DataFormats/ProtonReco/interface/ForwardProton.h"
#include "DataFormats/ProtonReco/interface/ForwardProtonFwd.h"

template <class T>
T get_min(const std::vector<T> &vec) {
  auto result = std::min_element(vec.begin(), vec.end());
  if (result == vec.end()) {
    for (long unsigned int i = 0; i < vec.size(); i++) {
      std::cout << vec[i] << " ";
    }
    std::cout << std::endl;
  }
  return *result;
}

template <class T>
T get_max(const std::vector<T> &vec) {
  auto result = std::max_element(vec.begin(), vec.end());
  if (result == vec.end()) {
    for (long unsigned int i = 0; i < vec.size(); i++) {
      std::cout << vec[i] << " ";
    }
    std::cout << std::endl;
  }
  return *result;
}


class EfficiencyTool_2018DQMWorker : public DQMEDAnalyzer {
public:
  explicit EfficiencyTool_2018DQMWorker(const edm::ParameterSet &);
  ~EfficiencyTool_2018DQMWorker();
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

protected:
  virtual void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  virtual void dqmBeginRun(edm::Run const &, edm::EventSetup const &) override;

private:
  virtual void analyze(const edm::Event &, const edm::EventSetup &) override;
  void endJob();
  virtual void initialize();
  void printMeans();

  // Computes the probability of having numberToExtract inefficient planes
  float probabilityNplanesBlind(const std::vector<uint32_t> &inputPlaneList,
                                int numberToExtract,
                                const std::map<unsigned, float> &planeEfficiency);

  // Computes all the combination of planes with numberToExtract planes
  // extracted
  void getPlaneCombinations(const std::vector<uint32_t> &inputPlaneList,
                            uint32_t numberToExtract,
                            std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> &planesExtractedAndNot);

  // Computes the efficiency, given certain plane efficiency values
  float probabilityCalculation(const std::map<uint32_t, float> &planeEfficiency);

  // Obsolete and not correct error propagation
  float errorCalculation(const std::map<uint32_t, float> &planeEfficiency,
                         const std::map<uint32_t, float> &planeEfficiencyError);
  float efficiencyPartialDerivativewrtPlane(uint32_t plane,
                                            const std::vector<uint32_t> &inputPlaneList,
                                            int numberToExtract,
                                            const std::map<unsigned, float> &planeEfficiency);

  // Return true if a track should be discarded
  bool Cut(CTPPSPixelLocalTrack track, int arm, int station);
  void setGlobalBinSizes(CTPPSPixelDetId &rpId);

  edm::EDGetTokenT<edm::DetSetVector<CTPPSPixelLocalTrack>> pixelLocalTrackToken_;
  edm::EDGetTokenT<edm::DetSetVector<CTPPSPixelRecHit>> pixelRecHitToken_;
  edm::ESGetToken<CTPPSGeometry, VeryForwardRealGeometryRecord> geomEsToken_;
  const edm::ESGetToken<LHCInfo, LHCInfoRcd> lhcInfoToken_;
  const edm::ESGetToken<LHCInfoPerLS, LHCInfoPerLSRcd> lhcInfoPerLSToken_;
  const edm::ESGetToken<LHCInfoPerFill, LHCInfoPerFillRcd> lhcInfoPerFillToken_;
  const bool useNewLHCInfo_;
  
  bool isCorrelationPlotEnabled_;
  bool supplementaryPlots_;
  int minNumberOfPlanesForEfficiency_;
  int minNumberOfPlanesForTrack_;
  int maxNumberOfPlanesForTrack_;
  int minTracksPerEvent_;
  int maxTracksPerEvent;

  static const unsigned int totalNumberOfBunches_ = 3564;
  std::string bunchSelection_;
  std::string bunchListFileName_;
  bool validBunchArray_[totalNumberOfBunches_];

  MonitorElement *h1BunchCrossing_;
  MonitorElement *h1CrossingAngle_;
  std::map<CTPPSPixelDetId, MonitorElement *> h2ModuleHitMap_;
  std::map<CTPPSPixelDetId, MonitorElement *> h2EfficiencyMap_;
  std::map<CTPPSPixelDetId, MonitorElement *> h2AuxEfficiencyMap_;
  std::map<CTPPSPixelDetId, MonitorElement *> h2EfficiencyNormalizationMap_;
  std::map<CTPPSPixelDetId, MonitorElement *> h2TrackHitDistribution_;
  std::map<CTPPSPixelDetId, MonitorElement *> h23PointsTrackHitDistribution_;
  std::map<CTPPSPixelDetId, MonitorElement *> h2TrackEfficiencyMap_;
  std::map<CTPPSPixelDetId, MonitorElement *> h2TrackEfficiencyErrorMap_;
  std::map<CTPPSPixelDetId, MonitorElement *> h1NumberOfTracks_;
  std::map<CTPPSPixelDetId, MonitorElement *> h2X0Correlation_;
  std::map<CTPPSPixelDetId, MonitorElement *> h2Y0Correlation_;
  std::map<CTPPSPixelDetId, MonitorElement *> h2AvgPlanesUsed_;
  std::map<CTPPSPixelDetId, MonitorElement *> h1PlanesUsed_;
  std::map<CTPPSPixelDetId, MonitorElement *> h1ChiSquaredOverNDF_;
  std::map<CTPPSPixelDetId, MonitorElement *> h1ConsecutivePlanes_;

  std::map<CTPPSPixelDetId, MonitorElement *> h2ModuleHitMap_rotated;
  std::map<CTPPSPixelDetId, MonitorElement *> h2EfficiencyMap_rotated;
  std::map<CTPPSPixelDetId, MonitorElement *> h2AuxEfficiencyMap_rotated;
  std::map<CTPPSPixelDetId, MonitorElement *> h2EfficiencyNormalizationMap_rotated;
  std::map<CTPPSPixelDetId, MonitorElement *> h2TrackHitDistribution_rotated;
  std::map<CTPPSPixelDetId, MonitorElement *> h2TrackEfficiencyMap_rotated;
  std::map<CTPPSPixelDetId, MonitorElement *> h2TrackEfficiencyErrorMap_rotated;
  std::map<CTPPSPixelDetId, MonitorElement *> h2AvgPlanesUsed_rotated;

  // Resolution histograms
  std::map<CTPPSPixelDetId, std::map<std::pair<int, int>, MonitorElement *>>
      h1X0Sigma;  
  std::map<CTPPSPixelDetId, std::map<std::pair<int, int>, MonitorElement *>>
      h1Y0Sigma;  
  std::map<CTPPSPixelDetId, std::map<std::pair<int, int>, MonitorElement *>>
      h1TxSigma;  
  std::map<CTPPSPixelDetId, std::map<std::pair<int, int>, MonitorElement *>>
      h1TySigma;  

  // Prescale plots
  MonitorElement *h1L1Prescale_;
  MonitorElement *h1HLTPrescale_;

  std::vector<CTPPSPixelDetId> detectorIdVector_;

  double detectorTiltAngle_;
  double detectorRotationAngle_;

  int binGroupingX_ = 1;
  int binGroupingY_ = 1;
  int mapXbins = 200 / binGroupingX_;
  float mapXmin_;
  float mapXmax_;
  int mapYbins = 240 / binGroupingY_;
  float mapYmin = -12.;
  float mapYmax = 12.;
  float fitXmin = 45.;
  float fitXmax = 58.;

  std::vector<uint32_t> listOfPlanes_;
  std::map<CTPPSPixelDetId, int> binAlignmentParameters = {{CTPPSPixelDetId(0, 0, 3), 0},
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

  //INTERPOT
  bool Cut(CTPPSLocalTrackLite track);
  float Aperture(Float_t xangle, Int_t arm, TString era);

  std::map<CTPPSPixelDetId, double> mapXbin_changeCoordinate = {{CTPPSPixelDetId(0, 0, 3), 13},
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

  // output histograms
  std::map<CTPPSPixelDetId, MonitorElement *> h2ProtonHitExpectedDistribution_;
  std::map<CTPPSPixelDetId, MonitorElement *> h2AuxProtonHitDistributionWithNoMultiRP_;
  std::map<CTPPSPixelDetId, MonitorElement *> h2InterPotEfficiencyMap_;
  std::map<CTPPSPixelDetId, MonitorElement *> h2InterPotEfficiencyMapMultiRP_;
  std::map<CTPPSPixelDetId, MonitorElement *> h1AuxXi_;
  std::map<CTPPSPixelDetId, MonitorElement *> h1InterPotEfficiencyVsXi_;
  std::map<CTPPSPixelDetId, MonitorElement *> h1DeltaXiMatch_;
  std::map<CTPPSPixelDetId, MonitorElement *> h1DeltaYMatch_;
  std::map<CTPPSPixelDetId, MonitorElement *> h1TxMatch_;
  std::map<CTPPSPixelDetId, MonitorElement *> h1TyMatch_;
  std::map<CTPPSPixelDetId, MonitorElement *> h1ProtonsInProbePotWhenNoMatchFound_;
  std::map<CTPPSPixelDetId, MonitorElement *> h2TxCorrelationMatch_;
  std::map<CTPPSPixelDetId, MonitorElement *> h2TyCorrelationMatch_;
  std::map<CTPPSPixelDetId, MonitorElement *> h2XCorrelationMatch_;
  std::map<CTPPSPixelDetId, MonitorElement *> h2YCorrelationMatch_;

  std::map<CTPPSPixelDetId, MonitorElement *> h1TrackMux_;

  //WARNING! WE ARE NOT SETTING trackMux_  VALUES AS WE ARE CREATING HISTOGRAM INDEPENDTLY OF THE FACT IF THERE IS ANY DATA TO FILL ON IT! (line 381 on InterpotEfficiency_2018.cc might not work)
  std::map<CTPPSPixelDetId, uint32_t> trackMux_;

  int recoInfoCut_;

  double mapXbinSize_small = (mapXmax_ - mapXmin_) / mapXbins;
  double mapXbinSize_large = (mapXmax_ - mapXmin_) / mapXbins * 2;

  uint32_t maxTracksInTagPot = 99;
  uint32_t minTracksInTagPot = 0;
  uint32_t maxTracksInProbePot = 99;
  uint32_t minTracksInProbePot = 0;
  double maxChi2Prob_;
  bool debug_ = false;
  bool fancyBinning_ = false;

  // z position of the pots (mm)
  std::map<CTPPSDetId, double> Z = {
      {CTPPSDetId(3, 0, 0, 3), -212550},  // strips, arm0, station0, rp3
      {CTPPSDetId(3, 1, 0, 3), 212550},   // strips, arm1, station0, rp3
      {CTPPSDetId(4, 0, 2, 3), -219550},  // pixels, arm0, station2, rp3
      {CTPPSDetId(4, 1, 2, 3), 219550}    // pixels, arm1, station2, rp3
  };

  edm::EDGetTokenT<reco::ForwardProtonCollection> protonsToken_;
  edm::EDGetTokenT<reco::ForwardProtonCollection> multiRP_protonsToken_;

  // Prescale parameters
  bool usePrescales_;
  std::string processName_, triggerPattern_, hltName_;
  HLTPrescaleProvider hltPrescaleProvider_;
};

EfficiencyTool_2018DQMWorker::EfficiencyTool_2018DQMWorker(const edm::ParameterSet &iConfig)
    : geomEsToken_(esConsumes<edm::Transition::BeginRun>()),
      lhcInfoToken_(esConsumes(edm::ESInputTag("", iConfig.getParameter<std::string>("lhcInfoLabel")))),
      lhcInfoPerLSToken_(esConsumes(edm::ESInputTag("", iConfig.getParameter<std::string>("lhcInfoPerLSLabel")))),
      lhcInfoPerFillToken_(esConsumes(edm::ESInputTag("", iConfig.getParameter<std::string>("lhcInfoPerFillLabel")))),
      useNewLHCInfo_(iConfig.getParameter<bool>("useNewLHCInfo")),      
      hltPrescaleProvider_(iConfig, consumesCollector(), *this)
 {
  pixelLocalTrackToken_ = consumes<edm::DetSetVector<CTPPSPixelLocalTrack>>(iConfig.getUntrackedParameter<edm::InputTag>("tagPixelLocalTracks"));
  minNumberOfPlanesForEfficiency_ = iConfig.getParameter<int>("minNumberOfPlanesForEfficiency");
  minNumberOfPlanesForTrack_ = iConfig.getParameter<int>("minNumberOfPlanesForTrack");
  maxNumberOfPlanesForTrack_ = iConfig.getParameter<int>("maxNumberOfPlanesForTrack");
  isCorrelationPlotEnabled_ = iConfig.getParameter<bool>("isCorrelationPlotEnabled");
  minTracksPerEvent_ = iConfig.getParameter<int>("minTracksPerEvent");
  maxTracksPerEvent = iConfig.getParameter<int>("maxTracksPerEvent");
  supplementaryPlots_ = iConfig.getParameter<bool>("supplementaryPlots");
  bunchSelection_ = iConfig.getUntrackedParameter<std::string>("bunchSelection");
  bunchListFileName_ = iConfig.getUntrackedParameter<std::string>("bunchListFileName");
  binGroupingX_ = iConfig.getUntrackedParameter<int>("binGroupingX");
  binGroupingY_ = iConfig.getUntrackedParameter<int>("binGroupingY");
  fiducialXLowVector_ = iConfig.getUntrackedParameter<std::vector<double>>("fiducialXLow");
  fiducialYLowVector_ = iConfig.getUntrackedParameter<std::vector<double>>("fiducialYLow");
  fiducialYHighVector_ = iConfig.getUntrackedParameter<std::vector<double>>("fiducialYHigh");
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
  detectorTiltAngle_ = iConfig.getUntrackedParameter<double>("detectorTiltAngle");
  mapXmin_ = 0. * TMath::Cos(detectorTiltAngle_ / 180. * TMath::Pi());
  mapXmax_ = 30. * TMath::Cos(detectorTiltAngle_ / 180. * TMath::Pi());  //18.4 is default angle
  detectorRotationAngle_ = iConfig.getUntrackedParameter<double>("detectorRotationAngle");

  initialize();
  
  //INTERPOT
  protonsToken_ = consumes<reco::ForwardProtonCollection>(iConfig.getUntrackedParameter<edm::InputTag>("tagProtonsSingleRP"));
  multiRP_protonsToken_ = consumes<reco::ForwardProtonCollection>(iConfig.getUntrackedParameter<edm::InputTag>("tagProtonsMultiRP"));
  maxTracksInTagPot = iConfig.getUntrackedParameter<int>("maxTracksInTagPot");
  minTracksInTagPot = iConfig.getUntrackedParameter<int>("minTracksInTagPot");
  maxTracksInProbePot = iConfig.getUntrackedParameter<int>("maxTracksInProbePot");
  minTracksInProbePot = iConfig.getUntrackedParameter<int>("minTracksInProbePot");
  maxChi2Prob_ = iConfig.getUntrackedParameter<double>("maxChi2Prob");

  binGroupingX_ = iConfig.getUntrackedParameter<int>("binGroupingX");  // UNUSED!
  binGroupingY_ = iConfig.getUntrackedParameter<int>("binGroupingY");  // UNUSED!
  recoInfoCut_ = iConfig.getUntrackedParameter<int>("recoInfo");
  
  // Compute binning arrays
  for (auto detID_and_coordinate : mapXbin_changeCoordinate) {
    CTPPSPixelDetId detId = detID_and_coordinate.first;
    int nBinsX_small = (int)((detID_and_coordinate.second - mapXmin_) / mapXbinSize_small);
    mapXbin_changeCoordinate[detId] = mapXmin_ + nBinsX_small * mapXbinSize_small;
    int nBinsX_large = (int)((mapXmax_ - detID_and_coordinate.second) / mapXbinSize_large);
    nBinsX_total[detId] = nBinsX_small + nBinsX_large;
    for (int i = 0; i <= nBinsX_total[detId]; i++) {
      if (i <= nBinsX_small)
        xBinEdges[detId].push_back(i * mapXbinSize_small);
      else
        xBinEdges[detId].push_back(nBinsX_small * mapXbinSize_small + (i - nBinsX_small) * mapXbinSize_large);
    }
  }
  debug_ = iConfig.getUntrackedParameter<bool>("debug");

  // Prescale parameters
  usePrescales_ = iConfig.getParameter<bool>("usePrescales");
  processName_ = iConfig.getParameter<std::string>("processName");
  triggerPattern_ = iConfig.getParameter<std::string>("triggerPattern");
}

EfficiencyTool_2018DQMWorker::~EfficiencyTool_2018DQMWorker() {}

void EfficiencyTool_2018DQMWorker::bookHistograms(DQMStore::IBooker &ibooker,
                                                  edm::Run const &run,
                                                  edm::EventSetup const &eventSetup) {
  ibooker.cd();
  h1BunchCrossing_ =
      ibooker.book1DD("h1BunchCrossing", "h1BunchCrossing", totalNumberOfBunches_, 0., totalNumberOfBunches_);
  h1CrossingAngle_ = ibooker.book1DD("h1CrossingAngle", "h1CrossingAngle", 70, 100., 170);
  // Assume max. 1000 LSs
  h1L1Prescale_ =
      ibooker.bookProfile("h1L1Prescale", "h1L1Prescale;LS;Prescale", 3000, 0., 3000, 1, 1E4, "");
  h1HLTPrescale_ =
      ibooker.bookProfile("h1HLTPrescale", "h1HLTPrescale;LS;Prescale", 3000, 0., 3000, 1, 1E4, "");

  const auto &geom = eventSetup.getData(geomEsToken_);

  std::set<uint32_t> planesSet;
  for (auto it = geom.beginSensor(); it != geom.endSensor(); ++it) {
    if (!CTPPSPixelDetId::check(it->first))
      continue;
    
    const CTPPSPixelDetId detId(it->first);
    uint32_t arm = detId.arm();
    uint32_t rp = detId.rp();
    uint32_t station = detId.station();
    uint32_t plane = detId.plane();

    planesSet.insert(plane);

    std::string romanPotFolderName = Form("Arm%i/st%i/rp%i", arm, station, rp);
    ibooker.setCurrentFolder(romanPotFolderName);
    
    CTPPSPixelDetId rpId(arm, station, rp);
    
    setGlobalBinSizes(rpId);
    h2TrackHitDistribution_[rpId] =
        ibooker.book2DD(Form("h2TrackHitDistribution_arm%i_st%i_rp%i", arm, station, rp),
                        Form("h2TrackHitDistribution_arm%i_st%i_rp%i;x (mm);y (mm)", arm, station, rp),
                        mapXbins,
                        mapXmin_,
                        mapXmax_,
                        mapYbins,
                        mapYmin,
                        mapYmax);

    h23PointsTrackHitDistribution_[rpId] =
        ibooker.book2DD(Form("h23PointsTrackHitDistribution_arm%i_st%i_rp%i", arm, station, rp),
                        Form("h23PointsTrackHitDistribution_arm%i_st%i_rp%i;x (mm);y (mm)", arm, station, rp),
                        mapXbins,
                        mapXmin_,
                        mapXmax_,
                        mapYbins,
                        mapYmin,
                        mapYmax);

    h2TrackEfficiencyMap_[rpId] =
        ibooker.book2DD(Form("h2TrackEfficiencyMap_arm%i_st%i_rp%i", arm, station, rp),
                        Form("h2TrackEfficiencyMap_arm%i_st%i_rp%i; x (mm); y (mm)", arm, station, rp),
                        mapXbins,
                        mapXmin_,
                        mapXmax_,
                        mapYbins,
                        mapYmin,
                        mapYmax);

    h2TrackEfficiencyErrorMap_[rpId] =
        ibooker.book2DD(Form("h2TrackEfficiencyErrorMap_arm%i_st%i_rp%i", arm, station, rp),
                        Form("h2TrackEfficiencyErrorMap_arm%i_st%i_rp%i; x (mm); y (mm)", arm, station, rp),
                        mapXbins,
                        mapXmin_,
                        mapXmax_,
                        mapYbins,
                        mapYmin,
                        mapYmax);

    h1NumberOfTracks_[rpId] = ibooker.book1DD(Form("h1NumberOfTracks_arm%i_st%i_rp%i", arm, station, rp),
                                              Form("h1NumberOfTracks_arm%i_st%i_rp%i; Tracks;", arm, station, rp),
                                              16,
                                              -0.5,
                                              15.5);
    if (supplementaryPlots_) {
      h2AvgPlanesUsed_[rpId] =
          ibooker.book2DD(Form("h2AvgPlanesUsed_arm%i_st%i_rp%i", arm, station, rp),
                          Form("h2AvgPlanesUsed_arm%i_st%i_rp%i; x (mm); y (mm)", arm, station, rp),
                          mapXbins,
                          mapXmin_,
                          mapXmax_,
                          mapYbins,
                          mapYmin,
                          mapYmax);

      h1PlanesUsed_[rpId] = ibooker.book1DD(Form("h1PlanesUsed_arm%i_st%i_rp%i", arm, station, rp),
                                            Form("h1PlanesUsed_arm%i_st%i_rp%i; Planes", arm, station, rp),
                                            7,
                                            -0.5,
                                            6.5);
      h1ChiSquaredOverNDF_[rpId] =
          ibooker.book1DD(Form("h1ChiSquaredOverNDF_arm%i_st%i_rp%i", arm, station, rp),
                          Form("h1ChiSquaredOverNDF_arm%i_st%i_rp%i; Planes", arm, station, rp),
                          100,
                          0,
                          5);

      for (int nPlanes = 3; nPlanes <= 6; nPlanes++) {
        for (int numberOfCls = 0; numberOfCls <= nPlanes; numberOfCls++) {
          h1X0Sigma[rpId][std::pair(nPlanes, numberOfCls)] = ibooker.book1DD(
              Form("h1X0Sigma_arm%i_st%i_rp%i_nPlanes%i_nCls%i", arm, station, rp, nPlanes, numberOfCls),
              Form("h1X0Sigma_arm%i_st%i_rp%i_nPlanes%i_nCls%i; "
                   "#sigma_{x} (mm);",
                   arm,
                   station,
                   rp,
                   nPlanes,
                   numberOfCls),
              100,
              0,
              0.1);

          h1Y0Sigma[rpId][std::pair(nPlanes, numberOfCls)] = ibooker.book1DD(
              Form("h1Y0Sigma_arm%i_st%i_rp%i_nPlanes%i_nCls%i", arm, station, rp, nPlanes, numberOfCls),
              Form("h1Y0Sigma_arm%i_st%i_rp%i_nPlanes%i_nCls%i; "
                   "#sigma_{y} (mm);",
                   arm,
                   station,
                   rp,
                   nPlanes,
                   numberOfCls),
              100,
              0,
              0.1);

          h1TxSigma[rpId][std::pair(nPlanes, numberOfCls)] = ibooker.book1DD(
              Form("h1TxSigma_arm%i_st%i_rp%i_nPlanes%i_nCls%i", arm, station, rp, nPlanes, numberOfCls),
              Form("h1TxSigma_arm%i_st%i_rp%i_nPlanes%i_nCls%i; #sigma_{Tx};", arm, station, rp, nPlanes, numberOfCls),
              100,
              0,
              0.02);

          h1TySigma[rpId][std::pair(nPlanes, numberOfCls)] = ibooker.book1DD(
              Form("h1TySigma_arm%i_st%i_rp%i_nPlanes%i_nCls%i", arm, station, rp, nPlanes, numberOfCls),
              Form("h1TySigma_arm%i_st%i_rp%i_nPlanes%i_nCls%i; #sigma_{Ty};", arm, station, rp, nPlanes, numberOfCls),
              100,
              0,
              0.02);
        }
      }

      h1ConsecutivePlanes_[rpId] =
          ibooker.book1DD(Form("h1ConsecutivePlanes_arm%i_st%i_rp%i", arm, station, rp),
                          Form("h1ConsecutivePlanes_arm%i_st%i_rp%i; #sigma_{Ty};", arm, station, rp),
                          2,
                          0,
                          2);
      h1ConsecutivePlanes_[rpId]->getTH1D()->GetXaxis()->SetBinLabel(1, "Non-consecutive");
      h1ConsecutivePlanes_[rpId]->getTH1D()->GetXaxis()->SetBinLabel(2, "Consecutive");
    }

    if (station == 0) {
      h2TrackEfficiencyMap_rotated[rpId] =
          ibooker.book2DD(Form("h2TrackEfficiencyMap_rotated_arm%i_st%i_rp%i", arm, station, rp),
                          Form("h2TrackEfficiencyMap_rotated_arm%i_st%i_rp%i; x (mm); y (mm)", arm, station, rp),
                          mapXbins,
                          mapXmin_,
                          mapXmax_,
                          mapYbins,
                          mapYmin,
                          mapYmax);

      h2TrackEfficiencyErrorMap_rotated[rpId] =
          ibooker.book2DD(Form("h2TrackEfficiencyErrorMap_rotated_arm%i_st%i_rp%i", arm, station, rp),
                          Form("h2TrackEfficiencyErrorMap_rotated_arm%i_st%i_rp%i; "
                               "x (mm); y (mm)",
                               arm,
                               station,
                               rp),
                          mapXbins,
                          mapXmin_,
                          mapXmax_,
                          mapYbins,
                          mapYmin,
                          mapYmax);

      if (supplementaryPlots_) {
        h2TrackHitDistribution_rotated[rpId] =
            ibooker.book2DD(Form("h2TrackHitDistribution_rotated_arm%i_st%i_rp%i", arm, station, rp),
                            Form("h2TrackHitDistribution_rotated_arm%i_st%i_rp%i;x "
                                 "(mm);y (mm)",
                                 arm,
                                 station,
                                 rp),
                            mapXbins,
                            mapXmin_,
                            mapXmax_,
                            mapYbins,
                            mapYmin,
                            mapYmax);
        h2AvgPlanesUsed_rotated[rpId] =
            ibooker.book2DD(Form("h2AvgPlanesUsed_rotated_arm%i_st%i_rp%i", arm, station, rp),
                            Form("h2AvgPlanesUsed_rotated_arm%i_st%i_rp%i; x (mm); y (mm)", arm, station, rp),
                            mapXbins,
                            mapXmin_,
                            mapXmax_,
                            mapYbins,
                            mapYmin,
                            mapYmax);
      }
    }

    rpId = CTPPSPixelDetId(arm, station, rp, plane);

    h2ModuleHitMap_[rpId] =
        ibooker.book2DD(Form("h2ModuleHitMap_arm%i_st%i_rp%i_pl%i", arm, station, rp, plane),
                        Form("h2ModuleHitMap_arm%i_st%i_rp%i_pl%i; x (mm); y (mm)", arm, station, rp, plane),
                        mapXbins,
                        mapXmin_,
                        mapXmax_,
                        mapYbins,
                        mapYmin,
                        mapYmax);

    h2EfficiencyMap_[rpId] =
        ibooker.book2DD(Form("h2EfficiencyMap_arm%i_st%i_rp%i_pl%i", arm, station, rp, plane),
                        Form("h2EfficiencyMap_arm%i_st%i_rp%i_pl%i; x (mm); y (mm)", arm, station, rp, plane),
                        mapXbins,
                        mapXmin_,
                        mapXmax_,
                        mapYbins,
                        mapYmin,
                        mapYmax);

    h2AuxEfficiencyMap_[rpId] =
        ibooker.book2DD(Form("h2AuxEfficiencyMap_arm%i_st%i_rp%i_pl%i", arm, station, rp, plane),
                        Form("h2AuxEfficiencyMap_arm%i_st%i_rp%i_pl%i", arm, station, rp, plane),
                        mapXbins,
                        mapXmin_,
                        mapXmax_,
                        mapYbins,
                        mapYmin,
                        mapYmax);

    h2EfficiencyNormalizationMap_[rpId] =
        ibooker.book2DD(Form("h2EfficiencyNormalizationMap_arm%i_st%i_rp%i_pl%i", arm, station, rp, plane),
                        Form("h2EfficiencyNormalizationMap_arm%i_st%i_rp%i_pl%i", arm, station, rp, plane),
                        mapXbins,
                        mapXmin_,
                        mapXmax_,
                        mapYbins,
                        mapYmin,
                        mapYmax);

    if (station == 0) {
      h2EfficiencyMap_rotated[rpId] =
          ibooker.book2DD(Form("h2EfficiencyMap_rotated_arm%i_st%i_rp%i_pl%i", arm, station, rp, plane),
                          Form("h2EfficiencyMap_rotated_arm%i_st%i_rp%i_pl%i; x (mm); y  "
                               "(mm)",
                               arm,
                               station,
                               rp,
                               plane),
                          mapXbins,
                          mapXmin_,
                          mapXmax_,
                          mapYbins,
                          mapYmin,
                          mapYmax);
      h2AuxEfficiencyMap_rotated[rpId] =
          ibooker.book2DD(Form("h2AuxEfficiencyMap_rotated_arm%i_st%i_rp%i_pl%i", arm, station, rp, plane),
                          Form("h2AuxEfficiencyMap_rotated_arm%i_st%i_rp%i_pl%i", arm, station, rp, plane),
                          mapXbins,
                          mapXmin_,
                          mapXmax_,
                          mapYbins,
                          mapYmin,
                          mapYmax);
      h2EfficiencyNormalizationMap_rotated[rpId] =
          ibooker.book2DD(Form("h2EfficiencyNormalizationMap_rotated_arm%i_st%i_rp%i_pl%i", arm, station, rp, plane),
                          Form("h2EfficiencyNormalizationMap_rotated_arm%i_st%i_rp%i_pl%i", arm, station, rp, plane),
                          mapXbins,
                          mapXmin_,
                          mapXmax_,
                          mapYbins,
                          mapYmin,
                          mapYmax);

      if (supplementaryPlots_) {
        h2ModuleHitMap_rotated[rpId] =
            ibooker.book2DD(Form("h2ModuleHitMap_rotated_arm%i_st%i_rp%i_pl%i", arm, station, rp, plane),
                            Form("h2ModuleHitMap_rotated_arm%i_st%i_rp%i_pl%i; x (mm); y "
                                 "(mm)",
                                 arm,
                                 station,
                                 rp,
                                 plane),
                            mapXbins,
                            mapXmin_,
                            mapXmax_,
                            mapYbins,
                            mapYmin,
                            mapYmax);
      }
    }
    //INTERPOT - TRACK MUX
    CTPPSPixelDetId detId_Tag = CTPPSPixelDetId(arm, station, rp);
    h1TrackMux_[detId_Tag] =
        ibooker.book1DD(Form("h1TrackMux_arm%i_st%i_rp%i", detId_Tag.arm(), detId_Tag.station(), detId_Tag.rp()),
                        Form("h1TrackMux_arm%i_st%i_rp%i", detId_Tag.arm(), detId_Tag.station(), detId_Tag.rp()),
                        11,
                        0,
                        11);
  }
  //INTERPOT - REST
  for (int arm = 0; arm <= 1; arm++) {
    std::string folderName = Form("Arm%i", arm);
    ibooker.setCurrentFolder(folderName);
    uint32_t arm_Probe = arm;
    uint32_t station_Probe = 2;
    uint32_t rp_Probe = 3;

    CTPPSPixelDetId pixelDetId(arm_Probe, station_Probe, rp_Probe);

    if (fancyBinning_) {
      h2ProtonHitExpectedDistribution_[pixelDetId] =
          ibooker.book2DD(Form("h2ProtonHitExpectedDistribution_arm%i", arm_Probe),
                          Form("h2ProtonHitExpectedDistribution_arm%i;x (mm);y (mm)", arm_Probe),
                          nBinsX_total[pixelDetId],
                          get_min(xBinEdges[pixelDetId]),
                          get_max(xBinEdges[pixelDetId]),
                          mapYbins,
                          mapYmin,
                          mapYmax);
      h2AuxProtonHitDistributionWithNoMultiRP_[pixelDetId] =
          ibooker.book2DD(Form("h2ProtonHitExpectedDistributionWithNoMultiRP_arm%i", arm_Probe),
                          Form("h2ProtonHitExpectedDistributionWithNoMultiRP_arm%i;"
                               "x (mm);y (mm)",
                               arm_Probe),
                          nBinsX_total[pixelDetId],
                          get_min(xBinEdges[pixelDetId]),
                          get_max(xBinEdges[pixelDetId]),
                          mapYbins,
                          mapYmin,
                          mapYmax);
      h2InterPotEfficiencyMap_[pixelDetId] =
          ibooker.book2DD(Form("h2InterPotEfficiencyMap_arm%i", arm_Probe),
                          Form("h2InterPotEfficiencyMap_arm%i;x (mm);y (mm)", arm_Probe),
                          nBinsX_total[pixelDetId],
                          get_min(xBinEdges[pixelDetId]),
                          get_max(xBinEdges[pixelDetId]),
                          mapYbins,
                          mapYmin,
                          mapYmax);

      ibooker.book2DD(Form("h2InterPotEfficiencyMapFinal_arm%i", arm_Probe),
                      Form("h2InterPotEfficiencyMapFinal_arm%i;x (mm);y (mm)", arm_Probe),
                      nBinsX_total[pixelDetId],
                      get_min(xBinEdges[pixelDetId]),
                      get_max(xBinEdges[pixelDetId]),
                      mapYbins,
                      mapYmin,
                      mapYmax);

      h2InterPotEfficiencyMapMultiRP_[pixelDetId] =
          ibooker.book2DD(Form("h2InterPotEfficiencyMapMultiRP_arm%i", arm_Probe),
                          Form("h2InterPotEfficiencyMapMultiRP_arm%i;x (mm);y (mm)", arm_Probe),
                          nBinsX_total[pixelDetId],
                          get_min(xBinEdges[pixelDetId]),
                          get_max(xBinEdges[pixelDetId]),
                          mapYbins,
                          mapYmin,
                          mapYmax);

      ibooker.book2DD(Form("h2InterPotEfficiencyMapMultiRPFinal_arm%i", arm_Probe),
                      Form("h2InterPotEfficiencyMapMultiRPFinal_arm%i;x (mm);y (mm)", arm_Probe),
                      nBinsX_total[pixelDetId],
                      get_min(xBinEdges[pixelDetId]),
                      get_max(xBinEdges[pixelDetId]),
                      mapYbins,
                      mapYmin,
                      mapYmax);

    } else {
      h2ProtonHitExpectedDistribution_[pixelDetId] =
          ibooker.book2DD(Form("h2ProtonHitExpectedDistribution_arm%i", arm_Probe),
                          Form("h2ProtonHitExpectedDistribution_arm%i;x (mm);y (mm)", arm_Probe),
                          mapXbins,
                          mapXmin_,
                          mapXmax_,
                          mapYbins,
                          mapYmin,
                          mapYmax);
      h2AuxProtonHitDistributionWithNoMultiRP_[pixelDetId] =
          ibooker.book2DD(Form("h2ProtonHitExpectedDistributionWithNoMultiRP_arm%i", arm_Probe),
                          Form("h2ProtonHitExpectedDistributionWithNoMultiRP_arm%i;"
                               "x (mm);y (mm)",
                               arm_Probe),
                          mapXbins,
                          mapXmin_,
                          mapXmax_,
                          mapYbins,
                          mapYmin,
                          mapYmax);

      h2InterPotEfficiencyMap_[pixelDetId] =
          ibooker.book2DD(Form("h2InterPotEfficiencyMap_arm%i", arm_Probe),
                          Form("h2InterPotEfficiencyMap_arm%i;x (mm);y (mm)", arm_Probe),
                          mapXbins,
                          mapXmin_,
                          mapXmax_,
                          mapYbins,
                          mapYmin,
                          mapYmax);
      h2InterPotEfficiencyMapMultiRP_[pixelDetId] =
          ibooker.book2DD(Form("h2InterPotEfficiencyMapMultiRP_arm%i", arm_Probe),
                          Form("h2InterPotEfficiencyMapMultiRP_arm%i;x (mm);y (mm)", rp_Probe),
                          mapXbins,
                          mapXmin_,
                          mapXmax_,
                          mapYbins,
                          mapYmin,
                          mapYmax);

      ibooker.book2DD(Form("h2InterPotEfficiencyMapMultiRPFinal_arm%i", arm_Probe),
                      Form("h2InterPotEfficiencyMapMultiRPFinal_arm%i;x (mm);y (mm)", arm_Probe),
                      mapXbins,
                      mapXmin_,
                      mapXmax_,
                      mapYbins,
                      mapYmin,
                      mapYmax);
    }
    h1InterPotEfficiencyVsXi_[pixelDetId] =
        ibooker.book1DD(Form("h1InterPotEfficiencyVsXi_arm%i", arm_Probe),
                        Form("h1InterPotEfficiencyVsXi_arm%i;#xi;Efficiency", arm_Probe),
                        xiBins,
                        xiMin,
                        xiMax);

    ibooker.book1DD(Form("h1InterPotEfficiencyVsXiFinal_arm%i", arm_Probe),
                    Form("h1InterPotEfficiencyVsXiFinal_arm%i;#xi;Efficiency", arm_Probe),
                    xiBins,
                    xiMin,
                    xiMax);

    h1AuxXi_[pixelDetId] = ibooker.book1DD(
        Form("h1AuxXi_arm%i", arm_Probe), Form("h1AuxXi_arm%i;#xi;Efficiency", arm_Probe), xiBins, xiMin, xiMax);
    
    h1DeltaXiMatch_[pixelDetId] = ibooker.book1DD(
        Form("h1DeltaXiMatch_arm%i", arm_Probe), Form("h1DeltaXiMatch_arm%i;#Delta_{#xi}", arm_Probe), 100, -0.02, 0.02);
    
    h1DeltaYMatch_[pixelDetId] = ibooker.book1DD(
        Form("h1DeltaYMatch_arm%i", arm_Probe), Form("h1DeltaYMatch_arm%i;#Delta_{#xi}", arm_Probe), 100, -5, 5);
    
    h1TxMatch_[pixelDetId] =
        ibooker.book1DD(Form("h1TxMatch_arm%i", arm_Probe), Form("h1TxMatch_%i;Tx", arm_Probe), 100, -0.02, 0.02);
    
    h1TyMatch_[pixelDetId] =
        ibooker.book1DD(Form("h1TyMatch_arm%i", arm_Probe), Form("h1TyMatch_%i;Ty", arm_Probe), 100, -0.02, 0.02);
    h1ProtonsInProbePotWhenNoMatchFound_[pixelDetId] =
        ibooker.book1DD(Form("h1ProtonsInProbePotWhenNoMatchFound_arm%i", arm_Probe),
                        Form("h1ProtonsInProbePotWhenNoMatchFound_arm%i", arm_Probe),
                        11,
                        0,
                        11);
    
    h2XCorrelationMatch_[pixelDetId] = ibooker.book2DD(Form("h2XCorrelationMatch_arm%i", arm_Probe),
                                                       Form("h2XCorrelationMatch_arm%i;x pixel (mm);x "
                                                            "strips (mm)",
                                                            arm_Probe),
                                                       mapXbins,
                                                       mapXmin_,
                                                       mapXmax_,
                                                       mapXbins,
                                                       mapXmin_,
                                                       mapXmax_);
    
    h2YCorrelationMatch_[pixelDetId] =
        ibooker.book2DD(Form("h2YCorrelationMatch_arm%i", arm_Probe),
                        Form("h2YCorrelationMatch_arm%i;y pixel (mm);y strips (mm)", arm_Probe),
                        mapYbins,
                        mapYmin,
                        mapYmax,
                        mapYbins,
                        mapYmin,
                        mapYmax);
    
    h2TxCorrelationMatch_[pixelDetId] = ibooker.book2DD(Form("h2TxCorrelationMatch_arm%i", arm_Probe),
                                                        Form("h2TxCorrelationMatch_arm%i;Tx pixel (mm);Ty "
                                                             "pixel (mm)",
                                                             arm_Probe),
                                                        100,
                                                        -0.01,
                                                        0.01,
                                                        100,
                                                        -0.01,
                                                        0.01);
    
    h2TyCorrelationMatch_[pixelDetId] = ibooker.book2DD(Form("h2TyCorrelationMatch_arm%i", arm_Probe),
                                                        Form("h2TyCorrelationMatch_arm%i;Tx pixel (mm);Ty "
                                                             "pixel (mm)",
                                                             arm_Probe),
                                                        100,
                                                        -0.01,
                                                        0.01,
                                                        100,
                                                        -0.01,
                                                        0.01);
  }

  listOfPlanes_.assign(planesSet.begin(), planesSet.end());
  ibooker.cd();
}


void EfficiencyTool_2018DQMWorker::setGlobalBinSizes(CTPPSPixelDetId &rpId) {
  double binSize = (mapXmax_ - mapXmin_) / mapXbins;
  mapXmin_ += binAlignmentParameters[rpId] * binSize / 150.;
  mapXmax_ += binAlignmentParameters[rpId] * binSize / 150.;
}


void EfficiencyTool_2018DQMWorker::dqmBeginRun(edm::Run const & iRun, edm::EventSetup const & iSetup) {
  bool changed(true);

  if (!usePrescales_)
    return;
  // Only prescale stuff below
  if (hltPrescaleProvider_.init(iRun, iSetup, processName_, changed)) {
    HLTConfigProvider const& hltConfig = hltPrescaleProvider_.hltConfigProvider();
    const std::vector<std::string> triggerNames(hltConfig.triggerNames());

    // Do the matching to find out the HLT name
    if (edm::is_glob(triggerPattern_)) {  // handle triggerPattern_ with wildcards (*,?)
      std::vector<std::vector<std::string>::const_iterator> matches = edm::regexMatch(triggerNames, triggerPattern_);
      if (matches.empty()) {
        throw cms::Exception("PPS") 
          << "requested trigger pattern [" << triggerPattern_ << "] does not match any HLT paths";
      }
      if (matches.size() > 1) {
        throw cms::Exception("PPS") 
          << "requested trigger pattern [" << triggerPattern_ << "] matches more than one HLT path";
      }
      hltName_ = *matches[0];
      if(debug_)
        std::cout << "Matched HLT path name: " << hltName_ << std::endl;
      
    } else {  // take full HLT path name given
      hltName_ = triggerPattern_;
    }

    if (changed) {
      edm::LogInfo("PPS") << "HLT configuration changed between runs";
      const unsigned int n(hltConfig.size());
      const unsigned int triggerIndex(hltConfig.triggerIndex(triggerPattern_));
      if (triggerIndex >= n) {
        edm::LogInfo("PPS") << "EfficiencyTool_2018DQMWorker::dqmBeginRun:"
                       << " TriggerName " << triggerPattern_ << " not available in (new) config!" << endl;
        edm::LogInfo("PPS") << "Available TriggerNames are: " << endl;
        hltConfig.dump("Triggers");
      }
    }
  } else {
    edm::LogError("PPS") << " HLT config extraction failure with process name " << processName_;
  }
}


void EfficiencyTool_2018DQMWorker::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  using namespace edm;

  Handle<edm::DetSetVector<CTPPSPixelLocalTrack>> pixelLocalTracks;
  iEvent.getByToken(pixelLocalTrackToken_, pixelLocalTracks);

  int ls = iEvent.getLuminosityBlock().id().luminosityBlock();
  double weight = 1.;
 
  if (usePrescales_){ 
    const auto prescales(hltPrescaleProvider_.prescaleValuesInDetail<double,double>(iEvent, iSetup, hltName_));

    // Print prescales
    if (debug_){
      std::cout << "LS: " << ls << std::endl;
      std::cout << "\tHLT prescale: " << prescales.second << std::endl;
      std::cout << "\tL1 prescales: " << std::endl;

    }

    bool l1Prescale_found = false;
    for (const auto &l1Prescale : prescales.first){
      if (debug_)
        std::cout << "\t\t" << l1Prescale.first << ": " << l1Prescale.second << std::endl;
      if (l1Prescale.first == "L1_ZeroBias"){
        h1L1Prescale_->Fill(ls, l1Prescale.second);
        weight *= l1Prescale.second; // Correct for the L1 prescale value
        l1Prescale_found = true;
      }
    }
    if (!l1Prescale_found){
      std::cout << "L1_ZeroBias prescale not found!" << std::endl;
      return;
    }

    h1HLTPrescale_->Fill(ls,prescales.second);
    // weight *= prescales.second; // Correct for the HLT prescale value
  }

  if (!validBunchArray_[iEvent.eventAuxiliary().bunchCrossing()])
    return;
  h1BunchCrossing_->Fill(iEvent.eventAuxiliary().bunchCrossing(), weight);

  LHCInfoCombined lhcInfoCombined(iSetup, lhcInfoPerLSToken_, lhcInfoPerFillToken_, lhcInfoToken_, useNewLHCInfo_);

  // re-initialise algorithm upon crossing-angle change
  h1CrossingAngle_->Fill(lhcInfoCombined.crossingAngle(), weight);
  if (debug_)
    std::cout << "Crossing angle: " << lhcInfoCombined.crossingAngle() << std::endl; 
  // search for rps missing from the track collection (because they are empty) and fill the multiplicity
  for (const auto &rpIdAndHist : h1NumberOfTracks_){
    if (pixelLocalTracks->find(rpIdAndHist.first) == pixelLocalTracks->end())
      rpIdAndHist.second->Fill(0, weight);
  }

  for (const auto &rpPixeltrack : *pixelLocalTracks) {
    CTPPSPixelDetId rpId = CTPPSPixelDetId(rpPixeltrack.detId());
    uint32_t arm = rpId.arm();
    uint32_t rp = rpId.rp();
    uint32_t station = rpId.station();

    h1NumberOfTracks_[rpId]->Fill(rpPixeltrack.size(), weight);
        
    if ((uint32_t)minTracksPerEvent_ > rpPixeltrack.size() || rpPixeltrack.size() > (uint32_t)maxTracksPerEvent)
      continue;
    
    for (const auto &pixeltrack : rpPixeltrack) {
      if (Cut(pixeltrack, arm, station))
        continue;
      if (!pixeltrack.isValid())
        continue;


      float pixelX0 = pixeltrack.x0();
      float pixelY0 = pixeltrack.y0();
      int numberOfRowCls2 = 0;
      int numberOfColCls2 = 0;

      // Rotating St0 tracks
      float pixelX0_rotated = 0;
      float pixelY0_rotated = 0;
      if (station == 0) {
        pixelX0_rotated = pixeltrack.x0() * TMath::Cos((detectorRotationAngle_ / 180.) * TMath::Pi()) -
                          pixeltrack.y0() * TMath::Sin((detectorRotationAngle_ / 180.) * TMath::Pi());
        pixelY0_rotated = pixeltrack.x0() * TMath::Sin((detectorRotationAngle_ / 180.) * TMath::Pi()) +
                          pixeltrack.y0() * TMath::Cos((detectorRotationAngle_ / 180.) * TMath::Pi());
      }

      int numberOfFittedPoints = 0;
      std::vector<int> planesContributingToTrack;
      edm::DetSetVector<CTPPSPixelFittedRecHit> fittedHits = pixeltrack.hits();

      std::map<uint32_t, int> numberOfPointPerPlaneEff;
      for (const auto pln : listOfPlanes_) {
        numberOfPointPerPlaneEff[pln] = 0;
      }

      for (const auto &planeHits : fittedHits) {
        CTPPSPixelDetId planeId = CTPPSPixelDetId(planeHits.detId());
        uint32_t plane = planeId.plane();
        planeId = CTPPSPixelDetId(arm, station, rp, plane);

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
              hitX0_rotated = hitX0 * TMath::Cos((detectorRotationAngle_ / 180.) * TMath::Pi()) -
                              hitY0 * TMath::Sin((detectorRotationAngle_ / 180.) * TMath::Pi());
              hitY0_rotated = hitX0 * TMath::Sin((detectorRotationAngle_ / 180.) * TMath::Pi()) +
                              hitY0 * TMath::Cos((detectorRotationAngle_ / 180.) * TMath::Pi());
            }
            h2ModuleHitMap_[planeId]->Fill(hitX0, hitY0, weight);

            if (supplementaryPlots_ && station == 0) {
              h2ModuleHitMap_rotated[planeId]->Fill(hitX0_rotated, hitY0_rotated, weight);
            }
            if (hit.clusterSizeRow() == 2)
              ++numberOfRowCls2;
            if (hit.clusterSizeCol() == 2)
              ++numberOfColCls2;
          }
        }
      }

      h2TrackHitDistribution_[rpId]->Fill(pixelX0, pixelY0, weight);

      if (supplementaryPlots_) {
        h2AvgPlanesUsed_[rpId]->Fill(pixelX0, pixelY0, numberOfFittedPoints * weight);
        h1PlanesUsed_[rpId]->Fill(numberOfFittedPoints, weight);
        h1ChiSquaredOverNDF_[rpId]->Fill(pixeltrack.chiSquaredOverNDF(), weight);

        // Sort the vector of planes and require them to be consecutive and fill
        // hist
        std::sort(planesContributingToTrack.begin(), planesContributingToTrack.end());
        bool areConsecutive = true;
        for (auto it = planesContributingToTrack.begin(); it != (planesContributingToTrack.end() - 1); it++) {
          if (*(it + 1) - *it != 1) {
            areConsecutive = false;
            break;
          }
        }
        h1ConsecutivePlanes_[rpId]->Fill(areConsecutive);

        if (station == 0) {
          h2AvgPlanesUsed_rotated[rpId]->Fill(pixelX0_rotated, pixelY0_rotated, numberOfFittedPoints * weight);
          h2TrackHitDistribution_rotated[rpId]->Fill(pixelX0_rotated, pixelY0_rotated, weight);
        }
      }
      if (numberOfFittedPoints == 3) {
        h23PointsTrackHitDistribution_[rpId]->Fill(pixelX0, pixelY0, weight);
      }

      if (supplementaryPlots_ && pixeltrack.chiSquaredOverNDF() < 2.) {
        h1X0Sigma[rpId][std::pair(numberOfFittedPoints, numberOfColCls2)]->Fill(pixeltrack.x0Sigma(), weight);
        h1Y0Sigma[rpId][std::pair(numberOfFittedPoints, numberOfRowCls2)]->Fill(pixeltrack.y0Sigma(), weight);
        h1TxSigma[rpId][std::pair(numberOfFittedPoints, numberOfColCls2)]->Fill(pixeltrack.txSigma(), weight);
        h1TySigma[rpId][std::pair(numberOfFittedPoints, numberOfRowCls2)]->Fill(pixeltrack.tySigma(), weight);
      }
      // Efficiency calculation
      for (const auto pln : listOfPlanes_) {
        CTPPSPixelDetId planeId = rpId;
        planeId.setPlane(pln);
        edm::DetSet<CTPPSPixelFittedRecHit> hitOnPlane = fittedHits[planeId];
        float hitX0 = hitOnPlane[0].globalCoordinates().x() + hitOnPlane[0].xResidual();
        ;
        float hitY0 = hitOnPlane[0].globalCoordinates().y() + hitOnPlane[0].yResidual();
        ;
        float hitX0_rotated = 0;
        float hitY0_rotated = 0;
        if (station == 0) {
          hitX0_rotated = hitX0 * TMath::Cos((detectorRotationAngle_ / 180.) * TMath::Pi()) -
                          hitY0 * TMath::Sin((detectorRotationAngle_ / 180.) * TMath::Pi());
          hitY0_rotated = hitX0 * TMath::Sin((detectorRotationAngle_ / 180.) * TMath::Pi()) +
                          hitY0 * TMath::Cos((detectorRotationAngle_ / 180.) * TMath::Pi());
        }
        if (numberOfPointPerPlaneEff[pln] >= minNumberOfPlanesForEfficiency_) {
          h2EfficiencyNormalizationMap_[planeId]->Fill(hitX0, hitY0, weight);

          if (station == 0) {
            h2EfficiencyNormalizationMap_rotated[planeId]->Fill(hitX0_rotated, hitY0_rotated, weight);
          }
          if (hitOnPlane[0].isRealHit()) {
            h2AuxEfficiencyMap_[planeId]->Fill(hitX0, hitY0, weight);
            if (station == 0) {
              h2AuxEfficiencyMap_rotated[planeId]->Fill(hitX0_rotated, hitY0_rotated, weight);
            }
          }
        }
      }
    }
  }

  //INTERPOT
  Handle<reco::ForwardProtonCollection> protons;
  iEvent.getByToken(protonsToken_, protons);

  Handle<reco::ForwardProtonCollection> multiRP_protons;
  iEvent.getByToken(multiRP_protonsToken_, multiRP_protons);

  double xangle = lhcInfoCombined.crossingAngle();

  trackMux_.clear();
  for (auto &proton_Tag : *protons) {
    if (!proton_Tag.validFit())
      continue;

    CTPPSLocalTrackLite track_Tag = *(proton_Tag.contributingLocalTracks().at(0));
    CTPPSPixelDetId detId_Tag = CTPPSPixelDetId(track_Tag.rpId());

    trackMux_[detId_Tag]++;
  }

  for (auto const &idAndHist : h1TrackMux_) {
    idAndHist.second->Fill(trackMux_[idAndHist.first], weight);
  }

  //INTERPOT
  for (auto &proton_Tag : *protons) {
    if (!proton_Tag.validFit()) {
      continue;
    }

    CTPPSLocalTrackLite track_Tag = *(proton_Tag.contributingLocalTracks().at(0));
    CTPPSPixelDetId detId_Tag = CTPPSPixelDetId(track_Tag.rpId());
    int arm_Tag = detId_Tag.arm();
    int station_Tag = detId_Tag.station();
    double trackX0_Tag = track_Tag.x();
    double trackY0_Tag = track_Tag.y();
    double trackTx_Tag = track_Tag.tx();
    double trackTy_Tag = track_Tag.ty();
    double xi_Tag = proton_Tag.xi();
    int matches = 0;

    if (trackMux_[detId_Tag] > maxTracksInTagPot || trackMux_[detId_Tag] < minTracksInTagPot) {
      continue;
    }
    // Start only from strips

    // Apply aperture cut
    if (debug_)
      std::cout << "Aperture cut for arm " << arm_Tag << ": xangle = " << xangle
                << " xiMax = " << Aperture(xangle, arm_Tag, "2018") << std::endl;

    // Apply the cuts
    if (Cut(track_Tag))
      continue;

    if (station_Tag != 0)
      continue;  //we are cutting tracks from stations other than the one nearest to the detector
    uint32_t arm_Probe = detId_Tag.arm();
    uint32_t station_Probe = (detId_Tag.station() == 0) ? 2 : 0;
    uint32_t rp_Probe = detId_Tag.rp();

    // CTPPSPixelDetId that the probe proton must have
    CTPPSPixelDetId pixelDetId(arm_Probe, station_Probe, rp_Probe);
    CTPPSPixelDetId detId_Probe(pixelDetId.rawId());

    if (trackMux_[detId_Probe] > maxTracksInProbePot || trackMux_[detId_Probe] < minTracksInProbePot) {
      continue;
    }

    double deltaZ = Z[detId_Probe] - Z[detId_Tag];
    double expectedTrackX0_Probe = trackX0_Tag;  //+ trackTx_Tag * deltaZ;
    double expectedTrackY0_Probe = trackY0_Tag;  //+ trackTy_Tag * deltaZ;
    int protonsInProbePot = 0;

    for (auto &proton_Probe : *protons) {  // Probe -> Roman Pot Under Test
      if (!proton_Probe.validFit()) {
        //std::cout<<"HERE"<<std::endl;
        continue;
      }
      CTPPSLocalTrackLite track_Probe = *(proton_Probe.contributingLocalTracks().at(0));
      // CTPPSPixelDetId detId_Probe = CTPPSPixelDetId(track_Probe.rpId());
      double trackX0_Probe = track_Probe.x();
      double trackY0_Probe = track_Probe.y();
      double trackTx_Probe = track_Probe.tx();
      double trackTy_Probe = track_Probe.ty();
      double xi_Probe = proton_Probe.xi();
      // Require the proton_Probe to be in the same arm, different station
      // This means that the CTPPSPixelDetId is the same as above

      if (detId_Probe != track_Probe.rpId()) {
        continue;
      }
      protonsInProbePot++;
      // Apply the cuts
      if (Cut(track_Probe)) {
        continue;
      }

      bool xiMatchPass = false;
      bool yMatchPass = false;
      bool xMatchPass = false;

      // Make it so that the difference is always NEAR - FAR
      double xiDiff = (station_Tag == 0) ? xi_Tag - xi_Probe : xi_Probe - xi_Tag;
      double xDiff = (station_Tag == 0) ? trackX0_Tag - trackX0_Probe : trackX0_Probe - trackX0_Tag;
      double yDiff = (station_Tag == 0) ? trackY0_Tag - trackY0_Probe : trackY0_Probe - trackY0_Tag;
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

      h1DeltaXiMatch_[pixelDetId]->Fill(xi_Tag - xi_Probe, weight);

      if (xiMatchPass) {
        h1DeltaYMatch_[pixelDetId]->Fill(trackY0_Tag - trackY0_Probe, weight);
        if (xMatchPass && yMatchPass) {
          matches++;
          if (debug_) {
            std::cout << "********MATCH FOUND********" << std::endl;
            std::cout << "Tag track:\n"
                      << "Arm: " << detId_Tag.arm() << " Station: " << detId_Tag.station() << " X: " << trackX0_Tag
                      << " Y: " << trackY0_Tag << " Tx: " << trackTx_Tag << " Ty: " << trackTy_Tag << " Xi: " << xi_Tag
                      << std::endl;
            std::cout << "Probe track:\n"
                      << "Arm: " << detId_Probe.arm() << " Station: " << detId_Probe.station()
                      << " X: " << trackX0_Probe << " Y: " << trackY0_Probe << " Tx: " << trackTx_Probe
                      << " Ty: " << trackTy_Probe << " Xi: " << xi_Probe << "\nDeltaZ: " << deltaZ
                      << " Expected X: " << expectedTrackX0_Probe << " Expected Y: " << expectedTrackY0_Probe
                      << " RecoInfo: " << (int)track_Probe.pixelTrackRecoInfo() << std::endl;
            std::cout << "**************************" << std::endl;
          }
          if (matches == 1) {
            h2InterPotEfficiencyMap_[pixelDetId]->Fill(expectedTrackX0_Probe, expectedTrackY0_Probe, weight);
            h1InterPotEfficiencyVsXi_[pixelDetId]->Fill(xi_Tag,
                                                        weight);  // xi_Tag and xi_Probe are expected to be the same
            h1TxMatch_[pixelDetId]->Fill(trackTx_Tag, weight);
            h1TyMatch_[pixelDetId]->Fill(trackTy_Tag, weight);
            h2XCorrelationMatch_[pixelDetId]->Fill(trackX0_Probe, trackX0_Tag, weight);
            h2YCorrelationMatch_[pixelDetId]->Fill(trackY0_Probe, trackY0_Tag, weight);
            h2TxCorrelationMatch_[pixelDetId]->Fill(trackTx_Probe, trackTx_Tag, weight);
            h2TyCorrelationMatch_[pixelDetId]->Fill(trackTy_Probe, trackTy_Tag, weight);
          }
          if (excludeMultipleMatches && matches == 2) {
            h2InterPotEfficiencyMap_[pixelDetId]->Fill(expectedTrackX0_Probe, expectedTrackY0_Probe, -1 * weight);
            h1InterPotEfficiencyVsXi_[pixelDetId]->Fill(
                xi_Tag, -1 * weight);  // xi_Tag and xi_Probe are expected to be the same
            h1TxMatch_[pixelDetId]->Fill(trackTx_Tag, -1 * weight);
            h1TyMatch_[pixelDetId]->Fill(trackTy_Tag, -1 * weight);
            h2XCorrelationMatch_[pixelDetId]->Fill(trackX0_Probe, trackX0_Tag, -1 * weight);
            h2YCorrelationMatch_[pixelDetId]->Fill(trackY0_Probe, trackY0_Tag, -1 * weight);
            h2TxCorrelationMatch_[pixelDetId]->Fill(trackTx_Probe, trackTx_Tag, -1 * weight);
            h2TyCorrelationMatch_[pixelDetId]->Fill(trackTy_Probe, trackTy_Tag, -1 * weight);
          }
        }
      }
    }

    // MultiRP efficiency
    uint32_t multiRPmatchFound = 0;
    for (auto &multiRP_proton : *multiRP_protons) {
      if (!multiRP_proton.validFit() || multiRP_proton.method() != reco::ForwardProton::ReconstructionMethod::multiRP) {
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
        if (debug_) {
          std::string rpName;
          CTPPSDetId(track.rpId()).rpName(rpName,CTPPSDetId::nFull);
          std::cout << "Track from " << rpName << std::endl;
        }

        CTPPSPixelDetId detId(0,0);
        try {
          detId = CTPPSPixelDetId(track.rpId());
        } catch (cms::Exception const& e){
          if (debug_) {
            std::cout << e;
            std::cout << "Ignoring this non-pixel track" << std::endl;
          }
          if (e.category() == "InvalidDetId")
            continue;
          else 
            throw e;
        }
        int arm = detId.arm();
        int station = detId.station();
        double trackX0 = track.x();
        double trackY0 = track.y();
        double trackTx = track.tx();
        double trackTy = track.ty();

        if (debug_) {
          std::cout << "Arm: " << arm << " Station: " << station << std::endl
                    << " X: " << trackX0 << " Y: " << trackY0 << " Tx: " << trackTx << " Ty: " << trackTy
                    << " recoInfo: " << (int)track.pixelTrackRecoInfo() << std::endl;
        }

        if (arm == arm_Tag && station == station_Tag && station != 1 && TMath::Abs(trackX0_Tag - trackX0) < 0.01 &&
            TMath::Abs(trackY0_Tag - trackY0) < 0.01) {
          if (debug_)
            std::cout << "**MultiRP proton matched to the tag track!**" << std::endl;
          multiRPmatchFound++;
          h2InterPotEfficiencyMapMultiRP_[pixelDetId]->Fill(expectedTrackX0_Probe, expectedTrackY0_Probe * weight);
        }
      }
    }
    if (multiRPmatchFound > 1) {
      std::cout << "WARNING: More than one multiRP matched!" << std::endl;
    }

    h2ProtonHitExpectedDistribution_[pixelDetId]->Fill(expectedTrackX0_Probe, expectedTrackY0_Probe, weight);
    if (multiRPmatchFound == 0)
      h2AuxProtonHitDistributionWithNoMultiRP_[pixelDetId]->Fill(expectedTrackX0_Probe, expectedTrackY0_Probe, weight);
    h1AuxXi_[pixelDetId]->Fill(xi_Tag, weight);

    if (matches > 1) {
      overmatches[pixelDetId]++;
      if (debug_)
        std::cout << "***WARNING: Overmatching!***" << std::endl;
    }
    tries[pixelDetId]++;

    bool goodInterPotMatch = (excludeMultipleMatches) ? matches == 1 : matches >= 1;

    if (!goodInterPotMatch) {
      h1ProtonsInProbePotWhenNoMatchFound_[pixelDetId]->Fill(protonsInProbePot, weight);
    }
  }
}

void EfficiencyTool_2018DQMWorker::initialize() {
  // Applying bunch selection

  std::ifstream bunchListFile(bunchListFileName_.data());
  if (bunchSelection_ == "NoSelection") {
    std::fill_n(validBunchArray_, totalNumberOfBunches_, true);
    return;
  }
  if (!bunchListFile.good()) {
    std::cout << "BunchList file not good. Skipping buch selection..." << std::endl;
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
        validBunchArray_[i] =
            filledBunchArray[totalNumberOfBunches_ - 1] && filledBunchArray[i] && filledBunchArray[i + 1];
      else if (i == totalNumberOfBunches_ - 1)
        validBunchArray_[i] = filledBunchArray[i - 1] && filledBunchArray[i] && filledBunchArray[0];
      else
        validBunchArray_[i] = filledBunchArray[i - 1] && filledBunchArray[i] && filledBunchArray[i + 1];
    } else if (bunchSelection_ == "FirstBunchInTrain") {
      if (i == 0)
        validBunchArray_[i] = filledBunchArray[i] && !filledBunchArray[totalNumberOfBunches_ - 1];
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

void EfficiencyTool_2018DQMWorker::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;

  // Configs for the plane efficiency plots
  desc.addUntracked<edm::InputTag>("tagPixelLocalTracks", edm::InputTag("ctppsPixelLocalTracks"));
  desc.add<int>("minNumberOfPlanesForEfficiency", 3)
    ->setComment("Minimum number of planes that must be hit for the efficiency to be computed");
  desc.add<int>("minNumberOfPlanesForTrack", 3)
    ->setComment("Minimum number of planes that must be hit for the track to be considered");
  desc.add<int>("maxNumberOfPlanesForTrack", 6)
    ->setComment("Maximum number of planes that must be hit for the track to be considered");
  desc.addUntracked<double>("maxChi2Prob", 0.2)
    ->setComment("Maximum chi2 probability for the proton track to be considered");
  desc.add<int>("minTracksPerEvent", 0)
    ->setComment("Minimum multiplicity for the event to be considered");
  desc.add<int>("maxTracksPerEvent", 99)
    ->setComment("Maximum multiplicity for the event to be considered");
  desc.add<bool>("supplementaryPlots", true)
    ->setComment("Enable to produce supplementary plots");
  desc.add<bool>("isCorrelationPlotEnabled", false)
    ->setComment("Only enable if the estimation of the correlation between Strips and Pixel tracks is under study "
                 "(disables filling of TGraph, reducing the output file size)");
  desc.addUntracked<std::string>("bunchSelection", "NoSelection")
    ->setComment("Bunch selection to be applied. Possible values are: NoSelection, CentralBunchesInTrain, "
                 "FirstBunchInTrain, LastBunchInTrain, FilledBunches");
  desc.addUntracked<std::string>("bunchListFileName", "injectionScheme.csv")
    ->setComment("Name of the file containing the bunch list");
  desc.addUntracked<int>("binGroupingX", 1)
    ->setComment("Number of bins to be grouped in the X axis of the efficiency plots");
  desc.addUntracked<int>("binGroupingY", 1)
    ->setComment("Number of bins to be grouped in the Y axis of the efficiency plots");
  desc.addUntracked<std::vector<double>>("fiducialXLow", {0.,0.,0.,0.})
    ->setComment("Lower bound of the fiducial region in X");
  desc.addUntracked<std::vector<double>>("fiducialYLow", {-20.0,-20.0,-20.0,-20.0})
    ->setComment("Lower bound of the fiducial region in Y");
  desc.addUntracked<std::vector<double>>("fiducialYHigh", {20.0,20.0,20.0,20.0})
    ->setComment("Upper bound of the fiducial region in Y");
  desc.addUntracked<double>("detectorTiltAngle", 20.)
    ->setComment("Detector tilt angle in degrees");
  desc.addUntracked<double>("detectorRotationAngle", -8.)
    ->setComment("Detector rotation angle in degrees");

  // Configs for the inter-pot efficiency plots
  desc.addUntracked<edm::InputTag>("tagProtonsSingleRP", edm::InputTag("ctppsProtons", "singleRP"))
    ->setComment("Tag for the single-RP protons collection");
  desc.addUntracked<edm::InputTag>("tagProtonsMultiRP", edm::InputTag("ctppsProtons", "multiRP"))
    ->setComment("Tag for the multi-RP protons collection");
  desc.addUntracked<int>("minTracksInProbePot", 0)
    ->setComment("Minimum number of tracks in the probe pot for the event to be considered");
  desc.addUntracked<int>("maxTracksInProbePot", 99)
    ->setComment("Maximum number of tracks in the probe pot for the event to be considered");
  desc.addUntracked<int>("minTracksInTagPot", 0)
    ->setComment("Minimum number of tracks in the tag pot for the event to be considered");
  desc.addUntracked<int>("maxTracksInTagPot", 99)
    ->setComment("Maximum number of tracks in the tag pot for the event to be considered");
  desc.addUntracked<int>("recoInfo", 0)
    ->setComment("RecoInfo to be used for the proton track selection. RecoInfo != 0 only in 2017 data.");
  
  // General configs
  desc.addUntracked<bool>("debug", false)
    ->setComment("Enable to print debug information");

  // LHCInfo configs
  desc.add<bool>("useNewLHCInfo", true)
    ->setComment("Enable to use the new LHCInfo classes (LHCInfoPerFill/LS)");
  desc.add<std::string>("lhcInfoLabel", "")
    ->setComment("Label for the LHCInfo collection");
  desc.add<std::string>("lhcInfoPerLSLabel","")
    ->setComment("Label for the LHCInfoPerLS collection");
  desc.add<std::string>("lhcInfoPerFillLabel","")
    ->setComment("Label for the LHCInfoPerFill collection");

  // Configs for prescale provider
  desc.add<bool>("usePrescales", false)
    ->setComment("Enable to use L1/HLT prescales for event reweighting");
  desc.add<std::string>("processName", "HLT")
    ->setComment("HLT process name");
  desc.add<std::string>("triggerPattern", "HLT_PPSMaxTracksPerRP4_v*")
    ->setComment("Trigger pattern to be used");
  desc.add<uint32_t>("stageL1Trigger", 2)
    ->setComment("Stage of the L1 trigger to be used");
  desc.add<edm::InputTag>("l1tAlgBlkInputTag", edm::InputTag("gtStage2Digis"))
    ->setComment("Input tag for the L1 algorithm block");
  desc.add<edm::InputTag>("l1tExtBlkInputTag", edm::InputTag("gtStage2Digis"))
    ->setComment("Input tag for the L1 extension block");

  descriptions.add("EfficiencyTool_2018DQMWorker",desc);
}

// This function produces all the possible plane combinations extracting
// numberToExtract planes over numberOfPlanes planes
void EfficiencyTool_2018DQMWorker::getPlaneCombinations(
    const std::vector<uint32_t> &inputPlaneList,
    uint32_t numberToExtract,
    std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> &planesExtractedAndNot) {
  uint32_t numberOfPlanes = inputPlaneList.size();
  std::string bitmask(numberToExtract, 1);  // numberToExtract leading 1's
  bitmask.resize(numberOfPlanes,
                 0);  // numberOfPlanes-numberToExtract trailing 0's
  planesExtractedAndNot.clear();

  // store the combination and permute bitmask
  do {
    planesExtractedAndNot.push_back(
        std::pair<std::vector<uint32_t>, std::vector<uint32_t>>(std::vector<uint32_t>(), std::vector<uint32_t>()));
    for (uint32_t i = 0; i < numberOfPlanes; ++i) {  // [0..numberOfPlanes-1] integers
      if (bitmask[i])
        planesExtractedAndNot.back().second.push_back(inputPlaneList.at(i));
      else
        planesExtractedAndNot.back().first.push_back(inputPlaneList.at(i));
    }
  } while (std::prev_permutation(bitmask.begin(), bitmask.end()));

  return;
}

float EfficiencyTool_2018DQMWorker::probabilityNplanesBlind(const std::vector<uint32_t> &inputPlaneList,
                                                            int numberToExtract,
                                                            const std::map<unsigned, float> &planeEfficiency) {
  std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> planesExtractedAndNot;
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
float EfficiencyTool_2018DQMWorker::efficiencyPartialDerivativewrtPlane(
    uint32_t plane,
    const std::vector<uint32_t> &inputPlaneList,
    int numberToExtract,
    const std::map<unsigned, float> &planeEfficiency) {
  std::vector<uint32_t> modifiedInputPlaneList = inputPlaneList;
  modifiedInputPlaneList.erase(std::find(modifiedInputPlaneList.begin(), modifiedInputPlaneList.end(), plane));
  float partialDerivative = 0.;
  if (numberToExtract > 0 && numberToExtract < 6) {
    partialDerivative = -probabilityNplanesBlind(modifiedInputPlaneList, numberToExtract, planeEfficiency) +
                        probabilityNplanesBlind(modifiedInputPlaneList, numberToExtract - 1, planeEfficiency);
  } else {
    if (numberToExtract == 6) {
      partialDerivative = probabilityNplanesBlind(modifiedInputPlaneList, numberToExtract - 1, planeEfficiency);
    } else {
      partialDerivative = -probabilityNplanesBlind(modifiedInputPlaneList, numberToExtract, planeEfficiency);
    }
  }
  return partialDerivative;
}

float EfficiencyTool_2018DQMWorker::errorCalculation(const std::map<uint32_t, float> &planeEfficiency,
                                                     const std::map<uint32_t, float> &planeEfficiencyError) {
  int minNumberOfBlindPlanes = 3;
  int maxNumberOfBlindPlanes = listOfPlanes_.size();
  float rpEfficiencySquareError = 0.;
  for (const auto &plane : listOfPlanes_) {
    float partialDerivative = 0.;
    for (uint32_t i = (uint32_t)minNumberOfBlindPlanes; i <= (uint32_t)maxNumberOfBlindPlanes; i++) {
      partialDerivative += efficiencyPartialDerivativewrtPlane(plane, listOfPlanes_, i, planeEfficiency);
    }
    rpEfficiencySquareError +=
        partialDerivative * partialDerivative * planeEfficiencyError.at(plane) * planeEfficiencyError.at(plane);
  }
  return TMath::Sqrt(rpEfficiencySquareError);
}

float EfficiencyTool_2018DQMWorker::probabilityCalculation(const std::map<uint32_t, float> &planeEfficiency) {
  int minNumberOfBlindPlanes = 3;
  int maxNumberOfBlindPlanes = listOfPlanes_.size();
  float rpEfficiency = 1.;

  for (uint32_t i = (uint32_t)minNumberOfBlindPlanes; i <= (uint32_t)maxNumberOfBlindPlanes; i++) {
    rpEfficiency -= probabilityNplanesBlind(listOfPlanes_, i, planeEfficiency);
  }
  return rpEfficiency;
}

bool EfficiencyTool_2018DQMWorker::Cut(CTPPSPixelLocalTrack track, int arm, int station) {
  uint32_t ndf = 2 * track.numberOfPointsUsedForFit() - 4;
  double x = track.x0();
  double y = track.y0();
  float pixelX0_rotated = 0;
  float pixelY0_rotated = 0;
  if (station == 0) {
    pixelX0_rotated = x * TMath::Cos((detectorRotationAngle_ / 180.) * TMath::Pi()) -
                      y * TMath::Sin((detectorRotationAngle_ / 180.) * TMath::Pi());
    pixelY0_rotated = x * TMath::Sin((detectorRotationAngle_ / 180.) * TMath::Pi()) +
                      y * TMath::Cos((detectorRotationAngle_ / 180.) * TMath::Pi());
    x = pixelX0_rotated;
    y = pixelY0_rotated;
  }

  double maxTx = 0.02;
  double maxTy = 0.02;
  double maxChi2 = TMath::ChisquareQuantile(0.95, ndf);

  if (TMath::Abs(track.tx()) > maxTx || TMath::Abs(track.ty()) > maxTy || track.chiSquaredOverNDF() * ndf > maxChi2 ||
      track.numberOfPointsUsedForFit() < minNumberOfPlanesForTrack_ ||
      track.numberOfPointsUsedForFit() > maxNumberOfPlanesForTrack_ ||
      y > fiducialYHigh_[std::pair<int, int>(arm, station)] || y < fiducialYLow_[std::pair<int, int>(arm, station)] ||
      x < fiducialXLow_[std::pair<int, int>(arm, station)])
    return true;
  else
    return false;
}

bool EfficiencyTool_2018DQMWorker::Cut(CTPPSLocalTrackLite track) {
  CTPPSPixelDetId detId = CTPPSPixelDetId(track.rpId());
  uint32_t arm = detId.arm();
  uint32_t station = detId.station();
  uint32_t ndf = 2 * track.numberOfPointsUsedForFit() - 4;
  double x = track.x();
  double y = track.y();
  float pixelX0_rotated = 0;
  float pixelY0_rotated = 0;
  if (station == 0) {
    pixelX0_rotated = x * TMath::Cos((-8. / 180.) * TMath::Pi()) - y * TMath::Sin((-8. / 180.) * TMath::Pi());
    pixelY0_rotated = x * TMath::Sin((-8. / 180.) * TMath::Pi()) + y * TMath::Cos((-8. / 180.) * TMath::Pi());
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

  if (TMath::Abs(track.tx()) > maxTx || TMath::Abs(track.ty()) > maxTy || track.chiSquaredOverNDF() * ndf > maxChi2 ||
      track.numberOfPointsUsedForFit() < minNumberOfPlanesForTrack_ ||
      track.numberOfPointsUsedForFit() > maxNumberOfPlanesForTrack_ ||
      y > fiducialYHigh_[std::pair<int, int>(arm, station)] || y < fiducialYLow_[std::pair<int, int>(arm, station)] ||
      x < fiducialXLow_[std::pair<int, int>(arm, station)]) {
    return true;
  } else {
    if (recoInfoCut_ != 5) {
      if (recoInfoCut_ != -1) {
        if ((int)track.pixelTrackRecoInfo() != recoInfoCut_) {
          return true;
        } else {
          return false;
        }
      } else {
        if ((int)track.pixelTrackRecoInfo() != 0 && (int)track.pixelTrackRecoInfo() != 2) {
          return true;
        }

        else {
          return false;
        }
      }
    } else {
      return false;
    }
  }
}

float EfficiencyTool_2018DQMWorker::Aperture(Float_t xangle, Int_t arm, TString era) {
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
    if (arm == 1)  // Note - 1 strip RP was not in, so no aperture cuts derived
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
DEFINE_FWK_MODULE(EfficiencyTool_2018DQMWorker);