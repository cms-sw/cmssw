// -*- C++ -*-
//
// Package:    RecoPPS/RPixEfficiencyTools
// Class:      ReferenceAnalysisDQMWorker
//
/**\class ReferenceAnalysisDQMWorker ReferenceAnalysisDQMWorker.cc
 RecoPPS/RPixEfficiencyTools/plugins/ReferenceAnalysisDQMWorker.cc

 Description: [one line class summary]

 Implementation:
                 [Notes on implementation]
*/
//
// Original Author:  Andrea Bellora
//         Created:  Wed, 22 Aug 2017 09:55:05 GMT
//
//

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "CondFormats/RunInfo/interface/LHCInfo.h"
#include "CondFormats/DataRecord/interface/LHCInfoRcd.h"

#include "DataFormats/Common/interface/DetSetVector.h"

#include "DataFormats/CTPPSDetId/interface/CTPPSDetId.h"
#include "DataFormats/CTPPSReco/interface/CTPPSPixelLocalTrack.h"

#include "DataFormats/ProtonReco/interface/ForwardProton.h"
#include "DataFormats/ProtonReco/interface/ForwardProtonFwd.h"
#include "DataFormats/CTPPSReco/interface/CTPPSLocalTrackLite.h"

#include "Geometry/VeryForwardGeometryBuilder/interface/CTPPSGeometry.h"
#include "Geometry/Records/interface/VeryForwardRealGeometryRecord.h"

#include <TEfficiency.h>
#include <TF1.h>
#include <TFile.h>
#include <TGraphErrors.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TMath.h>
#include <TObjArray.h>

#include <string>
#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <exception>
#include <fstream>
#include <memory>
#include <set>

class ReferenceAnalysisDQMWorker : public DQMEDAnalyzer {
public:
  explicit ReferenceAnalysisDQMWorker(const edm::ParameterSet &);
  ~ReferenceAnalysisDQMWorker();
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

protected:
  virtual void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  virtual void dqmBeginRun(edm::Run const &, edm::EventSetup const &) override;

private:
  virtual void analyze(const edm::Event &, const edm::EventSetup &) override;

  bool Cut(CTPPSPixelLocalTrack track, int arm, int station);
  bool CutForEfficiencyVsXi(CTPPSLocalTrackLite track);

  float probabilityCalculation(const std::map<uint32_t, float> &planeEfficiency);

  float probabilityNplanesBlind(const std::vector<uint32_t> &inputPlaneList,
                                int numberToExtract,
                                const std::map<unsigned, float> &planeEfficiency);

  void getPlaneCombinations(const std::vector<uint32_t> &inputPlaneList,
                            uint32_t numberToExtract,
                            std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> &planesExtractedAndNot);

  bool debug_ = false;

  edm::ESGetToken<CTPPSGeometry, VeryForwardRealGeometryRecord> geomEsToken_;

  // Data to get
  edm::EDGetTokenT<edm::DetSetVector<CTPPSPixelLocalTrack>> pixelLocalTrackToken_;

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

  int binGroupingX = 1;
  int binGroupingY = 1;

  int mapXbins = 200 / binGroupingX;
  float mapXmin;
  float mapXmax;
  int mapYbins = 240 / binGroupingY;
  float mapYmin = -16.;
  float mapYmax = 8.;

  double xiBins = 44;
  double xiMin = 0;
  double xiMax = 0.22;

  double angleBins = 100;
  double angleMin = -0.03;
  double angleMax = 0.03;

  bool useMultiRPEfficiency_ = false;
  // Use interPot efficiency map instead of InterpotEfficiency
  bool useInterpotEfficiency_ = false;
  // Use multiRP protons
  bool useMultiRPProtons_ = false;

  std::map<CTPPSDetId, uint32_t> trackMux_;
  std::vector<CTPPSPixelDetId> detectorIdVector_;
  std::vector<CTPPSPixelDetId> romanPotIdVector_;

  std::vector<uint32_t> listOfPlanes_;

  std::map<CTPPSPixelDetId, int> binAlignmentParameters = {{CTPPSPixelDetId(0, 0, 3), 0},
                                                           {CTPPSPixelDetId(0, 2, 3), 0},
                                                           {CTPPSPixelDetId(1, 0, 3), 0},
                                                           {CTPPSPixelDetId(1, 2, 3), 0}};

  // output histograms
  std::map<CTPPSPixelDetId, TH2D *> h2PlaneEfficiencyMap_;

  std::map<CTPPSPixelDetId, MonitorElement *> h2RefinedTrackEfficiency_;
  std::map<CTPPSPixelDetId, MonitorElement *> h2TrackHitDistribution_;
  std::map<CTPPSPixelDetId, MonitorElement *> h2RefinedTrackEfficiency_rotated;
  std::map<CTPPSPixelDetId, MonitorElement *> h2TrackHitDistribution_rotated;

  std::map<CTPPSPixelDetId, MonitorElement *> h1Xi_;
  std::map<CTPPSPixelDetId, MonitorElement *> h1EfficiencyVsXi_;
  std::map<CTPPSPixelDetId, MonitorElement *> h1RecoMethod_;
  std::map<CTPPSPixelDetId, MonitorElement *> h1Tx_;
  std::map<CTPPSPixelDetId, MonitorElement *> h1Ty_;
  std::map<CTPPSPixelDetId, MonitorElement *> h1EfficiencyVsTx_;
  std::map<CTPPSPixelDetId, MonitorElement *> h1EfficiencyVsTy_;
  std::map<CTPPSPixelDetId, MonitorElement *> h1Efficiency_;
  std::map<CTPPSPixelDetId, MonitorElement *> h2ProtonHitDistribution_;
  // file to insert the output hists in
  TFile *efficiencyFile_;

  std::vector<double> fiducialXLowVector_;
  std::vector<double> fiducialXHighVector_;
  std::vector<double> fiducialYLowVector_;
  std::vector<double> fiducialYHighVector_;
  std::map<std::pair<int, int>, double> fiducialXLow_;
  std::map<std::pair<int, int>, double> fiducialXHigh_;
  std::map<std::pair<int, int>, double> fiducialYLow_;
  std::map<std::pair<int, int>, double> fiducialYHigh_;

  double detectorTiltAngle;
  double detectorRotationAngle;

  edm::EDGetTokenT<reco::ForwardProtonCollection> singleRPprotonsToken_;
  edm::EDGetTokenT<reco::ForwardProtonCollection> multiRPprotonsToken_;
};

ReferenceAnalysisDQMWorker::ReferenceAnalysisDQMWorker(const edm::ParameterSet &iConfig)
    : geomEsToken_(esConsumes<edm::Transition::BeginRun>()) {
  producerTag = iConfig.getUntrackedParameter<std::string>("producerTag");

  pixelLocalTrackToken_ =
      consumes<edm::DetSetVector<CTPPSPixelLocalTrack>>(edm::InputTag("ctppsPixelLocalTracks", "", producerTag));

  singleRPprotonsToken_ =
      consumes<reco::ForwardProtonCollection>(edm::InputTag("ctppsProtons", "singleRP", producerTag));
  multiRPprotonsToken_ = consumes<reco::ForwardProtonCollection>(edm::InputTag("ctppsProtons", "multiRP", producerTag));

  efficiencyFileName_ = iConfig.getUntrackedParameter<std::string>("efficiencyFileName");
  outputFileName_ = iConfig.getUntrackedParameter<std::string>("outputFileName");
  minNumberOfPlanesForEfficiency_ = iConfig.getParameter<int>("minNumberOfPlanesForEfficiency");
  minNumberOfPlanesForTrack_ = iConfig.getParameter<int>("minNumberOfPlanesForTrack");
  minTracksPerEvent = iConfig.getParameter<int>("minTracksPerEvent");
  maxTracksPerEvent = iConfig.getParameter<int>("maxTracksPerEvent");
  binGroupingX = iConfig.getUntrackedParameter<int>("binGroupingX");
  binGroupingY = iConfig.getUntrackedParameter<int>("binGroupingY");

  useMultiRPEfficiency_ = iConfig.getUntrackedParameter<bool>("useMultiRPEfficiency");
  useInterpotEfficiency_ = iConfig.getUntrackedParameter<bool>("useInterPotEfficiency");
  useMultiRPProtons_ = iConfig.getUntrackedParameter<bool>("useMultiRPProtons");

  fiducialXLowVector_ = iConfig.getUntrackedParameter<std::vector<double>>("fiducialXLow");
  fiducialXHighVector_ = iConfig.getUntrackedParameter<std::vector<double>>("fiducialXHigh");
  fiducialYLowVector_ = iConfig.getUntrackedParameter<std::vector<double>>("fiducialYLow");
  fiducialYHighVector_ = iConfig.getUntrackedParameter<std::vector<double>>("fiducialYHigh");
  fiducialXLow_ = {
      {std::pair<int, int>(0, 0), fiducialXLowVector_.at(0)},
      {std::pair<int, int>(0, 2), fiducialXLowVector_.at(1)},
      {std::pair<int, int>(1, 0), fiducialXLowVector_.at(2)},
      {std::pair<int, int>(1, 2), fiducialXLowVector_.at(3)},
  };
  fiducialXHigh_ = {
      {std::pair<int, int>(0, 0), fiducialXHighVector_.at(0)},
      {std::pair<int, int>(0, 2), fiducialXHighVector_.at(1)},
      {std::pair<int, int>(1, 0), fiducialXHighVector_.at(2)},
      {std::pair<int, int>(1, 2), fiducialXHighVector_.at(3)},
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
  detectorTiltAngle = iConfig.getUntrackedParameter<double>("detectorTiltAngle");
  mapXmin = 0. * TMath::Cos(detectorTiltAngle / 180. * TMath::Pi());
  mapXmax = 30. * TMath::Cos(detectorTiltAngle / 180. * TMath::Pi());
  detectorRotationAngle = iConfig.getUntrackedParameter<double>("detectorRotationAngle");
}

ReferenceAnalysisDQMWorker::~ReferenceAnalysisDQMWorker() {}

void ReferenceAnalysisDQMWorker::bookHistograms(DQMStore::IBooker &ibooker,
                                                edm::Run const &run,
                                                edm::EventSetup const &eventSetup) {
  ibooker.cd();
  auto efficiencyFile_ = new TFile(efficiencyFileName_.data(), "READ");
  if (!efficiencyFile_->IsOpen()) {
    std::cout << "No efficiency file available!" << std::endl;
    throw 1;
  }
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

    std::string h2planeEfficiencyMapName = Form(
        "DQMData/Run 999999/Arm%i/Run summary/st%i/rp%i/"
        "h2EfficiencyMap_arm%i_st%i_rp%i_pl%i",
        arm,
        station,
        rp,
        arm,
        station,
        rp,
        plane);

    if (efficiencyFile_->Get(h2planeEfficiencyMapName.data())) {
      h2PlaneEfficiencyMap_[detId] = (TH2D *)efficiencyFile_->Get(h2planeEfficiencyMapName.data());
    }

    if (plane == 0) {  //to make sure that these trackHit historgrams will be booked just once
      h2TrackHitDistribution_[detId] =
          ibooker.book2DD(Form("h2TrackHitDistribution_arm%i_st%i_rp%i", arm, station, rp),
                          Form("h2TrackHitDistribution_arm%i_st%i_rp%i;x (mm);y (mm)", arm, station, rp),
                          mapXbins,
                          mapXmin,
                          mapXmax,
                          mapYbins,
                          mapYmin,
                          mapYmax);

      ibooker.book2DD(Form("h2RefinedTrackEfficiency_arm%i_st%i_rp%i", arm, station, rp),
                      Form("h2RefinedTrackEfficiency_arm%i_st%i_rp%i;x (mm);y (mm)", arm, station, rp),
                      mapXbins,
                      mapXmin,
                      mapXmax,
                      mapYbins,
                      mapYmin,
                      mapYmax);

      h2RefinedTrackEfficiency_[detId] =
          ibooker.book2DD(Form("h2RefinedTrackEfficiencyBuffer_arm%i_st%i_rp%i", arm, station, rp),
                          Form("h2RefinedTrackEfficiencyBuffer_arm%i_st%i_rp%i;x (mm);y (mm)", arm, station, rp),
                          mapXbins,
                          mapXmin,
                          mapXmax,
                          mapYbins,
                          mapYmin,
                          mapYmax);

      if (station == 0) {
        h2TrackHitDistribution_rotated[detId] =
            ibooker.book2DD(Form("h2TrackHitDistribution_rotated_arm%i_st%i_rp%i", arm, station, rp),
                            Form("h2TrackHitDistribution_rotated_arm%i_st%i_rp%i;x (mm);y (mm)", arm, station, rp),
                            mapXbins,
                            mapXmin,
                            mapXmax,
                            mapYbins,
                            mapYmin,
                            mapYmax);

        h2RefinedTrackEfficiency_rotated[detId] =
            ibooker.book2DD(Form("h2RefinedTrackEfficiencyBuffer_rotated_arm%i_st%i_rp%i", arm, station, rp),
                            Form("h2RefinedTrackEfficiencyBuffer_rotated_arm%i_st%i_rp%i;x "
                                 "(mm);y (mm)",
                                 arm,
                                 station,
                                 rp),
                            mapXbins,
                            mapXmin,
                            mapXmax,
                            mapYbins,
                            mapYmin,
                            mapYmax);

        ibooker.book2DD(Form("h2RefinedTrackEfficiency_rotated_arm%i_st%i_rp%i", arm, station, rp),
                        Form("h2RefinedTrackEfficiency_rotated_arm%i_st%i_rp%i;x "
                             "(mm);y (mm)",
                             arm,
                             station,
                             rp),
                        mapXbins,
                        mapXmin,
                        mapXmax,
                        mapYbins,
                        mapYmin,
                        mapYmax);
      }

      h1Xi_[detId] = ibooker.book1DD(Form("h1Xi_arm%i_st%i_rp%i", arm, station, rp),
                                     Form("h1Xi_arm%i_st%i_rp%i;#xi", arm, station, rp),
                                     xiBins,
                                     xiMin,
                                     xiMax);
      h1Tx_[detId] = ibooker.book1DD(Form("h1Tx_arm%i_st%i_rp%i", arm, station, rp),
                                     Form("h1Tx_arm%i_st%i_rp%i;Tx", arm, station, rp),
                                     angleBins,
                                     angleMin,
                                     angleMax);
      h1Ty_[detId] = ibooker.book1DD(Form("h1Ty_arm%i_st%i_rp%i", arm, station, rp),
                                     Form("h1Ty_arm%i_st%i_rp%i;Ty", arm, station, rp),
                                     angleBins,
                                     angleMin,
                                     angleMax);
      h1RecoMethod_[detId] = ibooker.book1DD(Form("h1RecoMethod_arm%i_st%i_rp%i", arm, station, rp),
                                             Form("h1RecoMethod_arm%i_st%i_rp%i", arm, station, rp),
                                             3,
                                             -1,
                                             1);
      h1EfficiencyVsXi_[detId] =
          ibooker.book1DD(Form("h1EfficiencyVsXi_arm%i_st%i_rp%i", arm, station, rp),
                          Form("h1EfficiencyVsXi_arm%i_st%i_rp%i;#xi;Efficiency", arm, station, rp),
                          xiBins,
                          xiMin,
                          xiMax);

      ibooker.book1DD(Form("h1EfficiencyVsXiFinal_arm%i_st%i_rp%i", arm, station, rp),
                      Form("h1EfficiencyVsXiFinal_arm%i_st%i_rp%i;#xi;Efficiency", arm, station, rp),
                      xiBins,
                      xiMin,
                      xiMax);

      h1EfficiencyVsTx_[detId] = ibooker.book1DD(Form("h1EfficiencyVsTx_arm%i_st%i_rp%i", arm, station, rp),
                                                 Form("h1EfficiencyVsTx_arm%i_st%i_rp%i;Tx", arm, station, rp),
                                                 angleBins,
                                                 angleMin,
                                                 angleMax);

      ibooker.book1DD(Form("h1EfficiencyVsTxFinal_arm%i_st%i_rp%i", arm, station, rp),
                      Form("h1EfficiencyVsTxFinal_arm%i_st%i_rp%i;Tx", arm, station, rp),
                      angleBins,
                      angleMin,
                      angleMax);

      h1EfficiencyVsTy_[detId] = ibooker.book1DD(Form("h1EfficiencyVsTy_arm%i_st%i_rp%i", arm, station, rp),
                                                 Form("h1EfficiencyVsTy_arm%i_st%i_rp%i;Ty", arm, station, rp),
                                                 angleBins,
                                                 angleMin,
                                                 angleMax);

      ibooker.book1DD(Form("h1EfficiencyVsTyFinal_arm%i_st%i_rp%i", arm, station, rp),
                      Form("h1EfficiencyVsTyFinal_arm%i_st%i_rp%i;Ty", arm, station, rp),
                      angleBins,
                      angleMin,
                      angleMax);

      h1Efficiency_[detId] = ibooker.book1DD(Form("h1Efficiency_arm%i_st%i_rp%i", arm, station, rp),
                                             Form("h1Efficiency_arm%i_st%i_rp%i;Ty", arm, station, rp),
                                             100,
                                             0,
                                             1);
      h2ProtonHitDistribution_[detId] =
          ibooker.book2DD(Form("h2ProtonHitDistribution_arm%i_st%i_rp%i", arm, station, rp),
                          Form("h2ProtonHitDistribution_arm%i_st%i_rp%i;x (mm);y (mm)", arm, station, rp),
                          mapXbins,
                          mapXmin,
                          mapXmax,
                          mapYbins,
                          mapYmin,
                          mapYmax);
    }
  }
  listOfPlanes_.assign(planesSet.begin(), planesSet.end());
  efficiencyFile_->Close();
  delete efficiencyFile_;
}

void ReferenceAnalysisDQMWorker::dqmBeginRun(edm::Run const &, edm::EventSetup const &eventSetup) {}

void ReferenceAnalysisDQMWorker::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  using namespace edm;

  Handle<edm::DetSetVector<CTPPSPixelLocalTrack>> pixelLocalTracks;
  iEvent.getByToken(pixelLocalTrackToken_, pixelLocalTracks);

  for (const auto &rpPixeltrack : *pixelLocalTracks) {
    if ((uint32_t)minTracksPerEvent > rpPixeltrack.size() || rpPixeltrack.size() > (uint32_t)maxTracksPerEvent)
      continue;
    CTPPSPixelDetId rpId = CTPPSPixelDetId(rpPixeltrack.detId());
    uint32_t arm = rpId.arm();
    uint32_t station = rpId.station();

    // Shift Xmin and Xmax to align bins with sensor edge
    double binSize = (mapXmax - mapXmin) / mapXbins;
    mapXmin += binAlignmentParameters[rpId] * binSize / 150.;
    mapXmax += binAlignmentParameters[rpId] * binSize / 150.;

    for (const auto &pixeltrack : rpPixeltrack) {
      if (Cut(pixeltrack, arm, station) || !pixeltrack.isValid())
        continue;
      float pixelX0 = pixeltrack.x0();
      float pixelY0 = pixeltrack.y0();
      // Rotating St0 tracks
      float pixelX0_rotated = 0;
      float pixelY0_rotated = 0;
      if (station == 0) {
        pixelX0_rotated = pixeltrack.x0() * TMath::Cos((detectorRotationAngle / 180.) * TMath::Pi()) -
                          pixeltrack.y0() * TMath::Sin((detectorRotationAngle / 180.) * TMath::Pi());
        pixelY0_rotated = pixeltrack.x0() * TMath::Sin((detectorRotationAngle / 180.) * TMath::Pi()) +
                          pixeltrack.y0() * TMath::Cos((detectorRotationAngle / 180.) * TMath::Pi());
      }

      edm::DetSetVector<CTPPSPixelFittedRecHit> fittedHits = pixeltrack.hits();
      h2TrackHitDistribution_[rpId]->Fill(pixelX0, pixelY0);
      if (station == 0)
        h2TrackHitDistribution_rotated[rpId]->Fill(pixelX0_rotated, pixelY0_rotated);
      std::map<uint32_t, float> planeEfficiency;
      for (const auto &planeHits : fittedHits) {
        CTPPSPixelDetId planeId = CTPPSPixelDetId(planeHits.detId());
        uint32_t plane = planeId.plane();
        for (const auto &hit : planeHits) {
          double hitX0 = hit.globalCoordinates().x() + hit.xResidual();
          double hitY0 = hit.globalCoordinates().y() + hit.yResidual();
          uint32_t xBin = h2PlaneEfficiencyMap_[planeId]->GetXaxis()->FindBin(hitX0);
          uint32_t yBin = h2PlaneEfficiencyMap_[planeId]->GetYaxis()->FindBin(hitY0);
          planeEfficiency[plane] = h2PlaneEfficiencyMap_[planeId]->GetBinContent(xBin, yBin);
          if (debug_)
            std::cout << "Hit coordinates: (" << hitX0 << "," << hitY0 << ")\n Hit bins (" << xBin << "." << yBin << ")"
                      << std::endl;
        }  // for each hit
      }    // for each hit collection
      for (const auto &plane : listOfPlanes_) {
        if (planeEfficiency.find(plane) == planeEfficiency.end()) {
          planeEfficiency[plane] = 0.;
        }
        if (debug_)
          std::cout << "Plane " << plane << " efficiency: " << planeEfficiency[plane] << std::endl;
      }
      float efficiency = probabilityCalculation(planeEfficiency);
      h2RefinedTrackEfficiency_[rpId]->Fill(pixelX0, pixelY0, efficiency);
      if (station == 0)
        h2RefinedTrackEfficiency_rotated[rpId]->Fill(pixelX0_rotated, pixelY0_rotated, efficiency);
      if (debug_)
        std::cout << "Track passing through: (" << pixelX0 << "," << pixelY0 << ")\n Efficiency: " << efficiency
                  << std::endl;
    }  // for each track in the collection
  }    // for each track collection

  //EFFICIENCY VS XI
  Handle<reco::ForwardProtonCollection> protons;
  if (useMultiRPProtons_) {
    iEvent.getByToken(multiRPprotonsToken_, protons);
  } else {
    iEvent.getByToken(singleRPprotonsToken_, protons);
  }

  for (auto &proton : *protons) {
    if (!proton.validFit())
      continue;
    CTPPSPixelDetId pixelDetId(0, 0);  // initialization
    for (auto &track_ptr : proton.contributingLocalTracks()) {
      CTPPSLocalTrackLite track = *track_ptr;

      CTPPSDetId detId = CTPPSDetId(track.rpId());
      trackMux_[detId]++;

      try {
        pixelDetId = CTPPSPixelDetId(detId.rawId());
      } catch (cms::Exception &e) {
        if (debug_)
          std::cout << "Caught exception!" << std::endl;
        continue;
      }
      if (std::find(romanPotIdVector_.begin(), romanPotIdVector_.end(), pixelDetId) == romanPotIdVector_.end())
        continue;
      if (CutForEfficiencyVsXi(track))
        continue;

      double trackX0 = track.x();
      double trackY0 = track.y();
      double trackTx = track.tx();
      double trackTy = track.ty();

      uint32_t xBin = h2RefinedTrackEfficiency_[pixelDetId]->getTH2D()->GetXaxis()->FindBin(trackX0);
      uint32_t yBin = h2RefinedTrackEfficiency_[pixelDetId]->getTH2D()->GetYaxis()->FindBin(trackY0);
      double trackEfficiency = h2RefinedTrackEfficiency_[pixelDetId]->getTH2D()->GetBinContent(xBin, yBin);

      if (debug_) {
        std::cout << "Contributing tracks: " << proton.contributingLocalTracks().size() << std::endl;
        std::cout << detId << std::endl;
        std::cout << "Arm: " << pixelDetId.arm() << " Station: " << pixelDetId.station() << std::endl;
        std::cout << "RecoInfo: " << (int)(track).pixelTrackRecoInfo() << std::endl;
        std::cout << "Track efficiency: " << trackEfficiency << std::endl;
      }

      h1EfficiencyVsXi_[pixelDetId]->Fill(proton.xi(), trackEfficiency);
      h1Xi_[pixelDetId]->Fill(proton.xi());
      h1RecoMethod_[pixelDetId]->Fill((int)proton.method());

      h1Tx_[pixelDetId]->Fill(trackTx);
      h1EfficiencyVsTx_[pixelDetId]->Fill(trackTx, trackEfficiency);
      h1Ty_[pixelDetId]->Fill(trackTy);
      h1EfficiencyVsTy_[pixelDetId]->Fill(trackTy, trackEfficiency);
      h1Efficiency_[pixelDetId]->Fill(trackEfficiency);
      h2ProtonHitDistribution_[pixelDetId]->Fill(trackX0, trackY0);
    }
  }
}

void ReferenceAnalysisDQMWorker::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

bool ReferenceAnalysisDQMWorker::Cut(CTPPSPixelLocalTrack track, int arm, int station) {
  double maxTx = 0.02;
  double maxTy = 0.02;
  double maxChi2 = TMath::ChisquareQuantile(0.95, track.ndf());
  double x = track.x0();
  double y = track.y0();
  float pixelX0_rotated = 0;
  float pixelY0_rotated = 0;
  if (station == 0) {
    pixelX0_rotated = x * TMath::Cos((detectorRotationAngle / 180.) * TMath::Pi()) -
                      y * TMath::Sin((detectorRotationAngle / 180.) * TMath::Pi());
    pixelY0_rotated = x * TMath::Sin((detectorRotationAngle / 180.) * TMath::Pi()) +
                      y * TMath::Cos((detectorRotationAngle / 180.) * TMath::Pi());
    x = pixelX0_rotated;
    y = pixelY0_rotated;
  }

  if (TMath::Abs(track.tx()) > maxTx || TMath::Abs(track.ty()) > maxTy ||
      track.numberOfPointsUsedForFit() < minNumberOfPlanesForTrack_ ||
      track.numberOfPointsUsedForFit() > maxNumberOfPlanesForTrack_ ||
      track.chiSquaredOverNDF() * track.ndf() > maxChi2 || y > fiducialYHigh_[std::pair<int, int>(arm, station)] ||
      y < fiducialYLow_[std::pair<int, int>(arm, station)] || x < fiducialXLow_[std::pair<int, int>(arm, station)])
    return true;
  else
    return false;
}

float ReferenceAnalysisDQMWorker::probabilityCalculation(const std::map<uint32_t, float> &planeEfficiency) {
  int minNumberOfBlindPlanes = 3;
  int maxNumberOfBlindPlanes = listOfPlanes_.size();
  float rpEfficiency = 1.;

  for (uint32_t i = (uint32_t)minNumberOfBlindPlanes; i <= (uint32_t)maxNumberOfBlindPlanes; i++) {
    rpEfficiency -= probabilityNplanesBlind(listOfPlanes_, i, planeEfficiency);
  }
  return rpEfficiency;
}

float ReferenceAnalysisDQMWorker::probabilityNplanesBlind(const std::vector<uint32_t> &inputPlaneList,
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

void ReferenceAnalysisDQMWorker::getPlaneCombinations(
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

bool ReferenceAnalysisDQMWorker::CutForEfficiencyVsXi(CTPPSLocalTrackLite track) {
  CTPPSDetId detId = CTPPSDetId(track.rpId());
  uint32_t arm = detId.arm();
  uint32_t station = detId.station();
  uint32_t ndf = 2 * track.numberOfPointsUsedForFit() - 4;
  double x = track.x();
  double y = track.y();

  double maxTx = 0.02;
  double maxTy = 0.02;
  double maxChi2 = TMath::ChisquareQuantile(0.95, ndf);

  if (TMath::Abs(track.tx()) > maxTx || TMath::Abs(track.ty()) > maxTy || track.chiSquaredOverNDF() * ndf > maxChi2 ||
      track.numberOfPointsUsedForFit() < minNumberOfPlanesForTrack_ ||
      track.numberOfPointsUsedForFit() > maxNumberOfPlanesForTrack_ ||
      y > fiducialYHigh_[std::pair<int, int>(arm, station)] || y < fiducialYLow_[std::pair<int, int>(arm, station)] ||
      x < fiducialXLow_[std::pair<int, int>(arm, station)] || x > fiducialXHigh_[std::pair<int, int>(arm, station)] ||
      ((int)track.pixelTrackRecoInfo() != 0 && (int)track.pixelTrackRecoInfo() != 2))
    return true;
  else
    return false;
}

// define this as a plug-in
DEFINE_FWK_MODULE(ReferenceAnalysisDQMWorker);