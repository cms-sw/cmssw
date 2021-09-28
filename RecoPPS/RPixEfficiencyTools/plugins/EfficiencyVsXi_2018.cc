// -*- C++ -*-
//
// Package:    RecoPPS/RPixEfficiencyTools
// Class:      EfficiencyVsXi_2018
//
/**\class EfficiencyVsXi_2018 EfficiencyVsXi_2018.cc
 RecoPPS/RPixEfficiencyTools/plugins/EfficiencyVsXi_2018.cc

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

#include <TEfficiency.h>
#include <TF1.h>
#include <TFile.h>
#include <TGraphErrors.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TMath.h>
#include <TObjArray.h>

class EfficiencyVsXi_2018
    : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit EfficiencyVsXi_2018(const edm::ParameterSet &);
  ~EfficiencyVsXi_2018();
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
  bool Cut(CTPPSLocalTrackLite track);

  bool debug_ = false;

  // z position of the pots (mm)
  std::map<CTPPSDetId, double> Z = {
      {CTPPSDetId(3, 0, 0, 3), -212550}, // strips, arm0, station0, rp3
      {CTPPSDetId(3, 1, 0, 3), 212550},  // strips, arm1, station0, rp3
      {CTPPSDetId(4, 0, 2, 3), -219550}, // pixels, arm0, station2, rp3
      {CTPPSDetId(4, 1, 2, 3), 219550}}; // pixels, arm1, station2, rp3

  // Data to get
  edm::EDGetTokenT<reco::ForwardProtonCollection> singleRPprotonsToken_;
  edm::EDGetTokenT<reco::ForwardProtonCollection> multiRPprotonsToken_;

  // Parameter set
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
  double xiMin = 0;
  double xiMax = 0.22;
  double angleBins = 100;
  double angleMin = -0.03;
  double angleMax = 0.03;

  // std::map<CTPPSPixelDetId, int> binAlignmentParameters = {
  //     {CTPPSPixelDetId(0, 0, 3), 0},
  //     {CTPPSPixelDetId(0, 2, 3), 0},
  //     {CTPPSPixelDetId(1, 0, 3), 0},
  //     {CTPPSPixelDetId(1, 2, 3), 0}};

  // output histograms
  std::map<CTPPSPixelDetId, TH2D *> h2RefinedTrackEfficiency_;
  std::map<CTPPSPixelDetId, TH1D *> h1Xi_;
  std::map<CTPPSPixelDetId, TH1D *> h1EfficiencyVsXi_;
  std::map<CTPPSPixelDetId, TH1D *> h1RecoMethod_;
  std::map<CTPPSPixelDetId, TH1D *> h1Tx_;
  std::map<CTPPSPixelDetId, TH1D *> h1Ty_;
  std::map<CTPPSPixelDetId, TH1D *> h1EfficiencyVsTx_;
  std::map<CTPPSPixelDetId, TH1D *> h1EfficiencyVsTy_;
  std::map<CTPPSPixelDetId, TH1D *> h1Efficiency_;
  std::map<CTPPSPixelDetId, TH2D *> h2ProtonHitDistribution_;

  std::map<CTPPSDetId, uint32_t> trackMux_;

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

  // Use multiRP efficiency map instead of InterpotEfficiency
  bool useMultiRPEfficiency_ = false;
  // Use interPot efficiency map instead of InterpotEfficiency
  bool useInterpotEfficiency_ = false;
  // Use multiRP protons
  bool useMultiRPProtons_ = false;
};

EfficiencyVsXi_2018::EfficiencyVsXi_2018(const edm::ParameterSet &iConfig) {
  usesResource("TFileService");

  producerTag = iConfig.getUntrackedParameter<std::string>("producerTag");

  singleRPprotonsToken_ = consumes<reco::ForwardProtonCollection>(
      edm::InputTag("ctppsProtons", "singleRP", producerTag));
  multiRPprotonsToken_ = consumes<reco::ForwardProtonCollection>(
      edm::InputTag("ctppsProtons", "multiRP", producerTag));

  efficiencyFileName_ =
      iConfig.getUntrackedParameter<std::string>("efficiencyFileName");
  outputFileName_ =
      iConfig.getUntrackedParameter<std::string>("outputFileName");
  minNumberOfPlanesForEfficiency_ =
      iConfig.getParameter<int>("minNumberOfPlanesForEfficiency"); // UNUSED!
  minNumberOfPlanesForTrack_ =
      iConfig.getParameter<int>("minNumberOfPlanesForTrack");
  minTracksPerEvent = iConfig.getParameter<int>("minTracksPerEvent"); // UNUSED!
  maxTracksPerEvent = iConfig.getParameter<int>("maxTracksPerEvent"); // UNUSED!
  binGroupingX = iConfig.getUntrackedParameter<int>("binGroupingX");  // UNUSED!
  binGroupingY = iConfig.getUntrackedParameter<int>("binGroupingY");  // UNUSED!
  fiducialXLowVector_ =
      iConfig.getUntrackedParameter<std::vector<double>>("fiducialXLow");
  fiducialXHighVector_ =
      iConfig.getUntrackedParameter<std::vector<double>>("fiducialXHigh");
  fiducialYLowVector_ =
      iConfig.getUntrackedParameter<std::vector<double>>("fiducialYLow");
  fiducialYHighVector_ =
      iConfig.getUntrackedParameter<std::vector<double>>("fiducialYHigh");
  useMultiRPEfficiency_ =
      iConfig.getUntrackedParameter<bool>("useMultiRPEfficiency");
  useInterpotEfficiency_ =
      iConfig.getUntrackedParameter<bool>("useInterPotEfficiency");
  useMultiRPProtons_ = iConfig.getUntrackedParameter<bool>("useMultiRPProtons");
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
}

EfficiencyVsXi_2018::~EfficiencyVsXi_2018() {
  for (auto &rpId : romanPotIdVector_) {
    delete h2RefinedTrackEfficiency_[rpId];
    delete h1RecoMethod_[rpId];
    delete h1Xi_[rpId];
    delete h1EfficiencyVsXi_[rpId];
    delete h1Tx_[rpId];
    delete h1Ty_[rpId];
    delete h1EfficiencyVsTx_[rpId];
    delete h1EfficiencyVsTy_[rpId];
    delete h1Efficiency_[rpId];
    delete h2ProtonHitDistribution_[rpId];
  }
}

void EfficiencyVsXi_2018::analyze(const edm::Event &iEvent,
                                  const edm::EventSetup &iSetup) {
  using namespace edm;

  Handle<reco::ForwardProtonCollection> protons;
  if (useMultiRPProtons_) {
    iEvent.getByToken(multiRPprotonsToken_, protons);
  } else {
    iEvent.getByToken(singleRPprotonsToken_, protons);
  }

  trackMux_.clear();

  for (auto &proton : *protons) {
    if (!proton.validFit())
      continue;
    CTPPSPixelDetId pixelDetId(0, 0); // initialization
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
      if (std::find(romanPotIdVector_.begin(), romanPotIdVector_.end(),
                    pixelDetId) == romanPotIdVector_.end())
        continue;
      if (Cut(track))
        continue;

      uint32_t arm = pixelDetId.arm();
      uint32_t station = pixelDetId.station();
      uint32_t rp = pixelDetId.rp();

      if (h1Xi_.find(pixelDetId) == h1Xi_.end()) {
        h1Xi_[pixelDetId] =
            new TH1D(Form("h1Xi_arm%i_st%i_rp%i", arm, station, rp),
                     Form("h1Xi_arm%i_st%i_rp%i;#xi", arm, station, rp), xiBins,
                     xiMin, xiMax);
        h1Tx_[pixelDetId] =
            new TH1D(Form("h1Tx_arm%i_st%i_rp%i", arm, station, rp),
                     Form("h1Tx_arm%i_st%i_rp%i;Tx", arm, station, rp),
                     angleBins, angleMin, angleMax);
        h1Ty_[pixelDetId] =
            new TH1D(Form("h1Ty_arm%i_st%i_rp%i", arm, station, rp),
                     Form("h1Ty_arm%i_st%i_rp%i;Ty", arm, station, rp),
                     angleBins, angleMin, angleMax);
        h1RecoMethod_[pixelDetId] = new TH1D(
            Form("h1RecoMethod_arm%i_st%i_rp%i", arm, station, rp),
            Form("h1RecoMethod_arm%i_st%i_rp%i", arm, station, rp), 3, -1, 1);
        h1EfficiencyVsXi_[pixelDetId] =
            new TH1D(Form("h1EfficiencyVsXi_arm%i_st%i_rp%i", arm, station, rp),
                     Form("h1EfficiencyVsXi_arm%i_st%i_rp%i;#xi;Efficiency",
                          arm, station, rp),
                     xiBins, xiMin, xiMax);
        h1EfficiencyVsTx_[pixelDetId] = new TH1D(
            Form("h1EfficiencyVsTx_arm%i_st%i_rp%i", arm, station, rp),
            Form("h1EfficiencyVsTx_arm%i_st%i_rp%i;Tx", arm, station, rp),
            angleBins, angleMin, angleMax);
        h1EfficiencyVsTy_[pixelDetId] = new TH1D(
            Form("h1EfficiencyVsTy_arm%i_st%i_rp%i", arm, station, rp),
            Form("h1EfficiencyVsTy_arm%i_st%i_rp%i;Ty", arm, station, rp),
            angleBins, angleMin, angleMax);
        h1Efficiency_[pixelDetId] =
            new TH1D(Form("h1Efficiency_arm%i_st%i_rp%i", arm, station, rp),
                     Form("h1Efficiency_arm%i_st%i_rp%i;Ty", arm, station, rp),
                     100, 0, 1);
        h2ProtonHitDistribution_[pixelDetId] = new TH2D(
            Form("h2ProtonHitDistribution_arm%i_st%i_rp%i", arm, station, rp),
            Form("h2ProtonHitDistribution_arm%i_st%i_rp%i;x (mm);y (mm)", arm,
                 station, rp),
            mapXbins, mapXmin, mapXmax, mapYbins, mapYmin, mapYmax);
      }
      double trackX0 = track.x();
      double trackY0 = track.y();
      double trackTx = track.tx();
      double trackTy = track.ty();

      uint32_t xBin =
          h2RefinedTrackEfficiency_[pixelDetId]->GetXaxis()->FindBin(trackX0);
      uint32_t yBin =
          h2RefinedTrackEfficiency_[pixelDetId]->GetYaxis()->FindBin(trackY0);
      double trackEfficiency =
          h2RefinedTrackEfficiency_[pixelDetId]->GetBinContent(xBin, yBin);

      if (debug_) {
        std::cout << "Contributing tracks: "
                  << proton.contributingLocalTracks().size() << std::endl;
        std::cout << detId << std::endl;
        std::cout << "Arm: " << pixelDetId.arm()
                  << " Station: " << pixelDetId.station() << std::endl;
        std::cout << "RecoInfo: " << (int)(track).pixelTrackRecoInfo()
                  << std::endl;
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

void EfficiencyVsXi_2018::fillDescriptions(
    edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

void EfficiencyVsXi_2018::beginJob() {
  efficiencyFile_ = new TFile(efficiencyFileName_.data(), "READ");
  if (!efficiencyFile_->IsOpen()) {
    std::cout << "No efficiency file available!" << std::endl;
    throw 1;
  }
  for (auto &arm : listOfArms_) {
    for (auto &station : listOfStations_) {
      CTPPSPixelDetId rpId = CTPPSPixelDetId(arm, station, 3);
      std::string h2RefinedEfficiencyMapName =
          Form("Arm%i_st%i_rp3/"
               "h2RefinedTrackEfficiency_arm%i_st%i_rp3",
               arm, station, arm, station);
      std::string h2InterpotEfficiencyMapName =
          Form("Arm%i_st%i_rp3/"
               "h2InterPotEfficiencyMap_arm%i_st%i_rp3",
               arm, station, arm, station);
      std::string h2InterpotEfficiencyMapMultiRPName =
          Form("Arm%i_st%i_rp3/"
               "h2InterPotEfficiencyMapMultiRP_arm%i_st%i_rp3",
               arm, station, arm, station);

      if (useMultiRPEfficiency_) {
        if (efficiencyFile_->Get(h2InterpotEfficiencyMapMultiRPName.data())) {
          h2RefinedTrackEfficiency_[rpId] =
              new TH2D(*((TH2D *)efficiencyFile_->Get(
                  h2InterpotEfficiencyMapMultiRPName.data())));
          romanPotIdVector_.push_back(rpId);
        }
      } else if (useInterpotEfficiency_) {
        if (efficiencyFile_->Get(h2InterpotEfficiencyMapName.data())) {
          h2RefinedTrackEfficiency_[rpId] =
              new TH2D(*((TH2D *)efficiencyFile_->Get(
                  h2InterpotEfficiencyMapName.data())));
          romanPotIdVector_.push_back(rpId);
        }
      } else {
        if (efficiencyFile_->Get(h2RefinedEfficiencyMapName.data())) {
          h2RefinedTrackEfficiency_[rpId] = new TH2D(*(
              (TH2D *)efficiencyFile_->Get(h2RefinedEfficiencyMapName.data())));
          romanPotIdVector_.push_back(rpId);
        }
      }
    }
  }
  efficiencyFile_->Close();
  delete efficiencyFile_;
}

void EfficiencyVsXi_2018::endJob() {

  TFile *outputFile_ = new TFile(outputFileName_.data(), "RECREATE");
  for (auto &rpId : romanPotIdVector_) {
    uint32_t arm = rpId.arm();
    uint32_t station = rpId.station();
    std::string rpDirName = Form("Arm%i_st%i_rp3", arm, station);
    outputFile_->mkdir(rpDirName.data());
    outputFile_->cd(rpDirName.data());
    h1RecoMethod_[rpId]->Write();
    h1Xi_[rpId]->Write();
    h1EfficiencyVsXi_[rpId]->Divide(h1EfficiencyVsXi_[rpId], h1Xi_[rpId], 1.,
                                    1.);
    h1EfficiencyVsXi_[rpId]->SetMaximum(1.1);
    h1EfficiencyVsXi_[rpId]->SetMinimum(0);
    h1EfficiencyVsXi_[rpId]->Write();

    h1Tx_[rpId]->Write();
    h1Ty_[rpId]->Write();
    h1EfficiencyVsTx_[rpId]->Divide(h1EfficiencyVsTx_[rpId], h1Tx_[rpId], 1.,
                                    1.);
    h1EfficiencyVsTx_[rpId]->SetMaximum(1.1);
    h1EfficiencyVsTx_[rpId]->SetMinimum(0);
    h1EfficiencyVsTx_[rpId]->Write();
    h1EfficiencyVsTy_[rpId]->Divide(h1EfficiencyVsTy_[rpId], h1Ty_[rpId], 1.,
                                    1.);
    h1EfficiencyVsTy_[rpId]->SetMaximum(1.1);
    h1EfficiencyVsTy_[rpId]->SetMinimum(0);
    h1EfficiencyVsTy_[rpId]->Write();
    h1Efficiency_[rpId]->Write();
    h2ProtonHitDistribution_[rpId]->Write();
  }
  outputFile_->Close();
  delete outputFile_;
}

bool EfficiencyVsXi_2018::Cut(CTPPSLocalTrackLite track) {
  CTPPSDetId detId = CTPPSDetId(track.rpId());
  uint32_t arm = detId.arm();
  uint32_t station = detId.station();
  uint32_t ndf = 2 * track.numberOfPointsUsedForFit() - 4;
  double x = track.x();
  double y = track.y();

  double maxTx = 0.02;
  double maxTy = 0.02;
  double maxChi2 = TMath::ChisquareQuantile(0.95, ndf);

  if (TMath::Abs(track.tx()) > maxTx || TMath::Abs(track.ty()) > maxTy ||
      track.chiSquaredOverNDF() * ndf > maxChi2 ||
      track.numberOfPointsUsedForFit() < minNumberOfPlanesForTrack_ ||
      track.numberOfPointsUsedForFit() > maxNumberOfPlanesForTrack_ ||
      y > fiducialYHigh_[std::pair<int, int>(arm, station)] ||
      y < fiducialYLow_[std::pair<int, int>(arm, station)] ||
      x < fiducialXLow_[std::pair<int, int>(arm, station)] ||
      x > fiducialXHigh_[std::pair<int, int>(arm, station)] ||
      ((int)track.pixelTrackRecoInfo() != 0 &&
       (int)track.pixelTrackRecoInfo() != 2))
    return true;
  else
    return false;
}

// define this as a plug-in
DEFINE_FWK_MODULE(EfficiencyVsXi_2018);