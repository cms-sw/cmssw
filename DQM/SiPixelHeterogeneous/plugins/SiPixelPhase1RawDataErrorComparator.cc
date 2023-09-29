// -*- C++ -*-
// Package:    SiPixelPhase1RawDataErrorComparator
// Class:      SiPixelPhase1RawDataErrorComparator
//
/**\class SiPixelPhase1RawDataErrorComparator SiPixelPhase1RawDataErrorComparator.cc
*/
//
// Author: Marco Musich
//
#include "CondFormats/DataRecord/interface/SiPixelFedCablingMapRcd.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingTree.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/SiPixelDetId/interface/PixelFEDChannel.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "EventFilter/SiPixelRawToDigi/interface/PixelDataFormatter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
// for string manipulations
#include <fmt/printf.h>

namespace {
  // same logic used for the MTV:
  // cf https://github.com/cms-sw/cmssw/blob/master/Validation/RecoTrack/src/MTVHistoProducerAlgoForTracker.cc
  typedef dqm::reco::DQMStore DQMStore;

  void setBinLog(TAxis* axis) {
    int bins = axis->GetNbins();
    float from = axis->GetXmin();
    float to = axis->GetXmax();
    float width = (to - from) / bins;
    std::vector<float> new_bins(bins + 1, 0);
    for (int i = 0; i <= bins; i++) {
      new_bins[i] = TMath::Power(10, from + i * width);
    }
    axis->Set(bins, new_bins.data());
  }

  void setBinLogX(TH1* h) {
    TAxis* axis = h->GetXaxis();
    setBinLog(axis);
  }
  void setBinLogY(TH1* h) {
    TAxis* axis = h->GetYaxis();
    setBinLog(axis);
  }

  template <typename... Args>
  dqm::reco::MonitorElement* make2DIfLog(DQMStore::IBooker& ibook, bool logx, bool logy, Args&&... args) {
    auto h = std::make_unique<TH2I>(std::forward<Args>(args)...);
    if (logx)
      setBinLogX(h.get());
    if (logy)
      setBinLogY(h.get());
    const auto& name = h->GetName();
    return ibook.book2I(name, h.release());
  }

  //errorType - a number (25-38) indicating the type of error recorded.
  enum SiPixelFEDErrorCodes {
    k_FED25 = 25,  // 25 indicates an invalid ROC of 25
    k_FED26 = 26,  // 26 indicates a gap word
    k_FED27 = 27,  // 27 indicates a dummy word
    k_FED28 = 28,  // 28 indicates a FIFO full error
    k_FED29 = 29,  // 29 indicates a timeout error
    k_FED30 = 30,  // 30 indicates a TBM error trailer
    k_FED31 = 31,  // 31 indicates an event number error (TBM and FED event number mismatch)
    k_FED32 = 32,  // 32 indicates an incorrectly formatted Slink Header
    k_FED33 = 33,  // 33 indicates an incorrectly formatted Slink Trailer
    k_FED34 = 34,  // 34 indicates the evt size encoded in Slink Trailer is different than size found at raw2digi
    k_FED35 = 35,  // 35 indicates an invalid FED channel number
    k_FED36 = 36,  // 36 indicates an invalid ROC value
    k_FED37 = 37,  // 37 indicates an invalid dcol or pixel value
    k_FED38 = 38   // 38 indicates the pixels on a ROC weren't read out from lowest to highest row and dcol value
  };

  using MapToCodes = std::map<SiPixelFEDErrorCodes, std::string>;

  const MapToCodes errorCodeToStringMap = {{k_FED25, "FED25 error"},
                                           {k_FED26, "FED26 error"},
                                           {k_FED27, "FED27 error"},
                                           {k_FED28, "FED28 error"},
                                           {k_FED29, "FED29 error"},
                                           {k_FED30, "FED30 error"},
                                           {k_FED31, "FED31 error"}};

  const MapToCodes errorCodeToTypeMap = {{k_FED25, "ROC of 25"},
                                         {k_FED26, "Gap word"},
                                         {k_FED27, "Dummy word"},
                                         {k_FED28, "FIFO full"},
                                         {k_FED29, "Timeout"},
                                         {k_FED30, "TBM error trailer"},
                                         {k_FED31, "Event number"},
                                         {k_FED32, "Slink header"},
                                         {k_FED33, "Slink trailer"},
                                         {k_FED34, "Event size"},
                                         {k_FED35, "Invalid channel#"},
                                         {k_FED36, "ROC value"},
                                         {k_FED37, "Dcol or pixel value"},
                                         {k_FED38, "Readout order"}};
}  // namespace

class SiPixelPhase1RawDataErrorComparator : public DQMEDAnalyzer {
public:
  explicit SiPixelPhase1RawDataErrorComparator(const edm::ParameterSet&);
  ~SiPixelPhase1RawDataErrorComparator() override = default;
  void bookHistograms(DQMStore::IBooker& ibooker, edm::Run const& iRun, edm::EventSetup const& iSetup) override;
  void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  const edm::InputTag pixelErrorSrcGPU_;
  const edm::InputTag pixelErrorSrcCPU_;
  const edm::EDGetTokenT<edm::DetSetVector<SiPixelRawDataError>> tokenErrorsGPU_;
  const edm::EDGetTokenT<edm::DetSetVector<SiPixelRawDataError>> tokenErrorsCPU_;
  const std::string topFolderName_;

  MonitorElement* h_totFEDErrors_;
  MonitorElement* h_FEDerrorVsFEDIdUnbalance_;
  std::unordered_map<SiPixelFEDErrorCodes, MonitorElement*> h_nFEDErrors_;

  // name of the plugins
  static constexpr const char* kName = "SiPixelPhase1RawDataErrorComparator";

  // Define the dimensions of the 2D array
  static constexpr int nFEDs = FEDNumbering::MAXSiPixeluTCAFEDID - FEDNumbering::MINSiPixeluTCAFEDID;
  static constexpr int nErrors = k_FED38 - k_FED25;
};

//
// constructors
//
SiPixelPhase1RawDataErrorComparator::SiPixelPhase1RawDataErrorComparator(const edm::ParameterSet& iConfig)
    : pixelErrorSrcGPU_(iConfig.getParameter<edm::InputTag>("pixelErrorSrcGPU")),
      pixelErrorSrcCPU_(iConfig.getParameter<edm::InputTag>("pixelErrorSrcCPU")),
      tokenErrorsGPU_(consumes<edm::DetSetVector<SiPixelRawDataError>>(pixelErrorSrcGPU_)),
      tokenErrorsCPU_(consumes<edm::DetSetVector<SiPixelRawDataError>>(pixelErrorSrcCPU_)),
      topFolderName_(iConfig.getParameter<std::string>("topFolderName")) {}

//
// -- Analyze
//
void SiPixelPhase1RawDataErrorComparator::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  std::map<int, int> countsOnCPU;
  std::map<int, int> countsOnGPU;

  std::array<std::array<int, nErrors>, nFEDs> countsMatrixOnCPU;
  std::array<std::array<int, nErrors>, nFEDs> countsMatrixOnGPU;

  // initialize the counts for FED/error matrix
  for (int i = 0; i < nFEDs; i++) {
    for (int j = 0; j < nErrors; j++) {
      countsMatrixOnCPU[i][j] = 0;
      countsMatrixOnGPU[i][j] = 0;
    }
  }

  // initialize the counts for errors per type scatter plots
  for (unsigned int j = k_FED25; j <= k_FED31; j++) {
    countsOnCPU[j] = 0.;
    countsOnGPU[j] = 0.;
  }

  // check upfront if the error collection is present
  edm::Handle<edm::DetSetVector<SiPixelRawDataError>> inputFromCPU;
  iEvent.getByToken(tokenErrorsCPU_, inputFromCPU);
  if (!inputFromCPU.isValid()) {
    edm::LogError(kName) << "reference (cpu) SiPixelRawDataErrors collection (" << pixelErrorSrcCPU_.encode()
                         << ") not found; \n"
                         << "the comparison will not run.";
    return;
  }

  edm::Handle<edm::DetSetVector<SiPixelRawDataError>> inputFromGPU;
  iEvent.getByToken(tokenErrorsGPU_, inputFromGPU);
  if (!inputFromGPU.isValid()) {
    edm::LogError(kName) << "target (gpu) SiPixelRawDataErrors collection (" << pixelErrorSrcGPU_.encode()
                         << ") not found; \n"
                         << "the comparison will not run.";
    return;
  }

  // fill the counters on host
  uint errorsOnCPU{0};
  for (auto it = inputFromCPU->begin(); it != inputFromCPU->end(); ++it) {
    for (auto& siPixelRawDataError : *it) {
      int fed = siPixelRawDataError.getFedId();
      int type = siPixelRawDataError.getType();
      DetId id = it->detId();

      // fill the error matrices for CPU
      countsOnCPU[type] += 1;
      countsMatrixOnCPU[fed - FEDNumbering::MINSiPixeluTCAFEDID][type - k_FED25] += 1;

      LogDebug(kName) << " (on cpu) FED: " << fed << " detid: " << id.rawId() << " type:" << type;
      errorsOnCPU++;
    }
  }

  // fill the counters on device
  uint errorsOnGPU{0};
  for (auto it = inputFromGPU->begin(); it != inputFromGPU->end(); ++it) {
    for (auto& siPixelRawDataError : *it) {
      int fed = siPixelRawDataError.getFedId();
      int type = siPixelRawDataError.getType();
      DetId id = it->detId();

      // fill the error matrices for GPU
      countsOnGPU[type] += 1;
      countsMatrixOnGPU[fed - FEDNumbering::MINSiPixeluTCAFEDID][type - k_FED25] += 1;

      LogDebug(kName) << " (on gpu) FED: " << fed << " detid: " << id.rawId() << " type:" << type;
      errorsOnGPU++;
    }
  }

  edm::LogPrint(kName) << " on gpu found: " << errorsOnGPU << " on cpu found: " << errorsOnCPU;

  h_totFEDErrors_->Fill(errorsOnCPU, errorsOnGPU);

  // fill the correlations per error type
  for (unsigned int j = k_FED25; j <= k_FED31; j++) {
    const SiPixelFEDErrorCodes code = static_cast<SiPixelFEDErrorCodes>(j);
    h_nFEDErrors_[code]->Fill(std::min(1000, countsOnCPU[j]), std::min(1000, countsOnGPU[j]));
  }

  // fill the error unbalance per FEDid per error type
  for (int i = 0; i < nFEDs; i++) {
    for (int j = 0; j < nErrors; j++) {
      if (countsMatrixOnGPU[i][j] != 0 || countsMatrixOnCPU[i][j] != 0) {
        edm::LogVerbatim(kName) << "FED: " << i + FEDNumbering::MINSiPixeluTCAFEDID << " error: " << j + k_FED25
                                << " | GPU counts: " << countsMatrixOnGPU[i][j]
                                << " CPU counts:" << countsMatrixOnCPU[i][j];
        h_FEDerrorVsFEDIdUnbalance_->Fill(
            j, i + FEDNumbering::MINSiPixeluTCAFEDID, countsMatrixOnGPU[i][j] - countsMatrixOnCPU[i][j]);
      }
    }
  }
}

//
// -- Book Histograms
//
void SiPixelPhase1RawDataErrorComparator::bookHistograms(DQMStore::IBooker& iBook,
                                                         edm::Run const& iRun,
                                                         edm::EventSetup const& iSetup) {
  iBook.cd();
  iBook.setCurrentFolder(topFolderName_);

  h_FEDerrorVsFEDIdUnbalance_ =
      iBook.book2I("FEErrorVsFEDIdUnbalance",
                   "difference (GPU-CPU) of FED errors per FEDid per error type;;FED Id number;GPU counts - CPU counts",
                   nErrors,
                   -0.5,
                   nErrors - 0.5,
                   nFEDs,
                   FEDNumbering::MINSiPixeluTCAFEDID - 0.5,
                   FEDNumbering::MAXSiPixeluTCAFEDID - 0.5);
  for (int j = 0; j < nErrors; j++) {
    const auto& errorCode = static_cast<SiPixelFEDErrorCodes>(j + k_FED25);
    h_FEDerrorVsFEDIdUnbalance_->setBinLabel(j + 1, errorCodeToTypeMap.at(errorCode));
  }

  h_totFEDErrors_ = make2DIfLog(iBook,
                                true,
                                true,
                                "nTotalFEDError",
                                "n. of total Pixel FEDError per event; CPU; GPU",
                                500,
                                log10(0.5),
                                log10(5000.5),
                                500,
                                log10(0.5),
                                log10(5000.5));

  for (const auto& element : errorCodeToStringMap) {
    h_nFEDErrors_[element.first] = iBook.book2I(fmt::sprintf("nFED%i_Errors", element.first),
                                                fmt::sprintf("n. of %ss per event; CPU; GPU", element.second),
                                                1000,
                                                -0.5,
                                                1000.5,
                                                1000,
                                                -0.5,
                                                1000.5);
  }
}

void SiPixelPhase1RawDataErrorComparator::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("pixelErrorSrcGPU", edm::InputTag("siPixelDigis@cuda"))
      ->setComment("input GPU SiPixel FED errors");
  desc.add<edm::InputTag>("pixelErrorSrcCPU", edm::InputTag("siPixelDigis@cpu"))
      ->setComment("input CPU SiPixel FED errors");
  desc.add<std::string>("topFolderName", "SiPixelHeterogeneous/PixelErrorCompareGPUvsCPU");
  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(SiPixelPhase1RawDataErrorComparator);
