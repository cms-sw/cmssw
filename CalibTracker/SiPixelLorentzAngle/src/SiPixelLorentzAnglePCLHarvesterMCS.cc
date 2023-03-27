// -*- C++ -*-
//
// Package:    CalibTracker/SiPixelLorentzAnglePCLHarvesterMCS
// Class:      SiPixelLorentzAnglePCLHarvesterMCS
//
/**\class SiPixelLorentzAnglePCLHarvesterMCS SiPixelLorentzAnglePCLHarvesterMCS.cc CalibTracker/SiPixelLorentzAngle/src/SiPixelLorentzAnglePCLHarvesterMCS.cc
 Description: reads the intermediate ALCAPROMPT DQMIO-like dataset and performs the fitting of the SiPixel Lorentz Angle in the Prompt Calibration Loop
 Implementation:
 Reads the 16 2D histograms of the cluster size x/y vs cot(alpha/beta) and 16*3 histograms of the magnetic field components created by SiPixelLorentzAnglePCLWorker module. The cluster size x/y vs cot(alpha/beta) histograms are used to generate 1D profiles (average cluster size x/y vs cot(alpha/beta)) which are then fit and the values of the cot (alpha/beta) for which the cluster sizes are minimal are determined. The obtained cot(alpha/beta)_min value for z- and z+ side are used to perform fit and the muH for different rings and panels of the Pixel Forward Phase 1 detector using the formulas: 
 cot(alpha)_min = vx/vz  = (muHBy + muH^2*Bz*Bx)/(1+muH^2*Bz^2)
 cot(beta)_min = vy/vz  = -(muHBx - muH^2*Bz*Bx)/(1+muH^2*Bz^2)
  
The extracted value of the muH are stored in an output sqlite file which is then uploaded to the conditions database.
*/
//
// Original Author:  tsusa
//         Created:  Sat, 14 Jan 2021 10:12:21 GMT
//
//

// system includes

#include <fmt/format.h>
#include <fmt/printf.h>
#include <fstream>

// user includes
#include "CalibTracker/SiPixelLorentzAngle/interface/SiPixelLorentzAngleCalibrationStruct.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondFormats/DataRecord/interface/SiPixelLorentzAngleRcd.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelLorentzAngle.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "DataFormats/TrackerCommon/interface/PixelBarrelName.h"
#include "DataFormats/TrackerCommon/interface/PixelEndcapName.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelTopologyMap.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

// for ROOT fits
#include "TFitResult.h"
#include "TVirtualFitter.h"
#include "TH1D.h"

namespace SiPixelLAHarvestMCS {

  enum fitStatus { kFitNotPerformed = -2, kNoFitResult = -1, kFitConverged = 0, kFitFailed = 1 };

  Double_t MCSFitFunction(Double_t* x, Double_t* par) {
    Double_t arg;

    if (x[0] < par[3]) {
      arg = par[1] * par[1] + par[2] * par[2] * (x[0] - par[3]) * (x[0] - par[3]);
    } else {
      arg = par[1] * par[1] + par[4] * par[4] * (x[0] - par[3]) * (x[0] - par[3]);
    }
    Double_t fitval = par[0] + sqrt(arg);
    return fitval;
  }

  struct FPixMuH {
    double bfield[3];
    double shiftx;  // Shift in x direction
    double shifty;  // Shift in y direction
    double shiftx_err;
    double shifty_err;
  };

  void fcn_func(Int_t& npar, Double_t* deriv, Double_t& f, Double_t* par, Int_t flag);

  class FitFPixMuH : public TObject {
  public:
    FitFPixMuH() : muH(0), muHErr(0) {}

    //-----------------------------------------------------------
    ~FitFPixMuH() override = default;

    friend void fcn_func(Int_t& npar, Double_t* deriv, Double_t& f, Double_t* par, Int_t flag);

    void add(const FPixMuH& fmh) { cmvar.push_back(fmh); }
    int fit(double initVal) {
      minuit = TVirtualFitter::Fitter(this, 1);
      minuit->SetFCN(fcn_func);
      double tanlorpertesla = initVal;
      minuit->SetParameter(0, "muH", tanlorpertesla, 0.1, 0., 0.);

      double arglist[100];
      arglist[0] = 3.;
      minuit->ExecuteCommand("SET PRINT", arglist, 0);
      double up = 1.0;
      minuit->SetErrorDef(up);
      arglist[0] = 100000.;
      int status = minuit->ExecuteCommand("MIGRAD", arglist, 0);
      muH = minuit->GetParameter(0);
      muHErr = minuit->GetParError(0);

      return status;
    }

    double getMuH() { return muH; }
    double getMuHErr() { return muHErr; }
    unsigned int size() { return cmvar.size(); }

  private:
    TVirtualFitter* minuit;
    std::vector<FPixMuH> cmvar;
    double muH;
    double muHErr;

    double calcChi2(double par_0) {
      double tanlorpertesla = par_0;
      double tlpt2 = tanlorpertesla * tanlorpertesla;
      double v[3], xshift, yshift;

      int n = cmvar.size();

      double chi2 = 0.0;
      for (int i = 0; i < n; i++) {
        v[0] = -(tanlorpertesla * cmvar[i].bfield[1] + tlpt2 * cmvar[i].bfield[0] * cmvar[i].bfield[2]);
        v[1] = tanlorpertesla * cmvar[i].bfield[0] - tlpt2 * cmvar[i].bfield[1] * cmvar[i].bfield[2];
        v[2] = -(1. + tlpt2 * cmvar[i].bfield[2] * cmvar[i].bfield[2]);

        xshift = v[0] / v[2];
        yshift = v[1] / v[2];

        chi2 += (xshift - cmvar[i].shiftx) * (xshift - cmvar[i].shiftx) / cmvar[i].shiftx_err / cmvar[i].shiftx_err +
                (yshift - cmvar[i].shifty) * (yshift - cmvar[i].shifty) / cmvar[i].shifty_err / cmvar[i].shifty_err;
      }
      return chi2;
    }
  };

  void fcn_func(Int_t& npar, Double_t* deriv, Double_t& f, Double_t* par, Int_t flag) {
    f = ((dynamic_cast<FitFPixMuH*>((TVirtualFitter::GetFitter())->GetObjectFit()))->calcChi2(par[0]));
  }
}  // namespace SiPixelLAHarvestMCS

//------------------------------------------------------------------------------
class SiPixelLorentzAnglePCLHarvesterMCS : public DQMEDHarvester {
public:
  SiPixelLorentzAnglePCLHarvesterMCS(const edm::ParameterSet&);
  ~SiPixelLorentzAnglePCLHarvesterMCS() override = default;
  using FPixCotAngleFitResults = std::unordered_map<uint32_t, std::pair<double, double>>;
  using FpixMuHResults = std::unordered_map<std::string, std::pair<double, double>>;
  using FitParametersInitValuesMuHFitMap = std::unordered_map<std::string, double>;

  void beginRun(const edm::Run&, const edm::EventSetup&) override;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  void dqmEndJob(DQMStore::IBooker&, DQMStore::IGetter&) override;
  void findMean(dqm::reco::MonitorElement* h_2D, dqm::reco::MonitorElement* h_mean, TH1D* h_slice);
  int getIndex(bool isBetaAngle, int r, int p, int s);

  // es tokens
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomEsToken_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> topoEsTokenBR_, topoEsTokenER_;
  const edm::ESGetToken<SiPixelLorentzAngle, SiPixelLorentzAngleRcd> siPixelLAEsToken_;
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magneticFieldToken_;

  const std::string dqmDir_;
  std::vector<std::string> newmodulelist_;
  const std::vector<double> fitRange_;
  const std::vector<double> fitParametersInitValues_;
  const std::vector<double> fitParametersInitValuesMuHFit_;
  FitParametersInitValuesMuHFitMap fitParametersInitValuesMuHFitMap_;

  const int minHitsCut_;
  const std::string recordName_;
  SiPixelLorentzAngleCalibrationHistograms hists_;
  SiPixelLAHarvestMCS::fitStatus fitMCSHistogram(dqm::reco::MonitorElement* h_mean);
  std::pair<double, double> theFitRange_{-1.5, 1.5};

  std::unique_ptr<TF1> f1_;
  FpixMuHResults fpixMuHResults;

  const SiPixelLorentzAngle* currentLorentzAngle_;
  const TrackerTopology* tTopo;
};

//------------------------------------------------------------------------------
SiPixelLorentzAnglePCLHarvesterMCS::SiPixelLorentzAnglePCLHarvesterMCS(const edm::ParameterSet& iConfig)
    : geomEsToken_(esConsumes<edm::Transition::BeginRun>()),
      topoEsTokenBR_(esConsumes<edm::Transition::BeginRun>()),
      topoEsTokenER_(esConsumes<edm::Transition::EndRun>()),
      siPixelLAEsToken_(esConsumes<edm::Transition::BeginRun>()),
      dqmDir_(iConfig.getParameter<std::string>("dqmDir")),
      newmodulelist_(iConfig.getParameter<std::vector<std::string>>("newmodulelist")),
      fitRange_(iConfig.getParameter<std::vector<double>>("fitRange")),
      fitParametersInitValues_(iConfig.getParameter<std::vector<double>>("fitParameters")),
      fitParametersInitValuesMuHFit_(iConfig.getParameter<std::vector<double>>("fitParametersMuHFit")),
      minHitsCut_(iConfig.getParameter<int>("minHitsCut")),
      recordName_(iConfig.getParameter<std::string>("record")) {
  // initialize the fit range

  if (fitRange_.size() == 2) {
    theFitRange_.first = fitRange_[0];
    theFitRange_.second = fitRange_[1];
  } else {
    throw cms::Exception("SiPixelLorentzAnglePCLHarvesterMCS") << "Wrong number of fit range parameters specified";
  }

  if (fitParametersInitValues_.size() != 5) {
    throw cms::Exception("SiPixelLorentzAnglePCLHarvesterMCS")
        << "Wrong number of initial values for fit parameters specified";
  }

  if (fitParametersInitValuesMuHFit_.size() != 4) {
    throw cms::Exception("SiPixelLorentzAnglePCLHarvesterMCS")
        << "Wrong number of initial values for fit parameters specified";
  }

  fitParametersInitValuesMuHFitMap_["R1_P1"] = fitParametersInitValuesMuHFit_[0];
  fitParametersInitValuesMuHFitMap_["R1_P2"] = fitParametersInitValuesMuHFit_[1];
  fitParametersInitValuesMuHFitMap_["R2_P1"] = fitParametersInitValuesMuHFit_[2];
  fitParametersInitValuesMuHFitMap_["R2_P2"] = fitParametersInitValuesMuHFit_[3];

  // first ensure DB output service is available
  edm::Service<cond::service::PoolDBOutputService> poolDbService;
  if (!poolDbService.isAvailable())
    throw cms::Exception("SiPixelLorentzAnglePCLHarvesterMCS") << "PoolDBService required";
}

//------------------------------------------------------------------------------
void SiPixelLorentzAnglePCLHarvesterMCS::beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {
  // geometry

  const TrackerGeometry* geom = &iSetup.getData(geomEsToken_);
  tTopo = &iSetup.getData(topoEsTokenBR_);
  currentLorentzAngle_ = &iSetup.getData(siPixelLAEsToken_);
  PixelTopologyMap map = PixelTopologyMap(geom, tTopo);
  hists_.nlay = geom->numberOfLayers(PixelSubdetector::PixelBarrel);
  hists_.nModules_.resize(hists_.nlay);
  hists_.nLadders_.resize(hists_.nlay);
  for (int i = 0; i < hists_.nlay; i++) {
    hists_.nModules_[i] = map.getPXBModules(i + 1);
    hists_.nLadders_[i] = map.getPXBLadders(i + 1);
  }

  // list of modules already filled, then return (we already entered here)
  if (!hists_.BPixnewDetIds_.empty() || !hists_.FPixnewDetIds_.empty())
    return;

  if (!newmodulelist_.empty()) {
    for (auto const& modulename : newmodulelist_) {
      if (modulename.find("BPix_") != std::string::npos) {
        PixelBarrelName bn(modulename, true);
        const auto& detId = bn.getDetId(tTopo);
        hists_.BPixnewmodulename_.push_back(modulename);
        hists_.BPixnewDetIds_.push_back(detId.rawId());
        hists_.BPixnewModule_.push_back(bn.moduleName());
        hists_.BPixnewLayer_.push_back(bn.layerName());
      } else if (modulename.find("FPix_") != std::string::npos) {
        PixelEndcapName en(modulename, true);
        const auto& detId = en.getDetId(tTopo);
        hists_.FPixnewmodulename_.push_back(modulename);
        hists_.FPixnewDetIds_.push_back(detId.rawId());
        hists_.FPixnewDisk_.push_back(en.diskName());
        hists_.FPixnewBlade_.push_back(en.bladeName());
      }
    }
  }

  uint count = 0;
  for (const auto& id : hists_.BPixnewDetIds_) {
    LogDebug("SiPixelLorentzAnglePCLHarvesterMCS") << id;
    count++;
  }
  LogDebug("SiPixelLorentzAnglePCLHarvesterMCS") << "Stored a total of " << count << " new detIds.";

  // list of modules already filled, return (we already entered here)
  if (!hists_.detIdsList.empty())
    return;

  std::vector<uint32_t> treatedIndices;

  for (const auto& det : geom->detsPXB()) {
    const PixelGeomDetUnit* pixelDet = dynamic_cast<const PixelGeomDetUnit*>(det);
    const auto& layer = tTopo->pxbLayer(pixelDet->geographicalId());
    const auto& module = tTopo->pxbModule(pixelDet->geographicalId());
    int i_index = module + (layer - 1) * hists_.nModules_[layer - 1];

    uint32_t rawId = pixelDet->geographicalId().rawId();

    // if the detId is already accounted for in the special class, do not attach it
    if (std::find(hists_.BPixnewDetIds_.begin(), hists_.BPixnewDetIds_.end(), rawId) != hists_.BPixnewDetIds_.end())
      continue;
    if (std::find(treatedIndices.begin(), treatedIndices.end(), i_index) != treatedIndices.end()) {
      hists_.detIdsList[i_index].push_back(rawId);
    } else {
      hists_.detIdsList.insert(std::pair<uint32_t, std::vector<uint32_t>>(i_index, {rawId}));
      treatedIndices.push_back(i_index);
    }
  }

  count = 0;
  for (const auto& i : treatedIndices) {
    for (const auto& id : hists_.detIdsList[i]) {
      LogDebug("SiPixelLorentzAnglePCLHarvesterMCS") << id;
      count++;
    };
  }
  LogDebug("SiPixelLorentzAnglePCLHarvesterMCS") << "Stored a total of " << count << " detIds.";
}
//------------------------------------------------------------------------------
void SiPixelLorentzAnglePCLHarvesterMCS::dqmEndJob(DQMStore::IBooker& iBooker, DQMStore::IGetter& iGetter) {
  iGetter.cd();
  iGetter.setCurrentFolder(dqmDir_);

  // get mean and size-angle hists, book summary hists
  std::string hname;
  int nBins = hists_.nAngles_ * hists_.nRings_ * hists_.nPanels_ * hists_.nSides_;
  hists_.h_fpixMeanHistoFitStatus_ =
      iBooker.book1D("fpixMeanHistoFitStatus",
                     "fit status by angles/rings/panels/sides; angle/ring/panel/side; fit status",
                     nBins,
                     -0.5,
                     nBins - 0.5);
  hists_.h_fpixMinClusterSizeCotAngle_ =
      iBooker.book1D("fpixMinClusterSizeCotAngle",
                     "cot angle of minimal cluster size by angles/rings/panels/sides; angle/ring/panel/side; ",
                     nBins,
                     -0.5,
                     nBins - 0.5);
  hists_.h_fpixNhitsClusterSizeCotAngle_ =
      iBooker.book1D("fpixNhitsClusterSizeCotAngle",
                     "number of hits by angles/rings/panels/sides; angle/ring/panel/side; ",
                     nBins,
                     -0.5,
                     nBins - 0.5);

  std::string binNameAlpha;
  std::string binNameBeta;

  const auto& prefix_ = fmt::sprintf("%s/FPix", dqmDir_);
  int histsCounter = 0;
  int nHistoExpected = 0;
  for (int r = 0; r < hists_.nRings_; ++r) {
    for (int p = 0; p < hists_.nPanels_; ++p) {
      for (int s = 0; s < hists_.nSides_; ++s) {
        int idx = getIndex(false, r, p, s);
        int idxBeta = getIndex(true, r, p, s);
        nHistoExpected++;
        hname = fmt::format("{}/R{}_P{}_z{}_alphaMean", dqmDir_, r + 1, p + 1, s + 1);
        if ((hists_.h_fpixMean_[idx] = iGetter.get(hname)) == nullptr) {
          edm::LogError("SiPixelLorentzAnglePCLHarvesterMCS::dqmEndJob")
              << "Failed to retrieve " << hname << " histogram";
        } else
          histsCounter++;

        nHistoExpected++;
        hname = fmt::format("{}/R{}_P{}_z{}_betaMean", dqmDir_, r + 1, p + 1, s + 1);
        if ((hists_.h_fpixMean_[idxBeta] = iGetter.get(hname)) == nullptr) {
          edm::LogError("SiPixelLorentzAnglePCLHarvesterMCS::dqmEndJob")
              << "Failed to retrieve " << hname << " histogram";
        } else
          histsCounter++;

        nHistoExpected++;
        hname = fmt::format("{}/R{}_P{}_z{}_alpha", prefix_, r + 1, p + 1, s + 1);
        if ((hists_.h_fpixAngleSize_[idx] = iGetter.get(hname)) == nullptr) {
          edm::LogError("SiPixelLorentzAnglePCLHarvesterMCS::dqmEndJob")
              << "Failed to retrieve " << hname << " histogram";
        } else
          histsCounter++;

        nHistoExpected++;
        hname = fmt::format("{}/R{}_P{}_z{}_beta", prefix_, r + 1, p + 1, s + 1);
        if ((hists_.h_fpixAngleSize_[idxBeta] = iGetter.get(hname)) == nullptr) {
          edm::LogError("SiPixelLorentzAnglePCLHarvesterMCS::dqmEndJob")
              << "Failed to retrieve " << hname << " histogram";
        } else
          histsCounter++;

        for (int m = 0; m < 3; ++m) {
          nHistoExpected++;
          hname = fmt::format("{}/R{}_P{}_z{}_B{}", prefix_, r + 1, p + 1, s + 1, m);
          if ((hists_.h_fpixMagField_[m][idx] = iGetter.get(hname)) == nullptr) {
            edm::LogError("SiPixelLorentzAnglePCLHarvesterMCS::dqmEndJob")
                << "Failed to retrieve " << hname << " histogram";
          } else
            histsCounter++;
        }

        // set labels & init summary histos
        int binAlpha = idx + 1;
        int binBeta = idxBeta + 1;
        char sign = s == 0 ? '-' : '+';
        binNameAlpha = fmt::sprintf("#alpha: R%d_P%d_z%c", r + 1, p + 1, sign);
        binNameBeta = fmt::sprintf("#beta:R%d_P%d_z%c", r + 1, p + 1, sign);
        hists_.h_fpixMeanHistoFitStatus_->setBinLabel(binAlpha, binNameAlpha);
        hists_.h_fpixMeanHistoFitStatus_->setBinLabel(binBeta, binNameBeta);
        hists_.h_fpixMeanHistoFitStatus_->setBinContent(binAlpha, SiPixelLAHarvestMCS::kFitNotPerformed);
        hists_.h_fpixMeanHistoFitStatus_->setBinContent(binBeta, SiPixelLAHarvestMCS::kFitNotPerformed);

        hists_.h_fpixMinClusterSizeCotAngle_->setBinLabel(binAlpha, binNameAlpha);
        hists_.h_fpixMinClusterSizeCotAngle_->setBinLabel(binBeta, binNameBeta);
        hists_.h_fpixNhitsClusterSizeCotAngle_->setBinLabel(binAlpha, binNameAlpha);
        hists_.h_fpixNhitsClusterSizeCotAngle_->setBinLabel(binBeta, binNameBeta);
      }
    }
  }

  if (histsCounter != nHistoExpected) {
    edm::LogError("SiPixelLorentzAnglePCLHarvesterMCS::dqmEndJob")
        << "Failed to retrieve all histograms, expected 56 got " << histsCounter;
    return;
  }

  // book hervesting hists
  iBooker.setCurrentFolder(fmt::format("{}Harvesting", dqmDir_));
  int nBinsMuH = hists_.nRings_ * hists_.nPanels_;
  hists_.h_fpixFitStatusMuH_ = iBooker.book1D(
      "fpixFitStatusMuH", "muH fit status by rings/panels; ring/panel; fitStatus", nBinsMuH, -0.5, nBinsMuH - 0.5);
  hists_.h_fpixMuH_ =
      iBooker.book1D("fpixMuH", "muH by rings/panels; ring/panel; #muH [1/T]", nBinsMuH, -0.5, nBinsMuH - 0.5);
  hists_.h_fpixDeltaMuH_ = iBooker.book1D(
      "fpixDeltaMuH", "#Delta muH by rings/panels; ring/panel; #Delta #muH [1/T]", nBinsMuH, -0.5, nBinsMuH - 0.5);
  hists_.h_fpixRelDeltaMuH_ = iBooker.book1D("fpixRelDeltaMuH",
                                             "#Delta #muH/#muH by rings/panels; ring/panel; #Delta #muH/#MuH",
                                             nBinsMuH,
                                             -0.5,
                                             nBinsMuH - 0.5);
  std::string binName;
  for (int r = 0; r < hists_.nRings_; ++r) {
    for (int p = 0; p < hists_.nPanels_; ++p) {
      int idx = r * hists_.nPanels_ + p + 1;
      binName = fmt::sprintf("R%d_P%d", r + 1, p + 1);
      hists_.h_fpixFitStatusMuH_->setBinLabel(idx, binName);
      hists_.h_fpixFitStatusMuH_->setBinContent(idx, SiPixelLAHarvestMCS::kFitNotPerformed);
      hists_.h_fpixMuH_->setBinLabel(idx, binName);
      hists_.h_fpixDeltaMuH_->setBinLabel(idx, binName);
      hists_.h_fpixRelDeltaMuH_->setBinLabel(idx, binName);
    }
  }

  // make and fit profile hists, fit muH
  int nSizeBins = hists_.h_fpixAngleSize_[0]->getNbinsY();
  double minSize = hists_.h_fpixAngleSize_[0]->getAxisMin(2);
  double maxSize = hists_.h_fpixAngleSize_[0]->getAxisMax(2);
  TH1D* h_slice = new TH1D("h_slice", "slice of cot_angle histogram", nSizeBins, minSize, maxSize);
  f1_ = std::make_unique<TF1>("f1", SiPixelLAHarvestMCS::MCSFitFunction, -3., 3., 5);
  f1_->SetParNames("Offset", "RMS Constant", "SlopeL", "cot(angle)_min", "SlopeR");

  for (int r = 0; r < hists_.nRings_; ++r) {
    for (int p = 0; p < hists_.nPanels_; ++p) {
      SiPixelLAHarvestMCS::FitFPixMuH fitMuH;
      SiPixelLAHarvestMCS::FPixMuH fmh;
      for (int s = 0; s < hists_.nSides_; ++s) {
        int idx = getIndex(false, r, p, s);
        int idxBeta = getIndex(true, r, p, s);
        int binAlpha = idx + 1;
        int binBeta = idxBeta + 1;

        int entriesAlpha = hists_.h_fpixAngleSize_[idx]->getEntries();
        int entriesBeta = hists_.h_fpixAngleSize_[idxBeta]->getEntries();
        hists_.h_fpixNhitsClusterSizeCotAngle_->setBinContent(binAlpha, entriesAlpha);
        hists_.h_fpixNhitsClusterSizeCotAngle_->setBinContent(binBeta, entriesBeta);
        findMean(hists_.h_fpixAngleSize_[idx], hists_.h_fpixMean_[idx], h_slice);
        findMean(hists_.h_fpixAngleSize_[idxBeta], hists_.h_fpixMean_[idxBeta], h_slice);

        SiPixelLAHarvestMCS::fitStatus statusAlphaFit = entriesAlpha < minHitsCut_
                                                            ? SiPixelLAHarvestMCS::kFitNotPerformed
                                                            : fitMCSHistogram(hists_.h_fpixMean_[idx]);
        SiPixelLAHarvestMCS::fitStatus statusBetaFit = entriesBeta < minHitsCut_
                                                           ? SiPixelLAHarvestMCS::kFitNotPerformed
                                                           : fitMCSHistogram(hists_.h_fpixMean_[idxBeta]);

        hists_.h_fpixMeanHistoFitStatus_->setBinContent(binAlpha, statusAlphaFit);
        hists_.h_fpixMeanHistoFitStatus_->setBinContent(binBeta, statusBetaFit);

        if (entriesAlpha < minHitsCut_ || entriesBeta < minHitsCut_)
          continue;

        assert(strcmp(f1_->GetName(), "f1") == 0);

        if (statusAlphaFit == SiPixelLAHarvestMCS::kFitConverged &&
            statusBetaFit == SiPixelLAHarvestMCS::kFitConverged) {
          double shiftX = hists_.h_fpixMean_[idx]->getTH1()->GetFunction("f1")->GetParameter(3);
          double errShiftX = hists_.h_fpixMean_[idx]->getTH1()->GetFunction("f1")->GetParError(3);
          double shiftY = hists_.h_fpixMean_[idxBeta]->getTH1()->GetFunction("f1")->GetParameter(3);
          double errShiftY = hists_.h_fpixMean_[idxBeta]->getTH1()->GetFunction("f1")->GetParError(3);

          hists_.h_fpixMinClusterSizeCotAngle_->setBinContent(binAlpha, shiftX);
          hists_.h_fpixMinClusterSizeCotAngle_->setBinError(binAlpha, errShiftX);
          hists_.h_fpixMinClusterSizeCotAngle_->setBinContent(binBeta, shiftY);
          hists_.h_fpixMinClusterSizeCotAngle_->setBinError(binBeta, errShiftY);

          fmh.shiftx = shiftX;
          fmh.shiftx_err = errShiftX;
          fmh.shifty = shiftY;
          fmh.shifty_err = errShiftY;
          fmh.bfield[0] = hists_.h_fpixMagField_[0][idx]->getMean();
          fmh.bfield[1] = hists_.h_fpixMagField_[1][idx]->getMean();
          fmh.bfield[2] = hists_.h_fpixMagField_[2][idx]->getMean();
          fitMuH.add(fmh);
        }  // if fut converged
      }    // loop over z sides

      if (fitMuH.size() == hists_.nSides_) {
        std::string fpixPartNames = "R" + std::to_string(r + 1) + "_P" + std::to_string(p + 1);
        double initMuH = fitParametersInitValuesMuHFitMap_[fpixPartNames];
        int status = fitMuH.fit(initMuH);
        int idxMuH = r * hists_.nPanels_ + p + 1;
        double muH = fitMuH.getMuH();
        double muHErr = fitMuH.getMuHErr();
        hists_.h_fpixFitStatusMuH_->setBinContent(idxMuH, status);
        hists_.h_fpixMuH_->setBinContent(idxMuH, muH);
        hists_.h_fpixMuH_->setBinError(idxMuH, muHErr);
        fpixMuHResults.insert(std::make_pair(fpixPartNames, std::make_pair(muH, muH)));
      }
    }
  }

  std::shared_ptr<SiPixelLorentzAngle> LorentzAngle = std::make_shared<SiPixelLorentzAngle>();

  bool isPayloadChanged{false};

  for (const auto& [id, value] : currentLorentzAngle_->getLorentzAngles()) {
    DetId ID = DetId(id);
    float muHForDB = value;
    if (ID.subdetId() == PixelSubdetector::PixelEndcap) {
      PixelEndcapName pen(ID, tTopo, true);  // use det-id phaseq
      int panel = pen.pannelName();
      int ring = pen.ringName();
      std::string fpixPartNames = "R" + std::to_string(ring) + "_P" + std::to_string(panel);
      if (fpixMuHResults.find(fpixPartNames) != fpixMuHResults.end()) {
        double measuredMuH = fpixMuHResults[fpixPartNames].first;
        double deltaMuH = value - measuredMuH;
        double deltaMuHoverMuH = deltaMuH / value;
        int idxMuH = (ring - 1) * hists_.nPanels_ + panel;
        hists_.h_fpixDeltaMuH_->setBinContent(idxMuH, deltaMuH);
        hists_.h_fpixRelDeltaMuH_->setBinContent(idxMuH, deltaMuHoverMuH);
        muHForDB = measuredMuH;
        if (!isPayloadChanged && (deltaMuHoverMuH != 0.f))
          isPayloadChanged = true;
      }
    }  // if endcap
    if (!LorentzAngle->putLorentzAngle(id, muHForDB)) {
      edm::LogError("SiPixelLorentzAnglePCLHarvesterMCS")
          << "[SiPixelLorentzAnglePCLHarvesterMCS::dqmEndJob]: detid (" << id << ") already exists";
    }
  }

  if (isPayloadChanged) {
    // fill the DB object record
    edm::Service<cond::service::PoolDBOutputService> mydbservice;
    if (mydbservice.isAvailable()) {
      try {
        mydbservice->writeOneIOV(*LorentzAngle, mydbservice->currentTime(), recordName_);
      } catch (const cond::Exception& er) {
        edm::LogError("SiPixelLorentzAngleDB") << er.what();
      } catch (const std::exception& er) {
        edm::LogError("SiPixelLorentzAngleDB") << "caught std::exception " << er.what();
      }
    } else {
      edm::LogError("SiPixelLorentzAngleDB") << "Service is unavailable";
    }
  } else {
    edm::LogPrint("SiPixelLorentzAngleDB") << __PRETTY_FUNCTION__ << " there is no new valid measurement to append! ";
  }
}

//------------------------------------------------------------------------------
void SiPixelLorentzAnglePCLHarvesterMCS::findMean(dqm::reco::MonitorElement* h_2D,
                                                  dqm::reco::MonitorElement* h_mean,
                                                  TH1D* h_slice) {
  int n_x = h_2D->getNbinsX();
  int n_y = h_2D->getNbinsY();

  for (int i = 1; i <= n_x; i++) {
    h_slice->Reset("ICE");

    //loop over bins in size

    for (int j = 1; j <= n_y; j++) {
      h_slice->SetBinContent(j, h_2D->getBinContent(i, j));
    }
    double mean = h_slice->GetMean();
    double error = h_slice->GetMeanError();
    h_mean->setBinContent(i, mean);
    h_mean->setBinError(i, error);
  }  // end loop over bins in depth
}

//------------------------------------------------------------------------------
SiPixelLAHarvestMCS::fitStatus SiPixelLorentzAnglePCLHarvesterMCS::fitMCSHistogram(dqm::reco::MonitorElement* h_mean) {
  SiPixelLAHarvestMCS::fitStatus retVal;

  f1_->SetParameters(fitParametersInitValues_[0],
                     fitParametersInitValues_[1],
                     fitParametersInitValues_[2],
                     fitParametersInitValues_[3],
                     fitParametersInitValues_[4]);

  int nFits = 0;
  while (nFits < 5) {
    nFits++;
    double fitMin = f1_->GetParameter(3) + theFitRange_.first;
    double fitMax = f1_->GetParameter(3) + theFitRange_.second;

    if (fitMin < -3)
      fitMin = -3;
    if (fitMax > 3)
      fitMax = 3;

    TFitResultPtr r = h_mean->getTH1()->Fit(f1_.get(), "ERSM", "", fitMin, fitMax);
    retVal = r == -1        ? SiPixelLAHarvestMCS::kNoFitResult
             : r->IsValid() ? SiPixelLAHarvestMCS::kFitConverged
                            : SiPixelLAHarvestMCS::kFitFailed;
  }
  return retVal;
}

//------------------------------------------------------------------------------
void SiPixelLorentzAnglePCLHarvesterMCS::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setComment("Harvester module of the SiPixel Lorentz Angle PCL monitoring workflow for MinimalClusterSizeMethod");
  desc.add<std::string>("dqmDir", "AlCaReco/SiPixelLorentzAngle")->setComment("the directory of PCL Worker output");
  desc.add<std::vector<std::string>>("newmodulelist", {})->setComment("the list of DetIds for new sensors");
  desc.add<std::vector<double>>("fitRange", {-1.5, 1.5})->setComment("range  to perform the fit");
  desc.add<std::vector<double>>("fitParameters", {1., 0.1, -1.6, 0., 1.6})
      ->setComment("initial values for fit parameters");
  desc.add<std::vector<double>>("fitParametersMuHFit", {0.08, 0.08, 0.08, 0.08})
      ->setComment("initial values for fit parameters (muH fit)");
  desc.add<int>("minHitsCut", 1000)->setComment("cut on minimum number of on-track hits to accept measurement");
  desc.add<std::string>("record", "SiPixelLorentzAngleRcdMCS")->setComment("target DB record");
  descriptions.addWithDefaultLabel(desc);
}

int SiPixelLorentzAnglePCLHarvesterMCS::getIndex(bool isBetaAngle, int r, int p, int s) {
  int idx = hists_.nSides_ * hists_.nPanels_ * r + hists_.nSides_ * p + s;
  return (isBetaAngle ? idx + hists_.betaStartIdx_ : idx);
}
DEFINE_FWK_MODULE(SiPixelLorentzAnglePCLHarvesterMCS);
