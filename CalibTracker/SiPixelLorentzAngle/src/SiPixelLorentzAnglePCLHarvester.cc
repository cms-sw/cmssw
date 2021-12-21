#include <fmt/format.h>
#include <fmt/printf.h>
#include <fstream>

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

//------------------------------------------------------------------------------
class SiPixelLorentzAnglePCLHarvester : public DQMEDHarvester {
public:
  SiPixelLorentzAnglePCLHarvester(const edm::ParameterSet&);
  ~SiPixelLorentzAnglePCLHarvester() override = default;
  void beginRun(const edm::Run&, const edm::EventSetup&) override;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  void dqmEndJob(DQMStore::IBooker&, DQMStore::IGetter&) override;
  void findMean(MonitorElement* h_drift_depth_adc_slice_, int i, int i_ring);

  // es tokens
  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomEsToken_;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> topoEsToken_;
  edm::ESGetToken<SiPixelLorentzAngle, SiPixelLorentzAngleRcd> siPixelLAEsToken_;
  edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magneticFieldToken_;

  std::vector<std::string> newmodulelist_;
  const std::string dqmDir_;
  const double fitProbCut_;
  const std::string recordName_;
  std::unique_ptr<TF1> f1;
  float width_;

  SiPixelLorentzAngleCalibrationHistograms hists;
  const SiPixelLorentzAngle* currentLorentzAngle;
  const MagneticField* magField;
};

//------------------------------------------------------------------------------
SiPixelLorentzAnglePCLHarvester::SiPixelLorentzAnglePCLHarvester(const edm::ParameterSet& iConfig)
    : geomEsToken_(esConsumes<edm::Transition::BeginRun>()),
      topoEsToken_(esConsumes<edm::Transition::BeginRun>()),
      siPixelLAEsToken_(esConsumes<edm::Transition::BeginRun>()),
      magneticFieldToken_(esConsumes<edm::Transition::BeginRun>()),
      newmodulelist_(iConfig.getParameter<std::vector<std::string>>("newmodulelist")),
      dqmDir_(iConfig.getParameter<std::string>("dqmDir")),
      fitProbCut_(iConfig.getParameter<double>("fitProbCut")),
      recordName_(iConfig.getParameter<std::string>("record")) {
  // first ensure DB output service is available
  edm::Service<cond::service::PoolDBOutputService> poolDbService;
  if (!poolDbService.isAvailable())
    throw cms::Exception("SiPixelLorentzAnglePCLHarvester") << "PoolDBService required";
}

//------------------------------------------------------------------------------
void SiPixelLorentzAnglePCLHarvester::beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {
  // geometry
  const TrackerGeometry* geom = &iSetup.getData(geomEsToken_);
  const TrackerTopology* tTopo = &iSetup.getData(topoEsToken_);

  magField = &iSetup.getData(magneticFieldToken_);
  currentLorentzAngle = &iSetup.getData(siPixelLAEsToken_);

  PixelTopologyMap map = PixelTopologyMap(geom, tTopo);
  hists.nlay = geom->numberOfLayers(PixelSubdetector::PixelBarrel);
  hists.nModules_.resize(hists.nlay);
  for (int i = 0; i < hists.nlay; i++) {
    hists.nModules_[i] = map.getPXBModules(i + 1);
  }

  // list of modules already filled, then return (we already entered here)
  if (!hists.BPixnewDetIds_.empty() || !hists.FPixnewDetIds_.empty())
    return;

  if (!newmodulelist_.empty()) {
    for (auto const& modulename : newmodulelist_) {
      if (modulename.find("BPix_") != std::string::npos) {
        PixelBarrelName bn(modulename, true);
        const auto& detId = bn.getDetId(tTopo);
        hists.BPixnewmodulename_.push_back(modulename);
        hists.BPixnewDetIds_.push_back(detId.rawId());
        hists.BPixnewModule_.push_back(bn.moduleName());
        hists.BPixnewLayer_.push_back(bn.layerName());
      } else if (modulename.find("FPix_") != std::string::npos) {
        PixelEndcapName en(modulename, true);
        const auto& detId = en.getDetId(tTopo);
        hists.FPixnewmodulename_.push_back(modulename);
        hists.FPixnewDetIds_.push_back(detId.rawId());
        hists.FPixnewDisk_.push_back(en.diskName());
        hists.FPixnewBlade_.push_back(en.bladeName());
      }
    }
  }

  uint count = 0;
  for (const auto& id : hists.BPixnewDetIds_) {
    LogDebug("SiPixelLorentzAnglePCLHarvester") << id << std::endl;
    count++;
  }
  LogDebug("SiPixelLorentzAnglePCLHarvester") << "Stored a total of " << count << " new detIds." << std::endl;

  // list of modules already filled, return (we already entered here)
  if (!hists.detIdsList.empty())
    return;

  std::vector<uint32_t> treatedIndices;

  for (auto det : geom->detsPXB()) {
    const PixelGeomDetUnit* pixelDet = dynamic_cast<const PixelGeomDetUnit*>(det);
    width_ = pixelDet->surface().bounds().thickness();
    const auto& layer = tTopo->pxbLayer(pixelDet->geographicalId());
    const auto& module = tTopo->pxbModule(pixelDet->geographicalId());
    int i_index = module + (layer - 1) * hists.nModules_[layer - 1];

    uint32_t rawId = pixelDet->geographicalId().rawId();

    // if the detId is already accounted for in the special class, do not attach it
    if (std::find(hists.BPixnewDetIds_.begin(), hists.BPixnewDetIds_.end(), rawId) != hists.BPixnewDetIds_.end())
      continue;

    if (std::find(treatedIndices.begin(), treatedIndices.end(), i_index) != treatedIndices.end()) {
      hists.detIdsList.at(i_index).push_back(rawId);
    } else {
      hists.detIdsList.insert(std::pair<uint32_t, std::vector<uint32_t>>(i_index, {rawId}));
      treatedIndices.push_back(i_index);
    }
  }

  count = 0;
  for (const auto& i : treatedIndices) {
    for (const auto& id : hists.detIdsList.at(i)) {
      LogDebug("SiPixelLorentzAnglePCLHarvester") << id << std::endl;
      count++;
    };
  }
  LogDebug("SiPixelLorentzAnglePCLHarvester") << "Stored a total of " << count << " detIds." << std::endl;
}

//------------------------------------------------------------------------------
void SiPixelLorentzAnglePCLHarvester::dqmEndJob(DQMStore::IBooker& iBooker, DQMStore::IGetter& iGetter) {
  // go in the right directory
  iGetter.cd();
  iGetter.setCurrentFolder(dqmDir_);

  // B-field value
  GlobalPoint center(0.0, 0.0, 0.0);
  float theMagField = magField->inTesla(center).mag();

  // fetch the 2D histograms
  for (int i_layer = 1; i_layer <= hists.nlay; i_layer++) {
    const auto& prefix_ = fmt::sprintf("%s/BPix/BPixLayer%i", dqmDir_, i_layer);
    for (int i_module = 1; i_module <= hists.nModules_[i_layer - 1]; i_module++) {
      int i_index = i_module + (i_layer - 1) * hists.nModules_[i_layer - 1];

      hists.h_drift_depth_[i_index] =
          iGetter.get(fmt::format("{}/h_drift_depth_layer{}_module{}", prefix_, i_layer, i_module));

      if (hists.h_drift_depth_[i_index] == nullptr) {
        edm::LogError("SiPixelLorentzAnglePCLHarvester::dqmEndJob")
            << "Failed to retrieve electron drift over depth for layer " << i_layer << ", module " << i_module << ".";
        continue;
      }

      hists.h_drift_depth_adc_[i_index] =
          iGetter.get(fmt::format("{}/h_drift_depth_adc_layer{}_module{}", prefix_, i_layer, i_module));

      hists.h_drift_depth_adc2_[i_index] =
          iGetter.get(fmt::format("{}/h_drift_depth_adc2_layer{}_module{}", prefix_, i_layer, i_module));

      hists.h_drift_depth_noadc_[i_index] =
          iGetter.get(fmt::format("{}/h_drift_depth_noadc_layer{}_module{}", prefix_, i_layer, i_module));

      hists.h_mean_[i_index] = iGetter.get(fmt::format("{}/h_mean_layer{}_module{}", dqmDir_, i_layer, i_module));

      hists.h_drift_depth_[i_index]->divide(
          hists.h_drift_depth_adc_[i_index], hists.h_drift_depth_noadc_[i_index], 1., 1., "");
    }
  }

  // fetch the new modules 2D histograms
  for (int i = 0; i < (int)hists.BPixnewDetIds_.size(); i++) {
    int new_index = i + 1 + hists.nModules_[hists.nlay - 1] + (hists.nlay - 1) * hists.nModules_[hists.nlay - 1];

    hists.h_drift_depth_[new_index] = iGetter.get(
        fmt::format("{}/h_BPixnew_drift_depth_{}", dqmDir_ + "/BPix/NewModules", hists.BPixnewmodulename_[i]));

    if (hists.h_drift_depth_[new_index] == nullptr) {
      edm::LogError("SiPixelLorentzAnglePCLHarvester::dqmEndJob")
          << "Failed to retrieve electron drift over depth for new module " << hists.BPixnewmodulename_[i] << ".";
      continue;
    }

    hists.h_drift_depth_adc_[new_index] = iGetter.get(
        fmt::format("{}/h_BPixnew_drift_depth_adc_{}", dqmDir_ + "/BPix/NewModules", hists.BPixnewmodulename_[i]));

    hists.h_drift_depth_adc2_[new_index] = iGetter.get(
        fmt::format("{}/h_BPixnew_drift_depth_adc2_{}", dqmDir_ + "/BPix/NewModules", hists.BPixnewmodulename_[i]));

    hists.h_drift_depth_noadc_[new_index] = iGetter.get(
        fmt::format("{}/h_BPixnew_drift_depth_noadc_{}", dqmDir_ + "/BPix/NewModules", hists.BPixnewmodulename_[i]));

    hists.h_mean_[new_index] = iGetter.get(fmt::format("{}/h_BPixnew_mean_{}", dqmDir_, hists.BPixnewmodulename_[i]));

    hists.h_drift_depth_[new_index]->divide(
        hists.h_drift_depth_adc_[new_index], hists.h_drift_depth_noadc_[new_index], 1., 1., "");
  }

  hists.h_bySectLA_ = iGetter.get(fmt::format("{}/h_bySectorLA", dqmDir_ + "/SectorMonitoring"));
  hists.h_bySectChi2_ = iGetter.get(fmt::format("{}/h_bySectorChi2", dqmDir_ + "/SectorMonitoring"));

  int hist_drift_;
  int hist_depth_;
  double min_drift_;
  double max_drift_;

  if (hists.h_drift_depth_adc_[1] != nullptr) {
    hist_drift_ = hists.h_drift_depth_adc_[1]->getNbinsX();
    hist_depth_ = hists.h_drift_depth_adc_[1]->getNbinsY();
    min_drift_ = hists.h_drift_depth_adc_[1]->getAxisMin(1);
    max_drift_ = hists.h_drift_depth_adc_[1]->getAxisMax(1);
  } else {
    hist_drift_ = 100;
    hist_depth_ = 50;
    min_drift_ = -500.;
    max_drift_ = 500.;
  }

  iBooker.setCurrentFolder("AlCaReco/SiPixelLorentzAngleHarvesting/");
  MonitorElement* h_drift_depth_adc_slice_ =
      iBooker.book1D("h_drift_depth_adc_slice", "slice of adc histogram", hist_drift_, min_drift_, max_drift_);

  // clang-format off
  edm::LogPrint("LorentzAngle") << "module" << "\t" << "layer" << "\t"
                                << "offset" << "\t" << "e0" << "\t"
                                << "slope"  << "\t" << "e1" << "\t"
                                << "rel.err" << "\t" << "pull" << "\t"
                                << "p2" << "\t" << "e2" << "\t"
                                << "p3" << "\t" << "e3" << "\t"
                                << "p4" << "\t" << "e4" << "\t"
                                << "p5" << "\t" << "e5" << "\t"
                                << "chi2" << "\t" << "prob" << "\t"
                                << "newDetId" << "\t" << "tan(LA)" << "\t"
                                << "Error(LA)" << std::endl;
  // clang-format on

  std::unique_ptr<SiPixelLorentzAngle> LorentzAngle = std::make_unique<SiPixelLorentzAngle>();

  f1 = std::make_unique<TF1>("f1", "[0] + [1]*x + [2]*x*x + [3]*x*x*x + [4]*x*x*x*x + [5]*x*x*x*x*x", 5., 280.);
  f1->SetParName(0, "offset");
  f1->SetParName(1, "tan#theta_{LA}");
  f1->SetParName(2, "quad term");
  f1->SetParName(3, "cubic term");
  f1->SetParName(4, "quartic term");
  f1->SetParName(5, "quintic term");

  double p1_simul_newmodule = 0.294044;
  double half_width = width_ * 10000 / 2;  // pixel half thickness in units of micro meter

  for (int j = 0; j < (int)hists.BPixnewDetIds_.size(); j++) {
    uint32_t rawId = hists.BPixnewDetIds_[j];
    int new_index = j + 1 + hists.nModules_[hists.nlay - 1] + (hists.nlay - 1) * hists.nModules_[hists.nlay - 1];
    if (hists.h_drift_depth_adc_[new_index] == nullptr)
      continue;
    for (int i = 1; i <= hist_depth_; i++) {
      findMean(h_drift_depth_adc_slice_, i, new_index);
    }

    f1->SetParameter(0, 0);
    f1->SetParError(0, 0);
    f1->SetParameter(1, 0.4);
    f1->SetParError(1, 0);
    f1->SetParameter(2, 0.0);
    f1->SetParError(2, 0);
    f1->SetParameter(3, 0.0);
    f1->SetParError(3, 0);
    f1->SetParameter(4, 0.0);
    f1->SetParError(4, 0);
    f1->SetParameter(5, 0.0);
    f1->SetParError(5, 0);
    f1->SetChisquare(0);

    hists.h_mean_[new_index]->getTH1()->Fit(f1.get(), "ERQ");

    double p0 = f1->GetParameter(0);
    double e0 = f1->GetParError(0);
    double p1 = f1->GetParameter(1);
    double e1 = f1->GetParError(1);
    double p2 = f1->GetParameter(2);
    double e2 = f1->GetParError(2);
    double p3 = f1->GetParameter(3);
    double e3 = f1->GetParError(3);
    double p4 = f1->GetParameter(4);
    double e4 = f1->GetParError(4);
    double p5 = f1->GetParameter(5);
    double e5 = f1->GetParError(5);
    double chi2 = f1->GetChisquare();
    int ndf = f1->GetNDF();
    double prob = f1->GetProb();
    double redChi2 = ndf > 0. ? chi2 / ndf : 0.;

    double f1_halfwidth = p0 + p1 * half_width + p2 * pow(half_width, 2) + p3 * pow(half_width, 3) +
                          p4 * pow(half_width, 4) + p5 * pow(half_width, 5);

    double f1_zerowidth = p0;

    double tan_LA =
        (f1_halfwidth - f1_zerowidth) / half_width;  // tan_LA = (f1(x = half_width) - f1(x = 0)) / (half_width - 0)
    double errsq_LA = (pow(e1, 2) + pow((half_width * e2), 2) + pow((half_width * half_width * e3), 2) +
                       pow((half_width * half_width * half_width * e4), 2) +
                       pow((half_width * half_width * half_width * half_width * e5), 2));  // Propagation of uncertainty
    double error_LA = sqrt(errsq_LA);

    hists.h_bySectLA_->setBinContent(new_index, (tan_LA / theMagField));
    hists.h_bySectLA_->setBinError(new_index, (error_LA / theMagField));
    hists.h_bySectChi2_->setBinContent(new_index, redChi2);

    edm::LogPrint("LorentzAngle") << std::setprecision(4) << hists.BPixnewModule_[j] << "\t" << hists.BPixnewLayer_[j]
                                  << "\t" << p0 << "\t" << e0 << "\t" << p1 << std::setprecision(3) << "\t" << e1
                                  << "\t" << e1 / p1 * 100. << "\t" << (p1 - p1_simul_newmodule) / e1 << "\t" << p2
                                  << "\t" << e2 << "\t" << p3 << "\t" << e3 << "\t" << p4 << "\t" << e4 << "\t" << p5
                                  << "\t" << e5 << "\t" << chi2 << "\t" << prob << "\t" << hists.BPixnewDetIds_[j]
                                  << "\t" << tan_LA << "\t" << error_LA << std::endl;

    float bPixLorentzAnglePerTesla_;
    // if the fit quality is OK
    if (prob > fitProbCut_) {
      bPixLorentzAnglePerTesla_ = tan_LA / theMagField;
      if (!LorentzAngle->putLorentzAngle(rawId, bPixLorentzAnglePerTesla_)) {
        edm::LogError("SiPixelLorentzAnglePCLHarvester")
            << "[SiPixelLorentzAnglePCLHarvester::dqmEndRun] filling new modules: detid already exists" << std::endl;
      }
    } else {
      // just copy the values from the existing payload
      bPixLorentzAnglePerTesla_ = currentLorentzAngle->getLorentzAngle(rawId);
      if (!LorentzAngle->putLorentzAngle(rawId, bPixLorentzAnglePerTesla_)) {
        edm::LogError("SiPixelLorentzAnglePCLHarvester")
            << "[SiPixelLorentzAnglePCLHarvester::dqmEndRun] filling new modules (from current): detid already exists"
            << std::endl;
      }
    }
  }  // loop on BPix new modules

  double p1_simul[hists.nlay][hists.nModules_[hists.nlay - 1]];
  for (int i_layer = 1; i_layer <= hists.nlay; i_layer++) {
    for (int i_module = 1; i_module <= hists.nModules_[i_layer - 1]; i_module++) {
      if (i_layer == 1)
        p1_simul[i_layer - 1][i_module - 1] = 0.436848;
      else if (i_layer == 2)
        p1_simul[i_layer - 1][i_module - 1] = 0.25802;
      else if (i_layer == 3 && i_module <= 4)
        p1_simul[i_layer - 1][i_module - 1] = 0.29374;
      else if (i_layer == 3 && i_module >= 5)
        p1_simul[i_layer - 1][i_module - 1] = 0.31084;
      else if (i_layer == 4 && i_module <= 4)
        p1_simul[i_layer - 1][i_module - 1] = 0.29944;
      else
        p1_simul[i_layer - 1][i_module - 1] = 0.31426;
    }
  }

  //loop over modlues and layers to fit the lorentz angle
  for (int i_layer = 1; i_layer <= hists.nlay; i_layer++) {
    for (int i_module = 1; i_module <= hists.nModules_[i_layer - 1]; i_module++) {
      int i_index = i_module + (i_layer - 1) * hists.nModules_[i_layer - 1];
      if (hists.h_drift_depth_adc_[i_index] == nullptr)
        continue;
      //loop over bins in depth (z-local-coordinate) (in order to fit slices)
      for (int i = 1; i <= hist_depth_; i++) {
        findMean(h_drift_depth_adc_slice_, i, i_index);
      }  // end loop over bins in depth

      f1->SetParameter(0, 0);
      f1->SetParError(0, 0);
      f1->SetParameter(1, 0.4);
      f1->SetParError(1, 0);
      f1->SetParameter(2, 0.0);
      f1->SetParError(2, 0);
      f1->SetParameter(3, 0.0);
      f1->SetParError(3, 0);
      f1->SetParameter(4, 0.0);
      f1->SetParError(4, 0);
      f1->SetParameter(5, 0.0);
      f1->SetParError(5, 0);
      f1->SetChisquare(0);

      hists.h_mean_[i_index]->getTH1()->Fit(f1.get(), "ERQ");
      double p0 = f1->GetParameter(0);
      double e0 = f1->GetParError(0);
      double p1 = f1->GetParameter(1);
      double e1 = f1->GetParError(1);
      double p2 = f1->GetParameter(2);
      double e2 = f1->GetParError(2);
      double p3 = f1->GetParameter(3);
      double e3 = f1->GetParError(3);
      double p4 = f1->GetParameter(4);
      double e4 = f1->GetParError(4);
      double p5 = f1->GetParameter(5);
      double e5 = f1->GetParError(5);
      double chi2 = f1->GetChisquare();
      double prob = f1->GetProb();
      int ndf = f1->GetNDF();
      double redChi2 = ndf > 0. ? chi2 / ndf : 0.;

      double f1_halfwidth = p0 + p1 * half_width + p2 * pow(half_width, 2) + p3 * pow(half_width, 3) +
                            p4 * pow(half_width, 4) + p5 * pow(half_width, 5);

      double f1_zerowidth = p0;

      double tan_LA =
          (f1_halfwidth - f1_zerowidth) / half_width;  // tan_LA = (f1(x = half_width) - f1(x = 0)) / (half_width - 0)
      double errsq_LA =
          (pow(e1, 2) + pow((half_width * e2), 2) + pow((half_width * half_width * e3), 2) +
           pow((half_width * half_width * half_width * e4), 2) +
           pow((half_width * half_width * half_width * half_width * e5), 2));  // Propagation of uncertainty
      double error_LA = sqrt(errsq_LA);

      hists.h_bySectLA_->setBinContent(i_index, (tan_LA / theMagField));
      hists.h_bySectLA_->setBinError(i_index, (error_LA / theMagField));
      hists.h_bySectChi2_->setBinContent(i_index, redChi2);

      edm::LogPrint("LorentzAngle") << std::setprecision(4) << i_module << "\t" << i_layer << "\t" << p0 << "\t" << e0
                                    << "\t" << p1 << std::setprecision(3) << "\t" << e1 << "\t" << e1 / p1 * 100.
                                    << "\t" << (p1 - p1_simul[i_layer - 1][i_module - 1]) / e1 << "\t" << p2 << "\t"
                                    << e2 << "\t" << p3 << "\t" << e3 << "\t" << p4 << "\t" << e4 << "\t" << p5 << "\t"
                                    << e5 << "\t" << chi2 << "\t" << prob << "\t"
                                    << "null"
                                    << "\t" << tan_LA << "\t" << error_LA << std::endl;

      const auto& detIdsToFill = hists.detIdsList.at(i_index);

      LogDebug("SiPixelLorentzAnglePCLHarvester")
          << "index: " << i_index << " i_module: " << i_module << " i_layer: " << i_layer << std::endl;
      for (const auto& id : detIdsToFill) {
        LogDebug("SiPixelLorentzAnglePCLHarvester") << id << ",";
      }
      LogDebug("SiPixelLorentzAnglePCLHarvester") << std::endl;

      float bPixLorentzAnglePerTesla_;
      // if the fit quality is OK
      if (prob > fitProbCut_) {
        for (const auto& id : detIdsToFill) {
          bPixLorentzAnglePerTesla_ = tan_LA / theMagField;
          if (!LorentzAngle->putLorentzAngle(id, bPixLorentzAnglePerTesla_)) {
            edm::LogError("SiPixelLorentzAnglePCLHarvester")
                << "[SiPixelLorentzAnglePCLHarvester::dqmEndRun] filling BPix: detid already exists" << std::endl;
          }
        }
      } else {
        // just copy the values from the existing payload
        for (const auto& id : detIdsToFill) {
          bPixLorentzAnglePerTesla_ = currentLorentzAngle->getLorentzAngle(id);
          if (!LorentzAngle->putLorentzAngle(id, bPixLorentzAnglePerTesla_)) {
            edm::LogError("SiPixelLorentzAnglePCLHarvester")
                << "[SiPixelLorentzAnglePCLHarvester::dqmEndRun] filling BPix (from current): detid already exists"
                << std::endl;
          }
        }
      }
    }
  }  // end loop over modules and layers

  // fill the rest of DetIds not filled above (for the moment FPix)
  const auto& currentLAMap = currentLorentzAngle->getLorentzAngles();
  const auto& newLAMap = LorentzAngle->getLorentzAngles();
  std::vector<unsigned int> currentLADets;
  std::vector<unsigned int> newLADets;

  std::transform(currentLAMap.begin(),
                 currentLAMap.end(),
                 std::back_inserter(currentLADets),
                 [](const std::map<unsigned int, float>::value_type& pair) { return pair.first; });

  std::transform(newLAMap.begin(),
                 newLAMap.end(),
                 std::back_inserter(newLADets),
                 [](const std::map<unsigned int, float>::value_type& pair) { return pair.first; });

  std::vector<unsigned int> notCommon;
  std::set_symmetric_difference(
      currentLADets.begin(), currentLADets.end(), newLADets.begin(), newLADets.end(), std::back_inserter(notCommon));

  for (const auto& id : notCommon) {
    float fPixLorentzAnglePerTesla_ = currentLorentzAngle->getLorentzAngle(id);
    if (!LorentzAngle->putLorentzAngle(id, fPixLorentzAnglePerTesla_)) {
      edm::LogError("SiPixelLorentzAnglePCLHarvester")
          << "[SiPixelLorentzAnglePCLHarvester::dqmEndRun] filling rest of payload: detid already exists" << std::endl;
    }
  }

  // book histogram of differences
  MonitorElement* h_diffLA = iBooker.book1D(
      "h_diffLA", "difference in #mu_{H}; #Delta #mu_{H}/#mu_{H} (old-new)/old [%];n. modules", 100, -10, 10);

  for (const auto& id : newLADets) {
    float deltaMuHoverMuH = (currentLorentzAngle->getLorentzAngle(id) - LorentzAngle->getLorentzAngle(id)) /
                            currentLorentzAngle->getLorentzAngle(id);
    h_diffLA->Fill(deltaMuHoverMuH);
  }

  // fill the DB object record
  edm::Service<cond::service::PoolDBOutputService> mydbservice;
  if (mydbservice.isAvailable()) {
    try {
      mydbservice->writeOneIOV(*LorentzAngle, mydbservice->currentTime(), recordName_);
    } catch (const cond::Exception& er) {
      edm::LogError("SiPixelLorentzAngleDB") << er.what() << std::endl;
    } catch (const std::exception& er) {
      edm::LogError("SiPixelLorentzAngleDB") << "caught std::exception " << er.what() << std::endl;
    }
  } else {
    edm::LogError("SiPixelLorentzAngleDB") << "Service is unavailable" << std::endl;
  }
}

//------------------------------------------------------------------------------
void SiPixelLorentzAnglePCLHarvester::findMean(MonitorElement* h_drift_depth_adc_slice_, int i, int i_ring) {
  double nentries = 0;
  h_drift_depth_adc_slice_->Reset();
  int hist_drift_ = h_drift_depth_adc_slice_->getNbinsX();

  // determine sigma and sigma^2 of the adc counts and average adc counts
  //loop over bins in drift width
  for (int j = 1; j <= hist_drift_; j++) {
    if (hists.h_drift_depth_noadc_[i_ring]->getBinContent(j, i) >= 1) {
      double adc_error2 = (hists.h_drift_depth_adc2_[i_ring]->getBinContent(j, i) -
                           hists.h_drift_depth_adc_[i_ring]->getBinContent(j, i) *
                               hists.h_drift_depth_adc_[i_ring]->getBinContent(j, i) /
                               hists.h_drift_depth_noadc_[i_ring]->getBinContent(j, i)) /
                          hists.h_drift_depth_noadc_[i_ring]->getBinContent(j, i);

      hists.h_drift_depth_adc_[i_ring]->setBinError(j, i, sqrt(adc_error2));
      double error2 = adc_error2 / (hists.h_drift_depth_noadc_[i_ring]->getBinContent(j, i) - 1.);
      hists.h_drift_depth_[i_ring]->setBinError(j, i, sqrt(error2));
    } else {
      hists.h_drift_depth_[i_ring]->setBinError(j, i, 0);
      hists.h_drift_depth_adc_[i_ring]->setBinError(j, i, 0);
    }
    h_drift_depth_adc_slice_->setBinContent(j, hists.h_drift_depth_adc_[i_ring]->getBinContent(j, i));
    h_drift_depth_adc_slice_->setBinError(j, hists.h_drift_depth_adc_[i_ring]->getBinError(j, i));
    nentries += hists.h_drift_depth_noadc_[i_ring]->getBinContent(j, i);
  }  // end loop over bins in drift width

  double mean = h_drift_depth_adc_slice_->getMean(1);
  double error = 0;
  if (nentries != 0) {
    error = h_drift_depth_adc_slice_->getRMS(1) / std::sqrt(nentries);
  }
  hists.h_mean_[i_ring]->setBinContent(i, mean);
  hists.h_mean_[i_ring]->setBinError(i, error);
}

//------------------------------------------------------------------------------
void SiPixelLorentzAnglePCLHarvester::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setComment("Harvester module of the SiPixel Lorentz Angle PCL monitoring workflow");
  desc.add<std::vector<std::string>>("newmodulelist", {})->setComment("the list of DetIds for new sensors");
  desc.add<std::string>("dqmDir", "AlCaReco/SiPixelLorentzAngle")->setComment("the directory of PCL Worker output");
  desc.add<double>("fitProbCut", 0.1)->setComment("cut on fit chi2 probabiblity to accept measurement");
  desc.add<std::string>("record", "SiPixelLorentzAngleRcd")->setComment("target DB record");
  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(SiPixelLorentzAnglePCLHarvester);
