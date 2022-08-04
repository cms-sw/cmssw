// -*- C++ -*-
//
// Package:    CalibTracker/SiPixelLorentzAnglePCLHarvester
// Class:      SiPixelLorentzAnglePCLHarvester
//
/**\class SiPixelLorentzAnglePCLHarvester SiPixelLorentzAnglePCLHarvester.cc CalibTracker/SiPixelLorentzAngle/src/SiPixelLorentzAnglePCLHarvester.cc
 Description: reads the intermediate ALCAPROMPT DQMIO-like dataset and performs the fitting of the SiPixel Lorentz Angle in the Prompt Calibration Loop
 Implementation:
     Reads the 2D histograms of the drift vs depth created by SiPixelLorentzAnglePCLWorker modules and generates 1D profiles which are then fit
     with a 5th order polinomial. The extracted value of the tan(theta_L)/B are stored in an output sqlite file which is then uploaded to the conditions database
*/
//
// Original Author:  mmusich
//         Created:  Sat, 29 May 2021 14:46:19 GMT
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

/* 
 * Auxilliary struct to store fit results
 */
namespace SiPixelLAHarvest {
  struct fitResults {
  public:
    fitResults() {
      // set all parameters to default
      p0 = p1 = p2 = p3 = p4 = p5 = 0.;
      e0 = e1 = e2 = e3 = e4 = e5 = 0.;
      chi2 = prob = redChi2 = tan_LA = error_LA = -9999.;
      ndf = -999;
    };

    double p0;
    double e0;
    double p1;
    double e1;
    double p2;
    double e2;
    double p3;
    double e3;
    double p4;
    double e4;
    double p5;
    double e5;
    double chi2;
    int ndf;
    double prob;
    double redChi2;
    double tan_LA;
    double error_LA;
  };
}  // namespace SiPixelLAHarvest

//------------------------------------------------------------------------------
class SiPixelLorentzAnglePCLHarvester : public DQMEDHarvester {
public:
  SiPixelLorentzAnglePCLHarvester(const edm::ParameterSet&);
  ~SiPixelLorentzAnglePCLHarvester() override = default;
  void beginRun(const edm::Run&, const edm::EventSetup&) override;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  void dqmEndJob(DQMStore::IBooker&, DQMStore::IGetter&) override;
  void endRun(const edm::Run&, const edm::EventSetup&) override;
  void findMean(MonitorElement* h_drift_depth_adc_slice_, int i, int i_ring);
  SiPixelLAHarvest::fitResults fitAndStore(std::shared_ptr<SiPixelLorentzAngle> theLA, int i_idx, int i_lay, int i_mod);

  // es tokens
  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomEsToken_;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> topoEsTokenBR_, topoEsTokenER_;
  edm::ESGetToken<SiPixelLorentzAngle, SiPixelLorentzAngleRcd> siPixelLAEsToken_;
  edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magneticFieldToken_;

  std::vector<std::string> newmodulelist_;
  const std::string dqmDir_;
  const double fitChi2Cut_;
  const int minHitsCut_;
  const std::string recordName_;
  std::unique_ptr<TF1> f1_;
  float width_;
  float theMagField_{0.f};

  static constexpr float inverseGeVtoTesla_ = 2.99792458e-3f;

  SiPixelLorentzAngleCalibrationHistograms hists_;
  const SiPixelLorentzAngle* currentLorentzAngle_;
  std::unique_ptr<TrackerTopology> theTrackerTopology_;
};

//------------------------------------------------------------------------------
SiPixelLorentzAnglePCLHarvester::SiPixelLorentzAnglePCLHarvester(const edm::ParameterSet& iConfig)
    : geomEsToken_(esConsumes<edm::Transition::BeginRun>()),
      topoEsTokenBR_(esConsumes<edm::Transition::BeginRun>()),
      topoEsTokenER_(esConsumes<edm::Transition::EndRun>()),
      siPixelLAEsToken_(esConsumes<edm::Transition::BeginRun>()),
      magneticFieldToken_(esConsumes<edm::Transition::BeginRun>()),
      newmodulelist_(iConfig.getParameter<std::vector<std::string>>("newmodulelist")),
      dqmDir_(iConfig.getParameter<std::string>("dqmDir")),
      fitChi2Cut_(iConfig.getParameter<double>("fitChi2Cut")),
      minHitsCut_(iConfig.getParameter<int>("minHitsCut")),
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
  const TrackerTopology* tTopo = &iSetup.getData(topoEsTokenBR_);

  const MagneticField* magField = &iSetup.getData(magneticFieldToken_);
  currentLorentzAngle_ = &iSetup.getData(siPixelLAEsToken_);

  // B-field value
  // inverseBzAtOriginInGeV() returns the inverse of field z component for this map in GeV
  // for the conversion please consult https://github.com/cms-sw/cmssw/blob/master/MagneticField/Engine/src/MagneticField.cc#L17
  // theInverseBzAtOriginInGeV = 1.f / (at0z * 2.99792458e-3f);
  // ==> at0z = 1.f / (theInverseBzAtOriginInGeV * 2.99792458e-3f)

  theMagField_ = 1.f / (magField->inverseBzAtOriginInGeV() * inverseGeVtoTesla_);

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
    LogDebug("SiPixelLorentzAnglePCLHarvester") << id;
    count++;
  }
  LogDebug("SiPixelLorentzAnglePCLHarvester") << "Stored a total of " << count << " new detIds.";

  // list of modules already filled, return (we already entered here)
  if (!hists_.detIdsList.empty())
    return;

  std::vector<uint32_t> treatedIndices;

  for (auto det : geom->detsPXB()) {
    const PixelGeomDetUnit* pixelDet = dynamic_cast<const PixelGeomDetUnit*>(det);
    width_ = pixelDet->surface().bounds().thickness();
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
      LogDebug("SiPixelLorentzAnglePCLHarvester") << id;
      count++;
    };
  }
  LogDebug("SiPixelLorentzAnglePCLHarvester") << "Stored a total of " << count << " detIds.";
}

//------------------------------------------------------------------------------
void SiPixelLorentzAnglePCLHarvester::endRun(edm::Run const& run, edm::EventSetup const& isetup) {
  if (!theTrackerTopology_) {
    theTrackerTopology_ = std::make_unique<TrackerTopology>(isetup.getData(topoEsTokenER_));
  }
}

//------------------------------------------------------------------------------
void SiPixelLorentzAnglePCLHarvester::dqmEndJob(DQMStore::IBooker& iBooker, DQMStore::IGetter& iGetter) {
  // go in the right directory
  iGetter.cd();
  iGetter.setCurrentFolder(dqmDir_);

  // fetch the 2D histograms
  for (int i_layer = 1; i_layer <= hists_.nlay; i_layer++) {
    const auto& prefix_ = fmt::sprintf("%s/BPix/BPixLayer%i", dqmDir_, i_layer);
    for (int i_module = 1; i_module <= hists_.nModules_[i_layer - 1]; i_module++) {
      int i_index = i_module + (i_layer - 1) * hists_.nModules_[i_layer - 1];

      hists_.h_drift_depth_[i_index] =
          iGetter.get(fmt::format("{}/h_drift_depth_layer{}_module{}", prefix_, i_layer, i_module));

      if (hists_.h_drift_depth_[i_index] == nullptr) {
        edm::LogError("SiPixelLorentzAnglePCLHarvester::dqmEndJob")
            << "Failed to retrieve electron drift over depth for layer " << i_layer << ", module " << i_module << ".";
        continue;
      }

      hists_.h_drift_depth_adc_[i_index] =
          iGetter.get(fmt::format("{}/h_drift_depth_adc_layer{}_module{}", prefix_, i_layer, i_module));

      hists_.h_drift_depth_adc2_[i_index] =
          iGetter.get(fmt::format("{}/h_drift_depth_adc2_layer{}_module{}", prefix_, i_layer, i_module));

      hists_.h_drift_depth_noadc_[i_index] =
          iGetter.get(fmt::format("{}/h_drift_depth_noadc_layer{}_module{}", prefix_, i_layer, i_module));

      hists_.h_mean_[i_index] = iGetter.get(fmt::format("{}/h_mean_layer{}_module{}", dqmDir_, i_layer, i_module));

      hists_.h_drift_depth_[i_index]->divide(
          hists_.h_drift_depth_adc_[i_index], hists_.h_drift_depth_noadc_[i_index], 1., 1., "");
    }
  }

  // fetch the new modules 2D histograms
  for (int i = 0; i < (int)hists_.BPixnewDetIds_.size(); i++) {
    int new_index = i + 1 + hists_.nModules_[hists_.nlay - 1] + (hists_.nlay - 1) * hists_.nModules_[hists_.nlay - 1];

    hists_.h_drift_depth_[new_index] = iGetter.get(
        fmt::format("{}/h_BPixnew_drift_depth_{}", dqmDir_ + "/BPix/NewModules", hists_.BPixnewmodulename_[i]));

    if (hists_.h_drift_depth_[new_index] == nullptr) {
      edm::LogError("SiPixelLorentzAnglePCLHarvester")
          << "Failed to retrieve electron drift over depth for new module " << hists_.BPixnewmodulename_[i] << ".";
      continue;
    }

    hists_.h_drift_depth_adc_[new_index] = iGetter.get(
        fmt::format("{}/h_BPixnew_drift_depth_adc_{}", dqmDir_ + "/BPix/NewModules", hists_.BPixnewmodulename_[i]));

    hists_.h_drift_depth_adc2_[new_index] = iGetter.get(
        fmt::format("{}/h_BPixnew_drift_depth_adc2_{}", dqmDir_ + "/BPix/NewModules", hists_.BPixnewmodulename_[i]));

    hists_.h_drift_depth_noadc_[new_index] = iGetter.get(
        fmt::format("{}/h_BPixnew_drift_depth_noadc_{}", dqmDir_ + "/BPix/NewModules", hists_.BPixnewmodulename_[i]));

    hists_.h_mean_[new_index] = iGetter.get(fmt::format("{}/h_BPixnew_mean_{}", dqmDir_, hists_.BPixnewmodulename_[i]));

    hists_.h_drift_depth_[new_index]->divide(
        hists_.h_drift_depth_adc_[new_index], hists_.h_drift_depth_noadc_[new_index], 1., 1., "");
  }

  hists_.h_bySectOccupancy_ = iGetter.get(fmt::format("{}/h_bySectorOccupancy", dqmDir_ + "/SectorMonitoring"));
  if (hists_.h_bySectOccupancy_ == nullptr) {
    edm::LogError("SiPixelLorentzAnglePCLHarvester") << "Failed to retrieve the hit on track occupancy.";
    return;
  }

  int hist_drift_;
  int hist_depth_;
  double min_drift_;
  double max_drift_;

  if (hists_.h_drift_depth_adc_[1] != nullptr) {
    hist_drift_ = hists_.h_drift_depth_adc_[1]->getNbinsX();
    hist_depth_ = hists_.h_drift_depth_adc_[1]->getNbinsY();
    min_drift_ = hists_.h_drift_depth_adc_[1]->getAxisMin(1);
    max_drift_ = hists_.h_drift_depth_adc_[1]->getAxisMax(1);
  } else {
    hist_drift_ = 100;
    hist_depth_ = 50;
    min_drift_ = -500.;
    max_drift_ = 500.;
  }

  iBooker.setCurrentFolder(fmt::format("{}Harvesting", dqmDir_));
  MonitorElement* h_drift_depth_adc_slice_ =
      iBooker.book1D("h_drift_depth_adc_slice", "slice of adc histogram", hist_drift_, min_drift_, max_drift_);

  // book histogram of differences
  MonitorElement* h_diffLA = iBooker.book1D(
      "h_diffLA", "difference in #mu_{H}; #Delta #mu_{H}/#mu_{H} (old-new)/old [%];n. modules", 100, -150, 150);

  // retrieve the number of bins from the other monitoring histogram
  const auto& maxSect = hists_.h_bySectOccupancy_->getNbinsX();
  const double lo = -0.5;
  const double hi = maxSect - 0.5;

  // this will be booked in the Harvesting folder
  iBooker.setCurrentFolder(fmt::format("{}Harvesting/SectorMonitoring", dqmDir_));
  std::string repText = "%s tan#theta_{LA}/B by sector;pixel sector;%s tan(#theta_{LA})/B [1/T]";
  hists_.h_bySectMeasLA_ =
      iBooker.book1D("h_LAbySector_Measured", fmt::sprintf(repText, "measured", "measured"), maxSect, lo, hi);
  hists_.h_bySectSetLA_ =
      iBooker.book1D("h_LAbySector_Accepted", fmt::sprintf(repText, "accepted", "accepted"), maxSect, lo, hi);
  hists_.h_bySectRejectLA_ =
      iBooker.book1D("h_LAbySector_Rejected", fmt::sprintf(repText, "rejected", "rejected"), maxSect, lo, hi);
  hists_.h_bySectLA_ = iBooker.book1D("h_LAbySector", fmt::sprintf(repText, "payload", "payload"), maxSect, lo, hi);
  hists_.h_bySectDeltaLA_ =
      iBooker.book1D("h_deltaLAbySector", fmt::sprintf(repText, "#Delta", "#Delta"), maxSect, lo, hi);
  hists_.h_bySectChi2_ =
      iBooker.book1D("h_bySectorChi2", "Fit #chi^{2}/ndf by sector;pixel sector; fit #chi^{2}/ndf", maxSect, lo, hi);

  // copy the bin labels from the occupancy histogram
  for (int bin = 1; bin <= maxSect; bin++) {
    const auto& binName = hists_.h_bySectOccupancy_->getTH1()->GetXaxis()->GetBinLabel(bin);
    hists_.h_bySectMeasLA_->setBinLabel(bin, binName);
    hists_.h_bySectSetLA_->setBinLabel(bin, binName);
    hists_.h_bySectRejectLA_->setBinLabel(bin, binName);
    hists_.h_bySectLA_->setBinLabel(bin, binName);
    hists_.h_bySectDeltaLA_->setBinLabel(bin, binName);
    hists_.h_bySectChi2_->setBinLabel(bin, binName);
  }

  // this will be booked in the Harvesting folder
  iBooker.setCurrentFolder(fmt::format("{}Harvesting/LorentzAngleMaps", dqmDir_));
  for (int i = 0; i < hists_.nlay; i++) {
    std::string repName = "h2_byLayerLA_%i";
    std::string repText = "BPix Layer %i tan#theta_{LA}/B;module number;ladder number;tan#theta_{LA}/B [1/T]";

    hists_.h2_byLayerLA_.emplace_back(iBooker.book2D(fmt::sprintf(repName, i + 1),
                                                     fmt::sprintf(repText, i + 1),
                                                     hists_.nModules_[i],
                                                     0.5,
                                                     hists_.nModules_[i] + 0.5,
                                                     hists_.nLadders_[i],
                                                     0.5,
                                                     hists_.nLadders_[i] + 0.5));

    repName = "h2_byLayerDiff_%i";
    repText = "BPix Layer %i #Delta#mu_{H}/#mu_{H};module number;ladder number;#Delta#mu_{H}/#mu_{H} [%%]";
    hists_.h2_byLayerDiff_.emplace_back(iBooker.book2D(fmt::sprintf(repName, i + 1),
                                                       fmt::sprintf(repText, i + 1),
                                                       hists_.nModules_[i],
                                                       0.5,
                                                       hists_.nModules_[i] + 0.5,
                                                       hists_.nLadders_[i],
                                                       0.5,
                                                       hists_.nLadders_[i] + 0.5));
  }

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
                                << "Error(LA)" ;
  // clang-format on

  // payload to be written out
  std::shared_ptr<SiPixelLorentzAngle> LorentzAngle = std::make_shared<SiPixelLorentzAngle>();

  // fill the map of simulation values
  double p1_simul_newmodule = 0.294044;
  double p1_simul[hists_.nlay + 1][hists_.nModules_[hists_.nlay - 1]];
  for (int i_layer = 1; i_layer <= hists_.nlay; i_layer++) {
    for (int i_module = 1; i_module <= hists_.nModules_[i_layer - 1]; i_module++) {
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
  // fictitious n-th layer to store the values of new modules
  for (int i_module = 1; i_module <= hists_.nModules_[hists_.nlay - 1]; i_module++) {
    p1_simul[hists_.nlay][i_module - 1] = p1_simul_newmodule;
  }

  // loop over "new" BPix modules
  for (int j = 0; j < (int)hists_.BPixnewDetIds_.size(); j++) {
    //uint32_t rawId = hists_.BPixnewDetIds_[j];
    int new_index = j + 1 + hists_.nModules_[hists_.nlay - 1] + (hists_.nlay - 1) * hists_.nModules_[hists_.nlay - 1];
    if (hists_.h_drift_depth_adc_[new_index] == nullptr)
      continue;
    for (int i = 1; i <= hist_depth_; i++) {
      findMean(h_drift_depth_adc_slice_, i, new_index);
    }

    // fit the distributions and store the LA in the payload
    const auto& res = fitAndStore(LorentzAngle, new_index, hists_.BPixnewLayer_[j], hists_.BPixnewModule_[j]);

    edm::LogPrint("SiPixelLorentzAngle") << std::setprecision(4) << hists_.BPixnewModule_[j] << "\t"
                                         << hists_.BPixnewLayer_[j] << "\t" << res.p0 << "\t" << res.e0 << "\t"
                                         << res.p1 << std::setprecision(3) << "\t" << res.e1 << "\t"
                                         << res.e1 / res.p1 * 100. << "\t"
                                         << (res.p1 - p1_simul[hists_.nlay][0]) / res.e1 << "\t" << res.p2 << "\t"
                                         << res.e2 << "\t" << res.p3 << "\t" << res.e3 << "\t" << res.p4 << "\t"
                                         << res.e4 << "\t" << res.p5 << "\t" << res.e5 << "\t" << res.chi2 << "\t"
                                         << res.prob << "\t" << hists_.BPixnewDetIds_[j] << "\t" << res.tan_LA << "\t"
                                         << res.error_LA;
  }  // loop on BPix new modules

  //loop over modules and layers to fit the lorentz angle
  for (int i_layer = 1; i_layer <= hists_.nlay; i_layer++) {
    for (int i_module = 1; i_module <= hists_.nModules_[i_layer - 1]; i_module++) {
      int i_index = i_module + (i_layer - 1) * hists_.nModules_[i_layer - 1];
      if (hists_.h_drift_depth_adc_[i_index] == nullptr)
        continue;
      //loop over bins in depth (z-local-coordinate) (in order to fit slices)
      for (int i = 1; i <= hist_depth_; i++) {
        findMean(h_drift_depth_adc_slice_, i, i_index);
      }  // end loop over bins in depth

      // fit the distributions and store the LA in the payload
      const auto& res = fitAndStore(LorentzAngle, i_index, i_layer, i_module);

      edm::LogPrint("SiPixelLorentzAngle")
          << std::setprecision(4) << i_module << "\t" << i_layer << "\t" << res.p0 << "\t" << res.e0 << "\t" << res.p1
          << std::setprecision(3) << "\t" << res.e1 << "\t" << res.e1 / res.p1 * 100. << "\t"
          << (res.p1 - p1_simul[i_layer - 1][i_module - 1]) / res.e1 << "\t" << res.p2 << "\t" << res.e2 << "\t"
          << res.p3 << "\t" << res.e3 << "\t" << res.p4 << "\t" << res.e4 << "\t" << res.p5 << "\t" << res.e5 << "\t"
          << res.chi2 << "\t" << res.prob << "\t"
          << "null"
          << "\t" << res.tan_LA << "\t" << res.error_LA;
    }
  }  // end loop over modules and layers

  // fill the rest of DetIds not filled above (for the moment FPix)
  const auto& currentLAMap = currentLorentzAngle_->getLorentzAngles();
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
    float fPixLorentzAnglePerTesla_ = currentLorentzAngle_->getLorentzAngle(id);
    if (!LorentzAngle->putLorentzAngle(id, fPixLorentzAnglePerTesla_)) {
      edm::LogError("SiPixelLorentzAnglePCLHarvester")
          << "[SiPixelLorentzAnglePCLHarvester::dqmEndJob] filling rest of payload: detid already exists";
    }
  }

  for (const auto& id : newLADets) {
    float deltaMuHoverMuH = (currentLorentzAngle_->getLorentzAngle(id) - LorentzAngle->getLorentzAngle(id)) /
                            currentLorentzAngle_->getLorentzAngle(id);
    h_diffLA->Fill(deltaMuHoverMuH * 100.f);
  }

  bool isPayloadChanged{false};
  // fill the 2D output Lorentz Angle maps and check if the payload is different from the input one
  for (const auto& [id, value] : LorentzAngle->getLorentzAngles()) {
    DetId ID = DetId(id);
    if (ID.subdetId() == PixelSubdetector::PixelBarrel) {
      const auto& layer = theTrackerTopology_->pxbLayer(id);
      const auto& ladder = theTrackerTopology_->pxbLadder(id);
      const auto& module = theTrackerTopology_->pxbModule(id);
      hists_.h2_byLayerLA_[layer - 1]->setBinContent(module, ladder, value);

      float deltaMuHoverMuH =
          (currentLorentzAngle_->getLorentzAngle(id) - value) / currentLorentzAngle_->getLorentzAngle(id);
      hists_.h2_byLayerDiff_[layer - 1]->setBinContent(module, ladder, deltaMuHoverMuH * 100.f);

      if (!isPayloadChanged && (deltaMuHoverMuH != 0.f))
        isPayloadChanged = true;
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
void SiPixelLorentzAnglePCLHarvester::findMean(MonitorElement* h_drift_depth_adc_slice_, int i, int i_ring) {
  double nentries = 0;
  h_drift_depth_adc_slice_->Reset();
  int hist_drift_ = h_drift_depth_adc_slice_->getNbinsX();

  // determine sigma and sigma^2 of the adc counts and average adc counts
  //loop over bins in drift width
  for (int j = 1; j <= hist_drift_; j++) {
    if (hists_.h_drift_depth_noadc_[i_ring]->getBinContent(j, i) >= 1) {
      double adc_error2 = (hists_.h_drift_depth_adc2_[i_ring]->getBinContent(j, i) -
                           hists_.h_drift_depth_adc_[i_ring]->getBinContent(j, i) *
                               hists_.h_drift_depth_adc_[i_ring]->getBinContent(j, i) /
                               hists_.h_drift_depth_noadc_[i_ring]->getBinContent(j, i)) /
                          hists_.h_drift_depth_noadc_[i_ring]->getBinContent(j, i);

      hists_.h_drift_depth_adc_[i_ring]->setBinError(j, i, sqrt(adc_error2));
      double error2 = adc_error2 / (hists_.h_drift_depth_noadc_[i_ring]->getBinContent(j, i) - 1.);
      hists_.h_drift_depth_[i_ring]->setBinError(j, i, sqrt(error2));
    } else {
      hists_.h_drift_depth_[i_ring]->setBinError(j, i, 0);
      hists_.h_drift_depth_adc_[i_ring]->setBinError(j, i, 0);
    }
    h_drift_depth_adc_slice_->setBinContent(j, hists_.h_drift_depth_adc_[i_ring]->getBinContent(j, i));
    h_drift_depth_adc_slice_->setBinError(j, hists_.h_drift_depth_adc_[i_ring]->getBinError(j, i));
    nentries += hists_.h_drift_depth_noadc_[i_ring]->getBinContent(j, i);
  }  // end loop over bins in drift width

  double mean = h_drift_depth_adc_slice_->getMean(1);
  double error = 0;
  if (nentries != 0) {
    error = h_drift_depth_adc_slice_->getRMS(1) / std::sqrt(nentries);
  }
  hists_.h_mean_[i_ring]->setBinContent(i, mean);
  hists_.h_mean_[i_ring]->setBinError(i, error);

  h_drift_depth_adc_slice_->Reset();  // clear again after extracting the parameters
}

//------------------------------------------------------------------------------
SiPixelLAHarvest::fitResults SiPixelLorentzAnglePCLHarvester::fitAndStore(
    std::shared_ptr<SiPixelLorentzAngle> theLAPayload, int i_index, int i_layer, int i_module) {
  // output results
  SiPixelLAHarvest::fitResults res;

  double half_width = width_ * 10000 / 2;  // pixel half thickness in units of micro meter

  f1_ = std::make_unique<TF1>("f1", "[0] + [1]*x + [2]*x*x + [3]*x*x*x + [4]*x*x*x*x + [5]*x*x*x*x*x", 5., 280.);
  f1_->SetParName(0, "offset");
  f1_->SetParName(1, "tan#theta_{LA}");
  f1_->SetParName(2, "quad term");
  f1_->SetParName(3, "cubic term");
  f1_->SetParName(4, "quartic term");
  f1_->SetParName(5, "quintic term");

  f1_->SetParameter(0, 0);
  f1_->SetParError(0, 0);
  f1_->SetParameter(1, 0.4);
  f1_->SetParError(1, 0);
  f1_->SetParameter(2, 0.0);
  f1_->SetParError(2, 0);
  f1_->SetParameter(3, 0.0);
  f1_->SetParError(3, 0);
  f1_->SetParameter(4, 0.0);
  f1_->SetParError(4, 0);
  f1_->SetParameter(5, 0.0);
  f1_->SetParError(5, 0);
  f1_->SetChisquare(0);

  hists_.h_mean_[i_index]->getTH1()->Fit(f1_.get(), "ERQ");

  res.p0 = f1_->GetParameter(0);
  res.e0 = f1_->GetParError(0);
  res.p1 = f1_->GetParameter(1);
  res.e1 = f1_->GetParError(1);
  res.p2 = f1_->GetParameter(2);
  res.e2 = f1_->GetParError(2);
  res.p3 = f1_->GetParameter(3);
  res.e3 = f1_->GetParError(3);
  res.p4 = f1_->GetParameter(4);
  res.e4 = f1_->GetParError(4);
  res.p5 = f1_->GetParameter(5);
  res.e5 = f1_->GetParError(5);
  res.chi2 = f1_->GetChisquare();
  res.ndf = f1_->GetNDF();
  res.prob = f1_->GetProb();
  res.redChi2 = res.ndf > 0. ? res.chi2 / res.ndf : 0.;

  double f1_halfwidth = res.p0 + res.p1 * half_width + res.p2 * pow(half_width, 2) + res.p3 * pow(half_width, 3) +
                        res.p4 * pow(half_width, 4) + res.p5 * pow(half_width, 5);

  double f1_zerowidth = res.p0;

  // tan_LA = (f1(x = half_width) - f1(x = 0)) / (half_width - 0)
  res.tan_LA = (f1_halfwidth - f1_zerowidth) / half_width;
  double errsq_LA =
      (pow(res.e1, 2) + pow((half_width * res.e2), 2) + pow((half_width * half_width * res.e3), 2) +
       pow((half_width * half_width * half_width * res.e4), 2) +
       pow((half_width * half_width * half_width * half_width * res.e5), 2));  // Propagation of uncertainty
  res.error_LA = sqrt(errsq_LA);

  hists_.h_bySectMeasLA_->setBinContent(i_index, (res.tan_LA / theMagField_));
  hists_.h_bySectMeasLA_->setBinError(i_index, (res.error_LA / theMagField_));
  hists_.h_bySectChi2_->setBinContent(i_index, res.redChi2);
  hists_.h_bySectChi2_->setBinError(i_index, 0.);  // no errors

  int nentries = hists_.h_bySectOccupancy_->getBinContent(i_index);  // number of on track hits in that sector

  bool isNew = (i_index > hists_.nlay * hists_.nModules_[hists_.nlay - 1]);
  int shiftIdx = i_index - hists_.nlay * hists_.nModules_[hists_.nlay - 1] - 1;

  LogDebug("SiPixelLorentzAnglePCLHarvester")
      << " isNew: " << isNew << " i_index: " << i_index << " shift index: " << shiftIdx;

  const auto& detIdsToFill =
      isNew ? std::vector<unsigned int>({hists_.BPixnewDetIds_[shiftIdx]}) : hists_.detIdsList[i_index];

  LogDebug("SiPixelLorentzAnglePCLHarvester")
      << "index: " << i_index << " i_module: " << i_module << " i_layer: " << i_layer;
  for (const auto& id : detIdsToFill) {
    LogDebug("SiPixelLorentzAnglePCLHarvester") << id << ",";
  }

  // no errors on the following MEs
  hists_.h_bySectSetLA_->setBinError(i_index, 0.);
  hists_.h_bySectRejectLA_->setBinError(i_index, 0.);
  hists_.h_bySectLA_->setBinError(i_index, 0.);
  hists_.h_bySectDeltaLA_->setBinError(i_index, 0.);

  float LorentzAnglePerTesla_;
  float currentLA = currentLorentzAngle_->getLorentzAngle(detIdsToFill.front());
  // if the fit quality is OK
  if ((res.redChi2 != 0.) && (res.redChi2 < fitChi2Cut_) && (nentries > minHitsCut_)) {
    LorentzAnglePerTesla_ = res.tan_LA / theMagField_;
    // fill the LA actually written to payload
    hists_.h_bySectSetLA_->setBinContent(i_index, LorentzAnglePerTesla_);
    hists_.h_bySectRejectLA_->setBinContent(i_index, 0.);
    hists_.h_bySectLA_->setBinContent(i_index, LorentzAnglePerTesla_);

    const auto& deltaLA = (LorentzAnglePerTesla_ - currentLA);
    hists_.h_bySectDeltaLA_->setBinContent(i_index, deltaLA);

    for (const auto& id : detIdsToFill) {
      if (!theLAPayload->putLorentzAngle(id, LorentzAnglePerTesla_)) {
        edm::LogError("SiPixelLorentzAnglePCLHarvester") << "[SiPixelLorentzAnglePCLHarvester::fitAndStore]: detid ("
                                                         << i_layer << "," << i_module << ") already exists";
      }
    }
  } else {
    // just copy the values from the existing payload
    hists_.h_bySectSetLA_->setBinContent(i_index, 0.);
    hists_.h_bySectRejectLA_->setBinContent(i_index, (res.tan_LA / theMagField_));
    hists_.h_bySectLA_->setBinContent(i_index, currentLA);
    hists_.h_bySectDeltaLA_->setBinContent(i_index, 0.);

    for (const auto& id : detIdsToFill) {
      LorentzAnglePerTesla_ = currentLorentzAngle_->getLorentzAngle(id);
      if (!theLAPayload->putLorentzAngle(id, LorentzAnglePerTesla_)) {
        edm::LogError("SiPixelLorentzAnglePCLHarvester") << "[SiPixelLorentzAnglePCLHarvester::fitAndStore]: detid ("
                                                         << i_layer << "," << i_module << ") already exists";
      }
    }
  }
  // return the struct of fit details
  return res;
}

//------------------------------------------------------------------------------
void SiPixelLorentzAnglePCLHarvester::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setComment("Harvester module of the SiPixel Lorentz Angle PCL monitoring workflow");
  desc.add<std::vector<std::string>>("newmodulelist", {})->setComment("the list of DetIds for new sensors");
  desc.add<std::string>("dqmDir", "AlCaReco/SiPixelLorentzAngle")->setComment("the directory of PCL Worker output");
  desc.add<double>("fitChi2Cut", 20.)->setComment("cut on fit chi2/ndof to accept measurement");
  desc.add<int>("minHitsCut", 10000)->setComment("cut on minimum number of on-track hits to accept measurement");
  desc.add<std::string>("record", "SiPixelLorentzAngleRcd")->setComment("target DB record");
  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(SiPixelLorentzAnglePCLHarvester);
