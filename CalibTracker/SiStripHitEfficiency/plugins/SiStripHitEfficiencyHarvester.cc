#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"

#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "DQM/SiStripCommon/interface/TkHistoMap.h"

#include "CalibTracker/Records/interface/SiStripQualityRcd.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

// FIXME TEMPORARY
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"

class SiStripHitEfficiencyHarvester : public DQMEDHarvester {
public:
  explicit SiStripHitEfficiencyHarvester(const edm::ParameterSet&);
  ~SiStripHitEfficiencyHarvester() override = default;

  void endRun(edm::Run const&, edm::EventSetup const&) override;
  void dqmEndJob(DQMStore::IBooker&, DQMStore::IGetter&) override;

private:
  bool showRings_, autoIneffModTagging_;
  unsigned int nTEClayers_;
  std::string layerName(unsigned int k) const;
  double threshold_;
  int nModsMin_;
  double tkMapMin_;
  std::string title_;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tTopoToken_;
  std::unique_ptr<TrackerTopology> tTopo_;
  edm::ESGetToken<TkDetMap, TrackerTopologyRcd> tkDetMapToken_;
  std::unique_ptr<TkDetMap> tkDetMap_;
  edm::ESGetToken<SiStripQuality, SiStripQualityRcd> stripQualityToken_;
  std::unique_ptr<SiStripQuality> stripQuality_;
};

SiStripHitEfficiencyHarvester::SiStripHitEfficiencyHarvester(const edm::ParameterSet& conf) {
  showRings_ = conf.getUntrackedParameter<bool>("ShowRings", false);
  nTEClayers_ = (showRings_ ? 7 : 9);  // number of rings or wheels
  threshold_ = conf.getParameter<double>("Threshold");
  nModsMin_ = conf.getParameter<int>("nModsMin");
  tkMapMin_ = conf.getUntrackedParameter<double>("TkMapMin", 0.9);
  title_ = conf.getParameter<std::string>("Title");
  autoIneffModTagging_ = conf.getUntrackedParameter<bool>("AutoIneffModTagging", false);
  tTopoToken_ = esConsumes<edm::Transition::EndRun>();
  tkDetMapToken_ = esConsumes<edm::Transition::EndRun>();
  stripQualityToken_ = esConsumes<edm::Transition::EndRun>();
}

void SiStripHitEfficiencyHarvester::endRun(edm::Run const&, edm::EventSetup const& iSetup) {
  if (!tTopo_) {
    tTopo_ = std::make_unique<TrackerTopology>(iSetup.getData(tTopoToken_));
  }
  if (!tkDetMap_) {
    tkDetMap_ = std::make_unique<TkDetMap>(iSetup.getData(tkDetMapToken_));
  }
  if (!stripQuality_) {
    stripQuality_ = std::make_unique<SiStripQuality>(iSetup.getData(stripQualityToken_));
  }
}

namespace {
  unsigned int checkLayer(unsigned int iidd, const TrackerTopology* tTopo) {
    switch (DetId(iidd).subdetId()) {
      case SiStripSubdetector::TIB:
        return tTopo->tibLayer(iidd);
      case SiStripSubdetector::TOB:
        return tTopo->tobLayer(iidd) + 4;
      case SiStripSubdetector::TID:
        return tTopo->tidWheel(iidd) + 10;
      case SiStripSubdetector::TEC:
        return tTopo->tecWheel(iidd) + 13;
      default:
        return 0;
    }
  }
}  // namespace

std::string SiStripHitEfficiencyHarvester::layerName(unsigned int k) const {
  const std::string ringlabel{showRings_ ? "R" : "D"};
  if (k > 0 && k < 5) {
    return fmt::format("TIB L{:d}", k);
  } else if (k > 4 && k < 11) {
    return fmt::format("TOB L{:d}", k - 4);
  } else if (k > 10 && k < 14) {
    return fmt::format("TIB {0}{1:d}", ringlabel, k - 10);
  } else if (k > 13 && k < 14 + nTEClayers_) {
    return fmt::format("TEC {0}{1:d}", ringlabel, k - 13);
  } else {
    return "";
  }
}

void SiStripHitEfficiencyHarvester::dqmEndJob(DQMStore::IBooker& booker, DQMStore::IGetter& getter) {
  unsigned int nLayers = 22;
  if (showRings_)
    nLayers = 20;

  // TODO move down, in makeSummaryVsLumi
  for (unsigned int iLayer = 1; iLayer != nLayers; ++iLayer) {
    auto hfound = getter.get(fmt::format("SiStrip/HitEfficiency/layerfound_vsLumi_layer_{}", iLayer))->getTH1F();
    auto htotal = getter.get(fmt::format("SiStrip/HitEfficiency/layertotal_vsLumi_layer_{}", iLayer))->getTH1F();
    if (!hfound->GetSumw2())
      hfound->Sumw2();
    if (!htotal->GetSumw2())
      htotal->Sumw2();
    for (Long_t i = 0; i != hfound->GetNbinsX() + 1; ++i) {
      if (hfound->GetBinContent(i) == 0)
        hfound->SetBinContent(i, 1e-6);
      if (htotal->GetBinContent(i) == 0)
        htotal->SetBinContent(i, 1);
    }
    LogDebug("SiStripHitEfficiency:HitEff")
        << "Total hits for layer " << iLayer << " (vs lumi): " << htotal->GetEntries() << ", found "
        << hfound->GetEntries();
  }

  // TODO list
  // - makeHotColdMaps: should be done directly by worker
  // - makeTKMap: below here
  // - makeSQLite: TODO
  // - totalStatistics: TODO
  // - makeSummary: TODO
  // - makeSummaryVsBX: TODO
  // - makeSummaryVsLumi: TODO
  // - makeSummaryVsCM: TODO
  // - rest: TODO

  auto h_module_total = std::make_unique<TkHistoMap>(tkDetMap_.get());
  h_module_total->loadTkHistoMap("SiStrip/HitEfficiency", "perModule_total");
  auto h_module_found = std::make_unique<TkHistoMap>(tkDetMap_.get());
  h_module_found->loadTkHistoMap("SiStrip/HitEfficiency", "perModule_found");
  LogDebug("SiStripHitEfficiency:HitEff")
      << "Entries in total TkHistoMap for layer 3: " << h_module_total->getMap(3)->getEntries() << ", found "
      << h_module_found->getMap(3)->getEntries();

  edm::Service<TFileService> fs;
  std::vector<TH1F*> hEffInLayer(std::size_t(1), nullptr);
  hEffInLayer.reserve(23);
  for (std::size_t i = 1; i != 23; ++i) {
    hEffInLayer.push_back(
        fs->make<TH1F>(Form("eff_layer%i", int(i)), Form("Module efficiency in layer %i", int(i)), 201, 0, 1.005));
  }
  std::array<long, 23> layertotal{};
  std::array<long, 23> layerfound{};
  layertotal.fill(0);
  layerfound.fill(0);

  //////////////////
  // Tracker maps //
  //////////////////

  TrackerMap tkMap{"  Detector Inefficiency  "};
  TrackerMap tkMapBad{"  Inefficient Modules  "};
  TrackerMap tkMapEff{title_};
  TrackerMap tkMapNum{" Detector numerator   "};
  TrackerMap tkMapDen{" Detector denominator "};
  std::map<unsigned int, double> badModules;

  SiStripDetInfoFileReader reader{
      edm::FileInPath{"CalibTracker/SiStripCommon/data/SiStripDetInfo.dat"}
          .fullPath()};  // FIXME update to new version, if needed at all (should also be possible to get from quality ?)
  for (auto det : reader.getAllDetIds()) {
    // TODO probably need a way to exclude bad modules (cache sistripquality and federrids, or list of bad module DetIds directly - may be easier)

    auto layer = checkLayer(det, tTopo_.get());
    const auto num = h_module_found->getValue(det);
    const auto denom = h_module_total->getValue(det);
    const auto eff = denom > 0 ? num / denom : 0.;
    hEffInLayer[layer]->Fill(eff);
    if (!autoIneffModTagging_) {
      if ((denom >= nModsMin_) && (eff < threshold_)) {
        // We have a bad module, put it in the list!
        badModules[det] = eff;
        tkMapBad.fillc(det, 255, 0, 0);
        std::cout << "Layer " << layer << " (" << layerName(layer) << ")  module " << det << " efficiency: " << eff
                  << " , " << num << "/" << denom << std::endl;
      } else {
        //Fill the bad list with empty results for every module
        tkMapBad.fillc(det, 255, 255, 255);
      }
      if (denom && (denom < nModsMin_)) {
        std::cout << "Layer " << layer << " (" << layerName(layer) << ")  module " << det << " is under occupancy at "
                  << denom << std::endl;
      }
    }
    if (denom) {
      //Put any module into the TKMap
      tkMap.fill(det, 1. - eff);
      tkMapEff.fill(det, eff);
      tkMapNum.fill(det, num);
      tkMapDen.fill(det, denom);

      layertotal[layer] += denom;
      layerfound[layer] += num;
    }
  }

  if (autoIneffModTagging_) {
    for (Long_t i = 1; i <= 22; i++) {
      //Compute threshold to use for each layer
      hEffInLayer[i]->GetXaxis()->SetRange(3, hEffInLayer[i]->GetNbinsX() + 1);  // Remove from the avg modules below 1%
      const double eff_limit = hEffInLayer[i]->GetMean() - threshold_;
      std::cout << "Layer " << i << " threshold for bad modules: " << eff_limit << std::endl;
      hEffInLayer[i]->GetXaxis()->SetRange(1, hEffInLayer[i]->GetNbinsX() + 1);

      for (auto det : reader.getAllDetIds()) {
        // Second loop over modules to tag inefficient ones
        const auto layer = checkLayer(det, tTopo_.get());
        if (layer == i) {
          const auto num = h_module_found->getValue(det);
          const auto denom = h_module_total->getValue(det);
          const auto eff = denom > 0 ? num / denom : 0.;

          if ((denom >= nModsMin_) && (eff < eff_limit)) {
            //We have a bad module, put it in the list!
            badModules[det] = eff;
            tkMapBad.fillc(det, 255, 0, 0);
            std::cout << "Layer " << layer << " (" << layerName(layer) << ")  module " << det << " efficiency: " << eff
                      << " , " << num << "/" << denom << std::endl;
          } else {
            //Fill the bad list with empty results for every module
            tkMapBad.fillc(det, 255, 255, 255);
          }
          if (denom && (denom < nModsMin_)) {
            std::cout << "Layer " << layer << " (" << layerName(layer) << ")  module " << det << " layer " << layer
                      << " is under occupancy at " << denom << std::endl;
          }
        }
      }
    }
  }

  tkMap.save(true, 0, 0, "SiStripHitEffTKMap_NEW.png");
  tkMapBad.save(true, 0, 0, "SiStripHitEffTKMapBad_NEW.png");
  tkMapEff.save(true, tkMapMin_, 1., "SiStripHitEffTKMapEff_NEW.png");
  tkMapNum.save(true, 0, 0, "SiStripHitEffTKMapNum_NEW.png");
  tkMapDen.save(true, 0, 0, "SiStripHitEffTKMapDen_NEW.png");
  std::cout << "Finished TKMap Generation" << std::endl;
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SiStripHitEfficiencyHarvester);
