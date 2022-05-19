// user includes
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"
#include "CalibTracker/Records/interface/SiStripQualityRcd.h"
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"
#include "CalibTracker/SiStripHitEfficiency/interface/SiStripHitEfficiencyHelpers.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "DataFormats/SiStripCommon/interface/ConstantsForHardwareSystems.h" /* for STRIPS_PER_APV*/
#include "DQM/SiStripCommon/interface/TkHistoMap.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

//system includes
#include <sstream>

// ROOT includes
#include "TEfficiency.h"

// custom made printout
#define LOGPRINT edm::LogPrint("SiStripHitEfficiencyHarvester")

class SiStripHitEfficiencyHarvester : public DQMEDHarvester {
public:
  explicit SiStripHitEfficiencyHarvester(const edm::ParameterSet&);
  ~SiStripHitEfficiencyHarvester() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void endRun(edm::Run const&, edm::EventSetup const&) override;
  void dqmEndJob(DQMStore::IBooker&, DQMStore::IGetter&) override;

private:
  const bool isAtPCL_;
  const bool showRings_, autoIneffModTagging_, doStoreOnDB_;
  const unsigned int nTEClayers_;
  const double threshold_;
  const int nModsMin_;
  const double tkMapMin_;
  const std::string title_, record_;

  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tTopoToken_;
  const edm::ESGetToken<TkDetMap, TrackerTopologyRcd> tkDetMapToken_;
  const edm::ESGetToken<SiStripQuality, SiStripQualityRcd> stripQualityToken_;
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> tkGeomToken_;

  std::unique_ptr<TrackerTopology> tTopo_;
  std::unique_ptr<TkDetMap> tkDetMap_;
  std::unique_ptr<SiStripQuality> stripQuality_;
  std::vector<DetId> stripDetIds_;

  void writeBadStripPayload(const SiStripQuality& quality) const;
  void printTotalStatistics(const std::array<long, 23>& layerFound, const std::array<long, 23>& layerTotal) const;
  void printAndWriteBadModules(const SiStripQuality& quality, const SiStripDetInfo& detInfo) const;
  bool checkMapsValidity(const std::vector<MonitorElement*>& maps, const std::string& type) const;
  void makeSummary(DQMStore::IGetter& getter, TFileService& fs) const;
  void makeSummaryVsBX(DQMStore::IGetter& getter, TFileService& fs) const;
  void makeSummaryVsLumi(DQMStore::IGetter& getter) const;
  void makeSummaryVsCM(DQMStore::IGetter& getter, TFileService& fs) const;
};

SiStripHitEfficiencyHarvester::SiStripHitEfficiencyHarvester(const edm::ParameterSet& conf)
    : isAtPCL_(conf.getParameter<bool>("isAtPCL")),
      showRings_(conf.getUntrackedParameter<bool>("ShowRings", false)),
      autoIneffModTagging_(conf.getUntrackedParameter<bool>("AutoIneffModTagging", false)),
      doStoreOnDB_(conf.getParameter<bool>("doStoreOnDB")),
      nTEClayers_(showRings_ ? 7 : 9),  // number of rings or wheels
      threshold_(conf.getParameter<double>("Threshold")),
      nModsMin_(conf.getParameter<int>("nModsMin")),
      tkMapMin_(conf.getUntrackedParameter<double>("TkMapMin", 0.9)),
      title_(conf.getParameter<std::string>("Title")),
      record_(conf.getParameter<std::string>("Record")),
      tTopoToken_(esConsumes<edm::Transition::EndRun>()),
      tkDetMapToken_(esConsumes<edm::Transition::EndRun>()),
      stripQualityToken_(esConsumes<edm::Transition::EndRun>()),
      tkGeomToken_(esConsumes<edm::Transition::EndRun>()) {}

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
  if (stripDetIds_.empty()) {
    const auto& tkGeom = iSetup.getData(tkGeomToken_);
    for (const auto& det : tkGeom.detUnits()) {
      if (dynamic_cast<const StripGeomDetUnit*>(det)) {
        stripDetIds_.push_back(det->geographicalId());
      }
    }
  }
}

bool SiStripHitEfficiencyHarvester::checkMapsValidity(const std::vector<MonitorElement*>& maps,
                                                      const std::string& type) const {
  std::vector<bool> isAvailable;
  isAvailable.reserve(maps.size());
  std::transform(
      maps.begin() + 1, maps.end(), std::back_inserter(isAvailable), [](auto& x) { return !(x == nullptr); });

  int count{0};
  for (const auto& it : isAvailable) {
    count++;
    LogDebug("SiStripHitEfficiencyHarvester") << " layer: " << count << " " << it << std::endl;
    if (it)
      LogDebug("SiStripHitEfficiencyHarvester") << "resolving to " << maps[count]->getName() << std::endl;
  }

  // check on the input TkHistoMap
  bool areMapsAvailable{true};
  int layerCount{0};
  for (const auto& it : isAvailable) {
    layerCount++;
    if (!it) {
      edm::LogError("SiStripHitEfficiencyHarvester")
          << type << " TkHistoMap for layer " << layerCount << " was not found.\n -> Aborting!";
      areMapsAvailable = false;
      break;
    }
  }
  return areMapsAvailable;
}

void SiStripHitEfficiencyHarvester::dqmEndJob(DQMStore::IBooker& booker, DQMStore::IGetter& getter) {
  if (!autoIneffModTagging_)
    LOGPRINT << "A module is bad if efficiency < " << threshold_ << " and has at least " << nModsMin_ << " nModsMin.";
  else
    LOGPRINT << "A module is bad if the upper limit on the efficiency is < to the avg in the layer - " << threshold_
             << " and has at least " << nModsMin_ << " nModsMin.";

  auto h_module_total = std::make_unique<TkHistoMap>(tkDetMap_.get());
  h_module_total->loadTkHistoMap("AlCaReco/SiStripHitEfficiency", "perModule_total");
  auto h_module_found = std::make_unique<TkHistoMap>(tkDetMap_.get());
  h_module_found->loadTkHistoMap("AlCaReco/SiStripHitEfficiency", "perModule_found");

  // collect how many layers are missing
  const auto& totalMaps = h_module_total->getAllMaps();
  const auto& foundMaps = h_module_found->getAllMaps();

  LogDebug("SiStripHitEfficiencyHarvester")
      << "totalMaps.size(): " << totalMaps.size() << " foundMaps.size() " << foundMaps.size() << std::endl;

  // check on the input TkHistoMaps
  bool isTotalMapAvailable = this->checkMapsValidity(totalMaps, std::string("Total"));
  bool isFoundMapAvailable = this->checkMapsValidity(foundMaps, std::string("Found"));

  LogDebug("SiStripHitEfficiencyHarvester")
      << "isTotalMapAvailable: " << isTotalMapAvailable << " isFoundMapAvailable " << isFoundMapAvailable << std::endl;

  // no input TkHistoMaps -> early return
  if (!isTotalMapAvailable or !isFoundMapAvailable)
    return;

  LogDebug("SiStripHitEfficiencyHarvester")
      << "Entries in total TkHistoMap for layer 3: " << h_module_total->getMap(3)->getEntries() << ", found "
      << h_module_found->getMap(3)->getEntries();

  std::vector<MonitorElement*> hEffInLayer(std::size_t(1), nullptr);
  hEffInLayer.reserve(23);
  for (std::size_t i = 1; i != 23; ++i) {
    hEffInLayer.push_back(
        booker.book1D(Form("eff_layer%i", int(i)), Form("Module efficiency in layer %i", int(i)), 201, 0, 1.005));
  }
  std::array<long, 23> layerTotal{};
  std::array<long, 23> layerFound{};
  layerTotal.fill(0);
  layerFound.fill(0);

  ////////////////////////////////////////////////////////////////
  // Effiency calculation, bad module tagging, and tracker maps //
  ////////////////////////////////////////////////////////////////

  TrackerMap tkMap{"  Detector Inefficiency  "};
  TrackerMap tkMapBad{"  Inefficient Modules  "};
  TrackerMap tkMapEff{title_};
  TrackerMap tkMapNum{" Detector numerator   "};
  TrackerMap tkMapDen{" Detector denominator "};
  std::map<unsigned int, double> badModules;

  for (auto det : stripDetIds_) {
    auto layer = ::checkLayer(det, tTopo_.get());
    const auto num = h_module_found->getValue(det);
    const auto denom = h_module_total->getValue(det);
    if (denom) {
      const auto eff = num / denom;
      hEffInLayer[layer]->Fill(eff);
      if (!autoIneffModTagging_) {
        if ((denom >= nModsMin_) && (eff < threshold_)) {
          // We have a bad module, put it in the list!
          badModules[det] = eff;
          tkMapBad.fillc(det, 255, 0, 0);
          LOGPRINT << "Layer " << layer << " (" << ::layerName(layer, showRings_, nTEClayers_) << ")  module "
                   << det.rawId() << " efficiency: " << eff << " , " << num << "/" << denom;
        } else {
          //Fill the bad list with empty results for every module
          tkMapBad.fillc(det, 255, 255, 255);
        }
        if (eff < threshold_)
          LOGPRINT << "Layer " << layer << " (" << ::layerName(layer, showRings_, nTEClayers_) << ")  module "
                   << det.rawId() << " efficiency: " << eff << " , " << num << "/" << denom;

        if (denom < nModsMin_) {
          LOGPRINT << "Layer " << layer << " (" << ::layerName(layer, showRings_, nTEClayers_) << ")  module "
                   << det.rawId() << " is under occupancy at " << denom;
        }
      }
      //Put any module into the TKMap
      tkMap.fill(det, 1. - eff);
      tkMapEff.fill(det, eff);
      tkMapNum.fill(det, num);
      tkMapDen.fill(det, denom);

      layerTotal[layer] += denom;
      layerFound[layer] += num;
    }
  }

  if (autoIneffModTagging_) {
    for (Long_t i = 1; i <= 22; i++) {
      //Compute threshold to use for each layer
      hEffInLayer[i]->getTH1()->GetXaxis()->SetRange(
          3, hEffInLayer[i]->getNbinsX() + 1);  // Remove from the avg modules below 1%
      const double layer_min_eff = hEffInLayer[i]->getMean() - std::max(2.5 * hEffInLayer[i]->getRMS(), threshold_);
      LOGPRINT << "Layer " << i << " threshold for bad modules: <" << layer_min_eff
               << "  (layer mean: " << hEffInLayer[i]->getMean() << " rms: " << hEffInLayer[i]->getRMS() << ")";

      hEffInLayer[i]->getTH1()->GetXaxis()->SetRange(1, hEffInLayer[i]->getNbinsX() + 1);

      for (auto det : stripDetIds_) {
        const auto layer = ::checkLayer(det, tTopo_.get());
        if (layer == i) {
          const auto num = h_module_found->getValue(det);
          const auto denom = h_module_total->getValue(det);
          if (denom) {
            const auto eff = num / denom;
            const auto eff_up = TEfficiency::Bayesian(denom, num, .99, 1, 1, true);

            if ((denom >= nModsMin_) && (eff_up < layer_min_eff)) {
              //We have a bad module, put it in the list!
              badModules[det] = eff;
              tkMapBad.fillc(det, 255, 0, 0);
            } else {
              //Fill the bad list with empty results for every module
              tkMapBad.fillc(det, 255, 255, 255);
            }
            if (eff_up < layer_min_eff + 0.08)  // printing message also for modules sligthly above (8%) the limit

              LOGPRINT << "Layer " << layer << " (" << ::layerName(layer, showRings_, nTEClayers_) << ")  module "
                       << det.rawId() << " efficiency: " << eff << " , " << num << "/" << denom
                       << " , upper limit: " << eff_up;
            if (denom < nModsMin_) {
              LOGPRINT << "Layer " << layer << " (" << ::layerName(layer, showRings_, nTEClayers_) << ")  module "
                       << det.rawId() << " layer " << layer << " is under occupancy at " << denom;
            }
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

  const auto detInfo =
      SiStripDetInfoFileReader::read(edm::FileInPath{SiStripDetInfoFileReader::kDefaultFile}.fullPath());
  SiStripQuality pQuality{detInfo};
  //This is the list of the bad strips, use to mask out entire APVs
  //Now simply go through the bad hit list and mask out things that
  //are bad!
  for (const auto it : badModules) {
    const auto det = it.first;
    std::vector<unsigned int> badStripList;
    //We need to figure out how many strips are in this particular module
    //To Mask correctly!
    const auto nStrips = detInfo.getNumberOfApvsAndStripLength(det).first * sistrip::STRIPS_PER_APV;
    LOGPRINT << "Number of strips module " << det << " is " << nStrips;
    badStripList.push_back(pQuality.encode(0, nStrips, 0));
    //Now compact into a single bad module
    LOGPRINT << "ID1 shoudl match list of modules above " << det;
    pQuality.compact(det, badStripList);
    pQuality.put(det, SiStripQuality::Range(badStripList.begin(), badStripList.end()));
  }
  pQuality.fillBadComponents();
  if (doStoreOnDB_) {
    writeBadStripPayload(pQuality);
  } else {
    edm::LogInfo("SiStripHitEfficiencyHarvester") << "Will not produce payload!";
  }

  printTotalStatistics(layerFound, layerTotal);  // statistics by layer and subdetector
  //LOGPRINT << "\n-----------------\nNew IOV starting from run " << e.id().run() << " event " << e.id().event()
  //     << " lumiBlock " << e.luminosityBlock() << " time " << e.time().value() << "\n-----------------\n";
  printAndWriteBadModules(pQuality, detInfo);  // TODO

  if (!isAtPCL_) {
    edm::Service<TFileService> fs;
    makeSummary(getter, *fs);      // TODO
    makeSummaryVsBX(getter, *fs);  // TODO
    makeSummaryVsCM(getter, *fs);  // TODO
  }

  makeSummaryVsLumi(getter);  // TODO
}

void SiStripHitEfficiencyHarvester::printTotalStatistics(const std::array<long, 23>& layerFound,
                                                         const std::array<long, 23>& layerTotal) const {
  //Calculate the statistics by layer
  int totalfound = 0;
  int totaltotal = 0;
  double layereff;
  int subdetfound[5];
  int subdettotal[5];

  for (Long_t i = 1; i < 5; i++) {
    subdetfound[i] = 0;
    subdettotal[i] = 0;
  }

  for (Long_t i = 1; i <= 22; i++) {
    layereff = double(layerFound[i]) / double(layerTotal[i]);
    LOGPRINT << "Layer " << i << " (" << ::layerName(i, showRings_, nTEClayers_) << ") has total efficiency "
             << layereff << " " << layerFound[i] << "/" << layerTotal[i];
    totalfound += layerFound[i];
    totaltotal += layerTotal[i];
    if (i < 5) {
      subdetfound[1] += layerFound[i];
      subdettotal[1] += layerTotal[i];
    }
    if (i >= 5 && i < 11) {
      subdetfound[2] += layerFound[i];
      subdettotal[2] += layerTotal[i];
    }
    if (i >= 11 && i < 14) {
      subdetfound[3] += layerFound[i];
      subdettotal[3] += layerTotal[i];
    }
    if (i >= 14) {
      subdetfound[4] += layerFound[i];
      subdettotal[4] += layerTotal[i];
    }
  }

  LOGPRINT << "The total efficiency is " << double(totalfound) / double(totaltotal);
  LOGPRINT << "      TIB: " << double(subdetfound[1]) / subdettotal[1] << " " << subdetfound[1] << "/"
           << subdettotal[1];
  LOGPRINT << "      TOB: " << double(subdetfound[2]) / subdettotal[2] << " " << subdetfound[2] << "/"
           << subdettotal[2];
  LOGPRINT << "      TID: " << double(subdetfound[3]) / subdettotal[3] << " " << subdetfound[3] << "/"
           << subdettotal[3];
  LOGPRINT << "      TEC: " << double(subdetfound[4]) / subdettotal[4] << " " << subdetfound[4] << "/"
           << subdettotal[4];
}

void SiStripHitEfficiencyHarvester::writeBadStripPayload(const SiStripQuality& quality) const {
  SiStripBadStrip pBadStrip{};
  const auto pQdvBegin = quality.getDataVectorBegin();
  for (auto rIt = quality.getRegistryVectorBegin(); rIt != quality.getRegistryVectorEnd(); ++rIt) {
    const auto range = SiStripBadStrip::Range(pQdvBegin + rIt->ibegin, pQdvBegin + rIt->iend);
    if (!pBadStrip.put(rIt->detid, range))
      edm::LogError("SiStripHitEfficiencyHarvester") << "detid already exists in SiStripBadStrip";
  }
  edm::Service<cond::service::PoolDBOutputService> poolDbService;
  if (poolDbService.isAvailable()) {
    poolDbService->writeOneIOV(pBadStrip, poolDbService->currentTime(), record_);
  } else {
    throw cms::Exception("PoolDBService required");
  }
}

void SiStripHitEfficiencyHarvester::makeSummary(DQMStore::IGetter& getter, TFileService& fs) const {
  // use goodlayer_total/found and alllayer_total/found, collapse side and/or ring if needed
}

void SiStripHitEfficiencyHarvester::makeSummaryVsBX(DQMStore::IGetter& getter, TFileService& fs) const {
  // use found/totalVsBx_layer%i [0,23)
}

void SiStripHitEfficiencyHarvester::makeSummaryVsLumi(DQMStore::IGetter& getter) const {
  for (unsigned int iLayer = 1; iLayer != (showRings_ ? 20 : 22); ++iLayer) {
    auto hfound =
        getter.get(fmt::format("AlCaReco/SiStripHitEfficiency/layerfound_vsLumi_layer_{}", iLayer))->getTH1F();
    auto htotal =
        getter.get(fmt::format("AlCaReco/SiStripHitEfficiency/layertotal_vsLumi_layer_{}", iLayer))->getTH1F();

    if (hfound == nullptr or htotal == nullptr) {
      if (hfound == nullptr)
        edm::LogError("SiStripHitEfficiencyHarvester")
            << fmt::format("AlCaReco/SiStripHitEfficiency/layerfound_vsLumi_layer_{}", iLayer) << " was not found!";
      if (htotal == nullptr)
        edm::LogError("SiStripHitEfficiencyHarvester")
            << fmt::format("AlCaReco/SiStripHitEfficiency/layertotal_vsLumi_layer_{}", iLayer) << " was not found!";
      // no input histograms -> continue in the loop
      continue;
    }

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
    LogDebug("SiStripHitEfficiencyHarvester")
        << "Total hits for layer " << iLayer << " (vs lumi): " << htotal->GetEntries() << ", found "
        << hfound->GetEntries();
  }
  // continue
}

void SiStripHitEfficiencyHarvester::makeSummaryVsCM(DQMStore::IGetter& getter, TFileService& fs) const {}

namespace {
  void setBadComponents(int i,
                        int comp,
                        const SiStripQuality::BadComponent& bc,
                        std::stringstream ssV[4][19],
                        int nBad[4][19][4],
                        int nAPV) {
    ssV[i][comp] << "\n\t\t " << bc.detid << " \t " << bc.BadModule << " \t " << ((bc.BadFibers) & 0x1) << " ";
    if (nAPV == 4)
      ssV[i][comp] << "x " << ((bc.BadFibers >> 1) & 0x1);

    if (nAPV == 6)
      ssV[i][comp] << ((bc.BadFibers >> 1) & 0x1) << " " << ((bc.BadFibers >> 2) & 0x1);
    ssV[i][comp] << " \t " << ((bc.BadApvs) & 0x1) << " " << ((bc.BadApvs >> 1) & 0x1) << " ";
    if (nAPV == 4)
      ssV[i][comp] << "x x " << ((bc.BadApvs >> 2) & 0x1) << " " << ((bc.BadApvs >> 3) & 0x1);
    if (nAPV == 6)
      ssV[i][comp] << ((bc.BadApvs >> 2) & 0x1) << " " << ((bc.BadApvs >> 3) & 0x1) << " " << ((bc.BadApvs >> 4) & 0x1)
                   << " " << ((bc.BadApvs >> 5) & 0x1) << " ";

    if (bc.BadApvs) {
      nBad[i][0][2] += ((bc.BadApvs >> 5) & 0x1) + ((bc.BadApvs >> 4) & 0x1) + ((bc.BadApvs >> 3) & 0x1) +
                       ((bc.BadApvs >> 2) & 0x1) + ((bc.BadApvs >> 1) & 0x1) + ((bc.BadApvs) & 0x1);
      nBad[i][comp][2] += ((bc.BadApvs >> 5) & 0x1) + ((bc.BadApvs >> 4) & 0x1) + ((bc.BadApvs >> 3) & 0x1) +
                          ((bc.BadApvs >> 2) & 0x1) + ((bc.BadApvs >> 1) & 0x1) + ((bc.BadApvs) & 0x1);
    }
    if (bc.BadFibers) {
      nBad[i][0][1] += ((bc.BadFibers >> 2) & 0x1) + ((bc.BadFibers >> 1) & 0x1) + ((bc.BadFibers) & 0x1);
      nBad[i][comp][1] += ((bc.BadFibers >> 2) & 0x1) + ((bc.BadFibers >> 1) & 0x1) + ((bc.BadFibers) & 0x1);
    }
    if (bc.BadModule) {
      nBad[i][0][0]++;
      nBad[i][comp][0]++;
    }
  }
}  // namespace

void SiStripHitEfficiencyHarvester::printAndWriteBadModules(const SiStripQuality& quality,
                                                            const SiStripDetInfo& detInfo) const {
  ////////////////////////////////////////////////////////////////////////
  //try to write out what's in the quality record
  /////////////////////////////////////////////////////////////////////////////
  int nTkBadComp[4];  //k: 0=BadModule, 1=BadFiber, 2=BadApv, 3=BadStrips
  int nBadComp[4][19][4];
  //legend: nBadComp[i][j][k]= SubSystem i, layer/disk/wheel j, BadModule/Fiber/Apv k
  //     i: 0=TIB, 1=TID, 2=TOB, 3=TEC
  //     k: 0=BadModule, 1=BadFiber, 2=BadApv, 3=BadStrips
  std::stringstream ssV[4][19];

  for (int i = 0; i < 4; ++i) {
    nTkBadComp[i] = 0;
    for (int j = 0; j < 19; ++j) {
      ssV[i][j].str("");
      for (int k = 0; k < 4; ++k)
        nBadComp[i][j][k] = 0;
    }
  }

  for (const auto& bc : quality.getBadComponentList()) {
    // Full Tk
    if (bc.BadModule)
      nTkBadComp[0]++;
    if (bc.BadFibers)
      nTkBadComp[1] += ((bc.BadFibers >> 2) & 0x1) + ((bc.BadFibers >> 1) & 0x1) + ((bc.BadFibers) & 0x1);
    if (bc.BadApvs)
      nTkBadComp[2] += ((bc.BadApvs >> 5) & 0x1) + ((bc.BadApvs >> 4) & 0x1) + ((bc.BadApvs >> 3) & 0x1) +
                       ((bc.BadApvs >> 2) & 0x1) + ((bc.BadApvs >> 1) & 0x1) + ((bc.BadApvs) & 0x1);
    // single subsystem
    DetId det(bc.detid);
    if ((det.subdetId() >= SiStripSubdetector::TIB) && (det.subdetId() <= SiStripSubdetector::TEC)) {
      const auto nAPV = detInfo.getNumberOfApvsAndStripLength(det).first;
      switch (det.subdetId()) {
        case SiStripSubdetector::TIB:
          setBadComponents(0, tTopo_->tibLayer(det), bc, ssV, nBadComp, nAPV);
          break;
        case SiStripSubdetector::TID:
          setBadComponents(1,
                           (tTopo_->tidSide(det) == 2 ? tTopo_->tidWheel(det) : tTopo_->tidWheel(det) + 3),
                           bc,
                           ssV,
                           nBadComp,
                           nAPV);
          break;
        case SiStripSubdetector::TOB:
          setBadComponents(2, tTopo_->tobLayer(det), bc, ssV, nBadComp, nAPV);
          break;
        case SiStripSubdetector::TEC:
          setBadComponents(3,
                           (tTopo_->tecSide(det) == 2 ? tTopo_->tecWheel(det) : tTopo_->tecWheel(det) + 9),
                           bc,
                           ssV,
                           nBadComp,
                           nAPV);
          break;
        default:
          break;
      }
    }
  }
  // single strip info
  for (auto rp = quality.getRegistryVectorBegin(); rp != quality.getRegistryVectorEnd(); ++rp) {
    DetId det{rp->detid};
    int subdet = -999;
    int component = -999;
    switch (det.subdetId()) {
      case SiStripSubdetector::TIB:
        subdet = 0;
        component = tTopo_->tibLayer(det);
        break;
      case SiStripSubdetector::TID:
        subdet = 1;
        component = tTopo_->tidSide(det) == 2 ? tTopo_->tidWheel(det) : tTopo_->tidWheel(det) + 3;
        break;
      case SiStripSubdetector::TOB:
        subdet = 2;
        component = tTopo_->tobLayer(det);
        break;
      case SiStripSubdetector::TEC:
        subdet = 3;
        component = tTopo_->tecSide(det) == 2 ? tTopo_->tecWheel(det) : tTopo_->tecWheel(det) + 9;
        break;
      default:
        break;
    }

    const auto pQdvBegin = quality.getDataVectorBegin();
    const auto sqrange = SiStripQuality::Range(pQdvBegin + rp->ibegin, pQdvBegin + rp->iend);
    float percentage = 0;
    for (int it = 0; it < sqrange.second - sqrange.first; it++) {
      unsigned int range = quality.decode(*(sqrange.first + it)).range;
      nTkBadComp[3] += range;
      nBadComp[subdet][0][3] += range;
      nBadComp[subdet][component][3] += range;
      percentage += range;
    }
    if (percentage != 0)
      percentage /= (sistrip::STRIPS_PER_APV * detInfo.getNumberOfApvsAndStripLength(det).first);
    if (percentage > 1)
      edm::LogError("SiStripHitEfficiencyHarvester") << "PROBLEM detid " << det.rawId() << " value " << percentage;
  }

  // printout
  std::ostringstream ss;
  ss << "\n-----------------\nGlobal Info\n-----------------";
  ss << "\nBadComp \t	Modules \tFibers "
        "\tApvs\tStrips\n----------------------------------------------------------------";
  ss << "\nTracker:\t\t" << nTkBadComp[0] << "\t" << nTkBadComp[1] << "\t" << nTkBadComp[2] << "\t" << nTkBadComp[3];
  ss << "\nTIB:\t\t\t" << nBadComp[0][0][0] << "\t" << nBadComp[0][0][1] << "\t" << nBadComp[0][0][2] << "\t"
     << nBadComp[0][0][3];
  ss << "\nTID:\t\t\t" << nBadComp[1][0][0] << "\t" << nBadComp[1][0][1] << "\t" << nBadComp[1][0][2] << "\t"
     << nBadComp[1][0][3];
  ss << "\nTOB:\t\t\t" << nBadComp[2][0][0] << "\t" << nBadComp[2][0][1] << "\t" << nBadComp[2][0][2] << "\t"
     << nBadComp[2][0][3];
  ss << "\nTEC:\t\t\t" << nBadComp[3][0][0] << "\t" << nBadComp[3][0][1] << "\t" << nBadComp[3][0][2] << "\t"
     << nBadComp[3][0][3];
  ss << "\n";

  for (int i = 1; i < 5; ++i)
    ss << "\nTIB Layer " << i << " :\t\t" << nBadComp[0][i][0] << "\t" << nBadComp[0][i][1] << "\t" << nBadComp[0][i][2]
       << "\t" << nBadComp[0][i][3];
  ss << "\n";
  for (int i = 1; i < 4; ++i)
    ss << "\nTID+ Disk " << i << " :\t\t" << nBadComp[1][i][0] << "\t" << nBadComp[1][i][1] << "\t" << nBadComp[1][i][2]
       << "\t" << nBadComp[1][i][3];
  for (int i = 4; i < 7; ++i)
    ss << "\nTID- Disk " << i - 3 << " :\t\t" << nBadComp[1][i][0] << "\t" << nBadComp[1][i][1] << "\t"
       << nBadComp[1][i][2] << "\t" << nBadComp[1][i][3];
  ss << "\n";
  for (int i = 1; i < 7; ++i)
    ss << "\nTOB Layer " << i << " :\t\t" << nBadComp[2][i][0] << "\t" << nBadComp[2][i][1] << "\t" << nBadComp[2][i][2]
       << "\t" << nBadComp[2][i][3];
  ss << "\n";
  for (int i = 1; i < 10; ++i)
    ss << "\nTEC+ Disk " << i << " :\t\t" << nBadComp[3][i][0] << "\t" << nBadComp[3][i][1] << "\t" << nBadComp[3][i][2]
       << "\t" << nBadComp[3][i][3];
  for (int i = 10; i < 19; ++i)
    ss << "\nTEC- Disk " << i - 9 << " :\t\t" << nBadComp[3][i][0] << "\t" << nBadComp[3][i][1] << "\t"
       << nBadComp[3][i][2] << "\t" << nBadComp[3][i][3];
  ss << "\n";

  ss << "\n----------------------------------------------------------------\n\t\t   Detid  \tModules Fibers "
        "Apvs\n----------------------------------------------------------------";
  for (int i = 1; i < 5; ++i)
    ss << "\nTIB Layer " << i << " :" << ssV[0][i].str();
  ss << "\n";
  for (int i = 1; i < 4; ++i)
    ss << "\nTID+ Disk " << i << " :" << ssV[1][i].str();
  for (int i = 4; i < 7; ++i)
    ss << "\nTID- Disk " << i - 3 << " :" << ssV[1][i].str();
  ss << "\n";
  for (int i = 1; i < 7; ++i)
    ss << "\nTOB Layer " << i << " :" << ssV[2][i].str();
  ss << "\n";
  for (int i = 1; i < 10; ++i)
    ss << "\nTEC+ Disk " << i << " :" << ssV[3][i].str();
  for (int i = 10; i < 19; ++i)
    ss << "\nTEC- Disk " << i - 9 << " :" << ssV[3][i].str();

  LOGPRINT << ss.str();

  // store also bad modules in log file
  std::ofstream badModules;
  badModules.open("BadModules_NEW.log");
  badModules << "\n----------------------------------------------------------------\n\t\t   Detid  \tModules Fibers "
                "Apvs\n----------------------------------------------------------------";
  for (int i = 1; i < 5; ++i)
    badModules << "\nTIB Layer " << i << " :" << ssV[0][i].str();
  badModules << "\n";
  for (int i = 1; i < 4; ++i)
    badModules << "\nTID+ Disk " << i << " :" << ssV[1][i].str();
  for (int i = 4; i < 7; ++i)
    badModules << "\nTID- Disk " << i - 3 << " :" << ssV[1][i].str();
  badModules << "\n";
  for (int i = 1; i < 7; ++i)
    badModules << "\nTOB Layer " << i << " :" << ssV[2][i].str();
  badModules << "\n";
  for (int i = 1; i < 10; ++i)
    badModules << "\nTEC+ Disk " << i << " :" << ssV[3][i].str();
  for (int i = 10; i < 19; ++i)
    badModules << "\nTEC- Disk " << i - 9 << " :" << ssV[3][i].str();
  badModules.close();
}

void SiStripHitEfficiencyHarvester::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<bool>("isAtPCL", false);
  desc.add<bool>("doStoreOnDB", false);
  desc.add<std::string>("Record", "SiStripBadStrip");
  desc.add<double>("Threshold", 0.1);
  desc.add<std::string>("Title", "Hit Efficiency");
  desc.add<int>("nModsMin", 5);
  desc.addUntracked<bool>("AutoIneffModTagging", false);
  desc.addUntracked<double>("TkMapMin", 0.9);
  desc.addUntracked<bool>("ShowRings", false);
  descriptions.addWithDefaultLabel(desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SiStripHitEfficiencyHarvester);
