// user includes
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"
#include "CalibTracker/Records/interface/SiStripQualityRcd.h"
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"
#include "CalibTracker/SiStripHitEfficiency/interface/SiStripHitEffData.h"
#include "CalibTracker/SiStripHitEfficiency/interface/SiStripHitEfficiencyHelpers.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "DQM/SiStripCommon/interface/TkHistoMap.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "DataFormats/SiStripCommon/interface/ConstantsForHardwareSystems.h" /* for STRIPS_PER_APV*/
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
#include <boost/type_index.hpp>
#include <fmt/printf.h>
#include <numeric>  // for std::accumulate
#include <sstream>

// ROOT includes
#include "TCanvas.h"
#include "TEfficiency.h"
#include "TGraphAsymmErrors.h"
#include "TLegend.h"
#include "TStyle.h"
#include "TTree.h"

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
  SiStripHitEffData calibData_;
  const std::string inputFolder_;
  const bool isAtPCL_;
  const bool autoIneffModTagging_, doStoreOnDB_;
  const bool doStoreOnTree_;
  const bool showRings_, showEndcapSides_, showTOB6TEC9_, showOnlyGoodModules_;
  const unsigned int nTEClayers_;
  const double threshold_;
  const int nModsMin_;
  const float effPlotMin_;
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

  int goodlayertotal[bounds::k_END_OF_LAYS_AND_RINGS];
  int goodlayerfound[bounds::k_END_OF_LAYS_AND_RINGS];
  int alllayertotal[bounds::k_END_OF_LAYS_AND_RINGS];
  int alllayerfound[bounds::k_END_OF_LAYS_AND_RINGS];

  // information for the TTree
  TTree* tree;
  unsigned int t_DetId, t_found, t_total;
  unsigned char t_layer;
  bool t_isTaggedIneff;
  float t_threshold;

  void writeBadStripPayload(const SiStripQuality& quality) const;
  void printTotalStatistics(const std::array<long, bounds::k_END_OF_LAYERS>& layerFound,
                            const std::array<long, bounds::k_END_OF_LAYERS>& layerTotal) const;
  void printAndWriteBadModules(const SiStripQuality& quality, const SiStripDetInfo& detInfo) const;
  bool checkMapsValidity(const std::vector<MonitorElement*>& maps, const std::string& type) const;
  unsigned int countTotalHits(const std::vector<MonitorElement*>& maps); /* to check if TK was ON */
  void makeSummary(DQMStore::IGetter& getter, DQMStore::IBooker& booker, bool doProfiles = false) const;
  template <typename T>
  void setEffBinLabels(const T gr, const T gr2, const unsigned int nLayers) const;
  void makeSummaryVsVariable(DQMStore::IGetter& getter, DQMStore::IBooker& booker, ::projections theProj) const;
};

SiStripHitEfficiencyHarvester::SiStripHitEfficiencyHarvester(const edm::ParameterSet& conf)
    : inputFolder_(conf.getParameter<std::string>("inputFolder")),
      isAtPCL_(conf.getParameter<bool>("isAtPCL")),
      autoIneffModTagging_(conf.getUntrackedParameter<bool>("AutoIneffModTagging", false)),
      doStoreOnDB_(conf.getParameter<bool>("doStoreOnDB")),
      doStoreOnTree_(conf.getUntrackedParameter<bool>("doStoreOnTree")),
      showRings_(conf.getUntrackedParameter<bool>("ShowRings", false)),
      showEndcapSides_(conf.getUntrackedParameter<bool>("ShowEndcapSides", true)),
      showTOB6TEC9_(conf.getUntrackedParameter<bool>("ShowTOB6TEC9", false)),
      showOnlyGoodModules_(conf.getUntrackedParameter<bool>("ShowOnlyGoodModules", false)),
      nTEClayers_(showRings_ ? 7 : 9),  // number of rings or wheels
      threshold_(conf.getParameter<double>("Threshold")),
      nModsMin_(conf.getParameter<int>("nModsMin")),
      effPlotMin_(conf.getUntrackedParameter<double>("EffPlotMin", 0.9)),
      tkMapMin_(conf.getUntrackedParameter<double>("TkMapMin", 0.9)),
      title_(conf.getParameter<std::string>("Title")),
      record_(conf.getParameter<std::string>("Record")),
      tTopoToken_(esConsumes<edm::Transition::EndRun>()),
      tkDetMapToken_(esConsumes<edm::Transition::EndRun>()),
      stripQualityToken_(esConsumes<edm::Transition::EndRun>()),
      tkGeomToken_(esConsumes<edm::Transition::EndRun>()) {
  // zero in all counts
  for (int l = 0; l < bounds::k_END_OF_LAYS_AND_RINGS; l++) {
    goodlayertotal[l] = 0;
    goodlayerfound[l] = 0;
    alllayertotal[l] = 0;
    alllayerfound[l] = 0;
  }
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
  std::vector<bool> isThere;
  isThere.reserve(maps.size());
  std::transform(maps.begin() + 1, maps.end(), std::back_inserter(isThere), [](auto& x) { return !(x == nullptr); });

  int count{0};
  for (const auto& it : isThere) {
    count++;
    LogDebug("SiStripHitEfficiencyHarvester") << " layer: " << count << " " << it << std::endl;
    if (it)
      LogDebug("SiStripHitEfficiencyHarvester") << "resolving to " << maps[count]->getName() << std::endl;
  }

  // check on the input TkHistoMap
  bool areMapsAvailable{true};
  int layerCount{0};
  for (const auto& it : isThere) {
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

unsigned int SiStripHitEfficiencyHarvester::countTotalHits(const std::vector<MonitorElement*>& maps) {
  return std::accumulate(maps.begin() + 1, maps.end(), 0, [](unsigned int total, MonitorElement* item) {
    return total + item->getEntries();
  });
}

void SiStripHitEfficiencyHarvester::dqmEndJob(DQMStore::IBooker& booker, DQMStore::IGetter& getter) {
  if (!isAtPCL_) {
    edm::Service<TFileService> fs;
    if (!fs.isAvailable()) {
      throw cms::Exception("BadConfig") << "TFileService unavailable: "
                                        << "please add it to config file";
    }

    if (doStoreOnTree_) {
      // store information per DetId in the output tree
      tree = fs->make<TTree>("ModEff", "ModEff");
      tree->Branch("DetId", &t_DetId, "DetId/i");
      tree->Branch("Layer", &t_layer, "Layer/b");
      tree->Branch("FoundHits", &t_found, "FoundHits/i");
      tree->Branch("AllHits", &t_total, "AllHits/i");
      tree->Branch("IsTaggedIneff", &t_isTaggedIneff, "IsTaggedIneff/O");
      tree->Branch("TagThreshold", &t_threshold, "TagThreshold/F");
    }
  }

  if (!autoIneffModTagging_)
    LOGPRINT << "A module is bad if efficiency < " << threshold_ << " and has at least " << nModsMin_ << " nModsMin.";
  else
    LOGPRINT << "A module is bad if the upper limit on the efficiency is < to the avg in the layer - " << threshold_
             << " and has at least " << nModsMin_ << " nModsMin.";

  auto h_module_total = std::make_unique<TkHistoMap>(tkDetMap_.get());
  h_module_total->loadTkHistoMap(fmt::format("{}/TkDetMaps", inputFolder_), "perModule_total");
  auto h_module_found = std::make_unique<TkHistoMap>(tkDetMap_.get());
  h_module_found->loadTkHistoMap(fmt::format("{}/TkDetMaps", inputFolder_), "perModule_found");

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

  // count how many hits in the denominator we have
  const unsigned int totalHits = this->countTotalHits(totalMaps);

  // set colz
  for (size_t i = 1; i < totalMaps.size(); i++) {
    h_module_total->getMap(i)->setOption("colz");
    h_module_found->getMap(i)->setOption("colz");
  }

  // come back to the main folder
  booker.setCurrentFolder(inputFolder_);

  std::vector<MonitorElement*> hEffInLayer(std::size_t(1), nullptr);
  hEffInLayer.reserve(bounds::k_END_OF_LAYERS);
  for (std::size_t i = 1; i != bounds::k_END_OF_LAYERS; ++i) {
    const auto lyrName = ::layerName(i, showRings_, nTEClayers_);
    hEffInLayer.push_back(booker.book1D(
        Form("eff_layer%i", int(i)), Form("Module efficiency in layer %s", lyrName.c_str()), 201, 0, 1.005));
  }
  std::array<long, bounds::k_END_OF_LAYERS> layerTotal{};
  std::array<long, bounds::k_END_OF_LAYERS> layerFound{};
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

  // load the FEDError map
  const auto& EventStats = getter.get(fmt::format("{}/EventInfo/EventStats", inputFolder_));
  const int totalEvents = EventStats->getBinContent(1., 1.);  // first bin contains info on number of events run
  calibData_.FEDErrorOccupancy = std::make_unique<TkHistoMap>(tkDetMap_.get());
  calibData_.FEDErrorOccupancy->loadTkHistoMap(fmt::format("{}/FEDErrorTkDetMaps", inputFolder_),
                                               "perModule_FEDErrors");

  // tag as bad from FEDErrors the modules that have an error on 75% of the events
  calibData_.fillMapFromTkMap(totalEvents, 0.75, stripDetIds_);

  for (const auto& [badId, fraction] : calibData_.fedErrorCounts) {
    LogDebug("SiStripHitEfficiencyHarvester")
        << __PRETTY_FUNCTION__ << " bad module from FEDError " << badId << "," << fraction << std::endl;
  }

  for (auto det : stripDetIds_) {
    auto layer = ::checkLayer(det, tTopo_.get());
    const auto num = h_module_found->getValue(det);
    const auto denom = h_module_total->getValue(det);
    if (denom) {
      // use only the "good" modules
      if (stripQuality_->getBadApvs(det) == 0 && calibData_.checkFedError(det)) {
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

          if (doStoreOnTree_ && !isAtPCL_) {
            t_DetId = det.rawId();
            t_layer = layer;
            t_found = num;
            t_total = denom;
            t_isTaggedIneff = false;
            t_threshold = 0;
            tree->Fill();
          }
        }

        //Put any module into the TKMap
        tkMap.fill(det, 1. - eff);
        tkMapEff.fill(det, eff);
        tkMapNum.fill(det, num);
        tkMapDen.fill(det, denom);

        layerTotal[layer] += denom;
        layerFound[layer] += num;

        // for the summary
        // Have to do the decoding for which side to go on (ugh)
        if (layer <= bounds::k_LayersAtTOBEnd) {
          goodlayerfound[layer] += num;
          goodlayertotal[layer] += denom;
        } else if (layer > bounds::k_LayersAtTOBEnd && layer <= bounds::k_LayersAtTIDEnd) {
          if (tTopo_->tidSide(det) == 1) {
            goodlayerfound[layer] += num;
            goodlayertotal[layer] += denom;
          } else if (tTopo_->tidSide(det) == 2) {
            goodlayerfound[layer + 3] += num;
            goodlayertotal[layer + 3] += denom;
          }
        } else if (layer > bounds::k_LayersAtTIDEnd && layer <= bounds::k_LayersAtTECEnd) {
          if (tTopo_->tecSide(det) == 1) {
            goodlayerfound[layer + 3] += num;
            goodlayertotal[layer + 3] += denom;
          } else if (tTopo_->tecSide(det) == 2) {
            goodlayerfound[layer + 3 + nTEClayers_] += num;
            goodlayertotal[layer + 3 + nTEClayers_] += denom;
          }
        }
      }  // if the module is good!

      //Do the one where we don't exclude bad modules!
      if (layer <= bounds::k_LayersAtTOBEnd) {
        alllayerfound[layer] += num;
        alllayertotal[layer] += denom;
      } else if (layer > bounds::k_LayersAtTOBEnd && layer <= bounds::k_LayersAtTIDEnd) {
        if (tTopo_->tidSide(det) == 1) {
          alllayerfound[layer] += num;
          alllayertotal[layer] += denom;
        } else if (tTopo_->tidSide(det) == 2) {
          alllayerfound[layer + 3] += num;
          alllayertotal[layer + 3] += denom;
        }
      } else if (layer > bounds::k_LayersAtTIDEnd && layer <= bounds::k_LayersAtTECEnd) {
        if (tTopo_->tecSide(det) == 1) {
          alllayerfound[layer + 3] += num;
          alllayertotal[layer + 3] += denom;
        } else if (tTopo_->tecSide(det) == 2) {
          alllayerfound[layer + 3 + nTEClayers_] += num;
          alllayertotal[layer + 3 + nTEClayers_] += denom;
        }
      }

    }  // if denom
  }    // loop on DetIds

  if (autoIneffModTagging_) {
    for (unsigned int i = 1; i <= k_LayersAtTECEnd; i++) {
      //Compute threshold to use for each layer
      hEffInLayer[i]->getTH1()->GetXaxis()->SetRange(
          3, hEffInLayer[i]->getNbinsX() + 1);  // Remove from the avg modules below 1%
      const double layer_min_eff = hEffInLayer[i]->getMean() - std::max(2.5 * hEffInLayer[i]->getRMS(), threshold_);
      LOGPRINT << "Layer " << i << " threshold for bad modules: <" << layer_min_eff
               << "  (layer mean: " << hEffInLayer[i]->getMean() << " rms: " << hEffInLayer[i]->getRMS() << ")";

      hEffInLayer[i]->getTH1()->GetXaxis()->SetRange(1, hEffInLayer[i]->getNbinsX() + 1);

      for (auto det : stripDetIds_) {
        // use only the "good" modules
        if (stripQuality_->getBadApvs(det) == 0 && calibData_.checkFedError(det)) {
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
                if (!isAtPCL_ && doStoreOnTree_) {
                  t_isTaggedIneff = true;
                }
              } else {
                //Fill the bad list with empty results for every module
                tkMapBad.fillc(det, 255, 255, 255);
                if (!isAtPCL_ && doStoreOnTree_) {
                  t_isTaggedIneff = false;
                }
              }
              if (eff_up < layer_min_eff + 0.08) {
                // printing message also for modules sligthly above (8%) the limit
                LOGPRINT << "Layer " << layer << " (" << ::layerName(layer, showRings_, nTEClayers_) << ")  module "
                         << det.rawId() << " efficiency: " << eff << " , " << num << "/" << denom
                         << " , upper limit: " << eff_up;
              }
              if (denom < nModsMin_) {
                LOGPRINT << "Layer " << layer << " (" << ::layerName(layer, showRings_, nTEClayers_) << ")  module "
                         << det.rawId() << " layer " << layer << " is under occupancy at " << denom;
              }

              if (!isAtPCL_ && doStoreOnTree_) {
                t_DetId = det.rawId();
                t_layer = layer;
                t_found = num;
                t_total = denom;
                t_threshold = layer_min_eff;
                tree->Fill();
              }  // if storing tree
            }    // if denom
          }      // layer = i
        }        // if there are no bad APVs
      }          // loop on detids
    }            // loop on layers
  }              // if auto tagging

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
    if (totalHits > 0u) {
      writeBadStripPayload(pQuality);
    } else {
      edm::LogPrint("SiStripHitEfficiencyHarvester")
          << __PRETTY_FUNCTION__ << " There are no SiStrip hits for a valid measurement, skipping!";
    }
  } else {
    edm::LogInfo("SiStripHitEfficiencyHarvester") << "Will not produce payload!";
  }

  printTotalStatistics(layerFound, layerTotal);  // statistics by layer and subdetector
  //LOGPRINT << "\n-----------------\nNew IOV starting from run " << e.id().run() << " event " << e.id().event()
  //     << " lumiBlock " << e.luminosityBlock() << " time " << e.time().value() << "\n-----------------\n";
  printAndWriteBadModules(pQuality, detInfo);  // TODO

  // make summary plots
  makeSummary(getter, booker);
  makeSummaryVsVariable(getter, booker, projections::k_vs_LUMI);
  makeSummaryVsVariable(getter, booker, projections::k_vs_PU);
  makeSummaryVsVariable(getter, booker, projections::k_vs_BX);
}

void SiStripHitEfficiencyHarvester::printTotalStatistics(
    const std::array<long, bounds::k_END_OF_LAYERS>& layerFound,
    const std::array<long, bounds::k_END_OF_LAYERS>& layerTotal) const {
  //Calculate the statistics by layer
  int totalfound = 0;
  int totaltotal = 0;
  double layereff;
  int subdetfound[5] = {0, 0, 0, 0, 0};
  int subdettotal[5] = {0, 0, 0, 0, 0};

  for (unsigned int i = 1; i <= bounds::k_LayersAtTECEnd; i++) {
    layereff = double(layerFound[i]) / double(layerTotal[i]);
    LOGPRINT << "Layer " << i << " (" << ::layerName(i, showRings_, nTEClayers_) << ") has total efficiency "
             << layereff << " " << layerFound[i] << "/" << layerTotal[i];
    totalfound += layerFound[i];
    totaltotal += layerTotal[i];
    if (i <= bounds::k_LayersAtTIBEnd) {
      subdetfound[1] += layerFound[i];
      subdettotal[1] += layerTotal[i];
    }
    if (i > bounds::k_LayersAtTIBEnd && i <= bounds::k_LayersAtTOBEnd) {
      subdetfound[2] += layerFound[i];
      subdettotal[2] += layerTotal[i];
    }
    if (i > bounds::k_LayersAtTOBEnd && i <= bounds::k_LayersAtTIDEnd) {
      subdetfound[3] += layerFound[i];
      subdettotal[3] += layerTotal[i];
    }
    if (i > bounds::k_LayersAtTIDEnd) {
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

void SiStripHitEfficiencyHarvester::makeSummary(DQMStore::IGetter& getter,
                                                DQMStore::IBooker& booker,
                                                bool doProfiles) const {
  // use goodlayer_total/found and alllayer_total/found, collapse side and/or ring if needed
  unsigned int nLayers{34};  // default
  if (showRings_)
    nLayers = 30;
  if (!showEndcapSides_) {
    if (!showRings_)
      nLayers = 22;
    else
      nLayers = 20;
  }

  // come back to the main folder and create a final efficiency folder
  booker.setCurrentFolder(fmt::format("{}/EfficiencySummary", inputFolder_));
  MonitorElement* found = booker.book1D("found", "found", nLayers + 1, 0, nLayers + 1);
  MonitorElement* all = booker.book1D("all", "all", nLayers + 1, 0, nLayers + 1);
  MonitorElement* found2 = booker.book1D("found2", "found", nLayers + 1, 0, nLayers + 1);
  MonitorElement* all2 = booker.book1D("all2", "all2", nLayers + 1, 0, nLayers + 1);

  // first bin only to keep real data off the y axis so set to -1
  found->setBinContent(0, -1);
  all->setBinContent(0, 1);

  // new ROOT version: TGraph::Divide don't handle null or negative values
  for (unsigned int i = 1; i < nLayers + 2; ++i) {
    found->setBinContent(i, 1e-6);
    all->setBinContent(i, 1);
    found2->setBinContent(i, 1e-6);
    all2->setBinContent(i, 1);
  }

  TCanvas* c7 = new TCanvas("c7", " test ", 10, 10, 800, 600);
  c7->SetFillColor(0);
  c7->SetGrid();

  unsigned int nLayers_max = nLayers + 1;  // barrel+endcap
  if (!showEndcapSides_)
    nLayers_max = 11;  // barrel
  for (unsigned int i = 1; i < nLayers_max; ++i) {
    LOGPRINT << "Fill only good modules layer " << i << ":  S = " << goodlayerfound[i]
             << "    B = " << goodlayertotal[i];
    if (goodlayertotal[i] > 5) {
      found->setBinContent(i, goodlayerfound[i]);
      all->setBinContent(i, goodlayertotal[i]);
    }

    LOGPRINT << "Filling all modules layer " << i << ":  S = " << alllayerfound[i] << "    B = " << alllayertotal[i];
    if (alllayertotal[i] > 5) {
      found2->setBinContent(i, alllayerfound[i]);
      all2->setBinContent(i, alllayertotal[i]);
    }
  }

  // endcap - merging sides
  if (!showEndcapSides_) {
    for (unsigned int i = 11; i < 14; ++i) {  // TID disks
      LOGPRINT << "Fill only good modules layer " << i << ":  S = " << goodlayerfound[i] + goodlayerfound[i + 3]
               << "    B = " << goodlayertotal[i] + goodlayertotal[i + 3];
      if (goodlayertotal[i] + goodlayertotal[i + 3] > 5) {
        found->setBinContent(i, goodlayerfound[i] + goodlayerfound[i + 3]);
        all->setBinContent(i, goodlayertotal[i] + goodlayertotal[i + 3]);
      }
      LOGPRINT << "Filling all modules layer " << i << ":  S = " << alllayerfound[i] + alllayerfound[i + 3]
               << "    B = " << alllayertotal[i] + alllayertotal[i + 3];
      if (alllayertotal[i] + alllayertotal[i + 3] > 5) {
        found2->setBinContent(i, alllayerfound[i] + alllayerfound[i + 3]);
        all2->setBinContent(i, alllayertotal[i] + alllayertotal[i + 3]);
      }
    }
    for (unsigned int i = 17; i < 17 + nTEClayers_; ++i) {  // TEC disks
      LOGPRINT << "Fill only good modules layer " << i - 3
               << ":  S = " << goodlayerfound[i] + goodlayerfound[i + nTEClayers_]
               << "    B = " << goodlayertotal[i] + goodlayertotal[i + nTEClayers_];
      if (goodlayertotal[i] + goodlayertotal[i + nTEClayers_] > 5) {
        found->setBinContent(i - 3, goodlayerfound[i] + goodlayerfound[i + nTEClayers_]);
        all->setBinContent(i - 3, goodlayertotal[i] + goodlayertotal[i + nTEClayers_]);
      }
      LOGPRINT << "Filling all modules layer " << i - 3
               << ":  S = " << alllayerfound[i] + alllayerfound[i + nTEClayers_]
               << "    B = " << alllayertotal[i] + alllayertotal[i + nTEClayers_];
      if (alllayertotal[i] + alllayertotal[i + nTEClayers_] > 5) {
        found2->setBinContent(i - 3, alllayerfound[i] + alllayerfound[i + nTEClayers_]);
        all2->setBinContent(i - 3, alllayertotal[i] + alllayertotal[i + nTEClayers_]);
      }
    }
  }

  found->getTH1F()->Sumw2();
  all->getTH1F()->Sumw2();

  found2->getTH1F()->Sumw2();
  all2->getTH1F()->Sumw2();

  MonitorElement* h_eff_all =
      booker.book1D("eff_all", "Strip hit efficiency for all modules", nLayers + 1, 0, nLayers + 1);
  MonitorElement* h_eff_good =
      booker.book1D("eff_good", "Strip hit efficiency for good modules", nLayers + 1, 0, nLayers + 1);

  if (doProfiles) {
    // now do the profile
    TProfile* profile_all = ::computeEff(found2->getTH1F(), all2->getTH1F(), "all");
    profile_all->SetMinimum(tkMapMin_);
    profile_all->SetTitle("Strip hit efficiency for all modules");
    booker.bookProfile(profile_all->GetName(), profile_all);

    TProfile* profile_good = ::computeEff(found->getTH1F(), all->getTH1F(), "good");
    profile_good->SetMinimum(tkMapMin_);
    profile_good->SetTitle("Strip hit efficiency for good modules");
    booker.bookProfile(profile_good->GetName(), profile_good);

    // clean the house
    delete profile_all;
    delete profile_good;
  }

  for (int i = 1; i < found->getNbinsX(); i++) {
    const auto& den_all = all2->getBinContent(i);
    const auto& num_all = found2->getBinContent(i);
    const auto& den_good = all->getBinContent(i);
    const auto& num_good = found->getBinContent(i);

    // fill all modules efficiency
    if (den_all > 0.) {
      // naive binomial errors
      //float eff_all = num_all / den_all;
      //float err_eff_all = (eff_all * (1 - eff_all)) / den_all;

      // use Clopper-Pearson errors
      const auto& effPair_all = ::computeCPEfficiency(num_all, den_all);
      h_eff_all->setBinContent(i, effPair_all.value());
      h_eff_all->setBinError(i, effPair_all.error());
    }

    // fill good modules efficiency
    if (den_good > 0.) {
      // naive binomial errors
      //float eff_good = num_good / den_good;
      //float err_eff_good = (eff_good * (1 - eff_good)) / den_good;

      // use Clopper-Pearson errors
      const auto& effPair_good = ::computeCPEfficiency(num_good, den_good);
      h_eff_good->setBinContent(i, effPair_good.value());
      h_eff_good->setBinError(i, effPair_good.error());
    }
  }

  h_eff_all->getTH1F()->SetMinimum(effPlotMin_);
  h_eff_good->getTH1F()->SetMinimum(effPlotMin_);

  // set the histogram bin labels
  this->setEffBinLabels(h_eff_all->getTH1F(), h_eff_good->getTH1F(), nLayers);

  if (!isAtPCL_) {
    // if TFileService is not avaible, just go on
    edm::Service<TFileService> fs;
    if (!fs.isAvailable()) {
      throw cms::Exception("BadConfig") << "TFileService unavailable: "
                                        << "please add it to config file";
    }

    TGraphAsymmErrors* gr = (*fs).make<TGraphAsymmErrors>(nLayers + 1);
    gr->SetName("eff_good");
    gr->BayesDivide(found->getTH1F(), all->getTH1F());

    TGraphAsymmErrors* gr2 = (*fs).make<TGraphAsymmErrors>(nLayers + 1);
    gr2->SetName("eff_all");
    gr2->BayesDivide(found2->getTH1F(), all2->getTH1F());

    for (unsigned int j = 0; j < nLayers + 1; j++) {
      gr->SetPointError(j, 0., 0., gr->GetErrorYlow(j), gr->GetErrorYhigh(j));
      gr2->SetPointError(j, 0., 0., gr2->GetErrorYlow(j), gr2->GetErrorYhigh(j));
    }

    this->setEffBinLabels(gr, gr2, nLayers);

    gr->GetXaxis()->SetLimits(0, nLayers);
    gr->SetMarkerColor(2);
    gr->SetMarkerSize(1.2);
    gr->SetLineColor(2);
    gr->SetLineWidth(4);
    gr->SetMarkerStyle(20);
    gr->SetMinimum(effPlotMin_);
    gr->SetMaximum(1.001);
    gr->GetYaxis()->SetTitle("Efficiency");
    gStyle->SetTitleFillColor(0);
    gStyle->SetTitleBorderSize(0);
    gr->SetTitle("SiStripHitEfficiency by Layer");

    gr2->GetXaxis()->SetLimits(0, nLayers);
    gr2->SetMarkerColor(1);
    gr2->SetMarkerSize(1.2);
    gr2->SetLineColor(1);
    gr2->SetLineWidth(4);
    gr2->SetMarkerStyle(21);
    gr2->SetMinimum(effPlotMin_);
    gr2->SetMaximum(1.001);
    gr2->GetYaxis()->SetTitle("Efficiency");
    gr2->SetTitle("SiStripHitEfficiency by Layer");

    gr->Draw("AP");
    gr->GetXaxis()->SetNdivisions(36);

    c7->cd();
    TPad* overlay = new TPad("overlay", "", 0, 0, 1, 1);
    overlay->SetFillStyle(4000);
    overlay->SetFillColor(0);
    overlay->SetFrameFillStyle(4000);
    overlay->Draw("same");
    overlay->cd();
    if (!showOnlyGoodModules_)
      gr2->Draw("AP");

    TLegend* leg = new TLegend(0.70, 0.27, 0.88, 0.40);
    leg->AddEntry(gr, "Good Modules", "p");
    if (!showOnlyGoodModules_)
      leg->AddEntry(gr2, "All Modules", "p");
    leg->SetTextSize(0.020);
    leg->SetFillColor(0);
    leg->Draw("same");

    c7->SaveAs("Summary.png");
    c7->SaveAs("Summary.root");
  }  // if it's not run at PCL
}

template <typename T>
void SiStripHitEfficiencyHarvester::setEffBinLabels(const T gr, const T gr2, const unsigned int nLayers) const {
  LogDebug("SiStripHitEfficiencyHarvester")
      << "nLayers = " << nLayers << " number of bins, gr1: " << gr->GetXaxis()->GetNbins()
      << " number of bins, gr2: " << gr2->GetXaxis()->GetNbins() << " showRings: " << showRings_
      << " showEndcapSides: " << showEndcapSides_ << " type of object is "
      << boost::typeindex::type_id<T>().pretty_name();

  for (unsigned int k = 1; k < nLayers + 1; k++) {
    std::string label{};
    if (showEndcapSides_)
      label = ::layerSideName(k, showRings_, nTEClayers_);
    else
      label = ::layerName(k, showRings_, nTEClayers_);
    if (!showTOB6TEC9_) {
      if (k == 10)
        label = "";
      if (!showRings_ && k == nLayers)
        label = "";
      if (!showRings_ && showEndcapSides_ && k == 25)
        label = "";
    }

    int bin{-1};
    if constexpr (std::is_same_v<T, TGraphAsymmErrors*>) {
      edm::LogInfo("SiStripHitEfficiencyHarvester")
          << "class name: " << gr->ClassName() << " expected TGraphAsymErrors" << std::endl;
      if (!showRings_) {
        if (showEndcapSides_) {
          bin = (((k + 1) * 100 + 2) / (nLayers)-4);
        } else {
          bin = ((k + 1) * 100 / (nLayers)-6);
        }
      } else {
        if (showEndcapSides_) {
          bin = ((k + 1) * 100 / (nLayers)-4);
        } else {
          bin = ((k + 1) * 100 / (nLayers)-7);
        }
      }
    } else {
      edm::LogInfo("SiStripHitEfficiencyHarvester")
          << "class name: " << gr->ClassName() << " expected TH1F" << std::endl;
      bin = k;
    }
    gr->GetXaxis()->SetBinLabel(bin, label.data());
    gr2->GetXaxis()->SetBinLabel(bin, label.data());
  }
}

void SiStripHitEfficiencyHarvester::makeSummaryVsVariable(DQMStore::IGetter& getter,
                                                          DQMStore::IBooker& booker,
                                                          ::projections theProj) const {
  std::vector<MonitorElement*> effVsVariable;
  effVsVariable.reserve(showRings_ ? 20 : 22);

  const auto& folderString = ::projFolder[theProj];
  const auto& foundHistoString = ::projFoundHisto[theProj];
  const auto& totalHistoString = ::projTotalHisto[theProj];
  const auto& titleString = ::projTitle[theProj];
  const auto& titleXString = ::projXtitle[theProj];

  LogDebug("SiStripHitEfficiencyHarvester")
      << " inside" << __PRETTY_FUNCTION__ << " from " << ::projFolder[theProj] << " " << __LINE__ << std::endl;

  for (unsigned int iLayer = 1; iLayer != (showRings_ ? 20 : 22); ++iLayer) {
    LogDebug("SiStripHitEfficiencyHarvester")
        << "iLayer " << iLayer << " " << fmt::format("{}/{}/{}{}", inputFolder_, folderString, foundHistoString, iLayer)
        << std::endl;

    const auto lyrName = ::layerName(iLayer, showRings_, nTEClayers_);
    auto hfound = getter.get(fmt::format("{}/{}/{}{}", inputFolder_, folderString, foundHistoString, iLayer));
    auto htotal = getter.get(fmt::format("{}/{}/{}{}", inputFolder_, folderString, totalHistoString, iLayer));

    if (hfound == nullptr or htotal == nullptr) {
      if (hfound == nullptr)
        edm::LogError("SiStripHitEfficiencyHarvester")
            << fmt::format("{}/{}/{}{}", inputFolder_, folderString, foundHistoString, iLayer) << " was not found!";
      if (htotal == nullptr)
        edm::LogError("SiStripHitEfficiencyHarvester")
            << fmt::format("{}/{}/{}{}", inputFolder_, folderString, totalHistoString, iLayer) << " was not found!";
      // no input histograms -> continue in the loop
      continue;
    }

    // in order to display correct errors when taking the ratio
    if (!hfound->getTH1F()->GetSumw2())
      hfound->getTH1F()->Sumw2();
    if (!htotal->getTH1F()->GetSumw2())
      htotal->getTH1F()->Sumw2();

    // prevent dividing by 0
    for (int i = 0; i != hfound->getNbinsX() + 1; ++i) {
      if (hfound->getBinContent(i) == 0)
        hfound->setBinContent(i, 1e-6);
      if (htotal->getBinContent(i) == 0)
        htotal->setBinContent(i, 1);
    }
    LogDebug("SiStripHitEfficiencyHarvester") << "Total hits for layer " << iLayer << " (" << folderString
                                              << "): " << htotal->getEntries() << ", found " << hfound->getEntries();

    booker.setCurrentFolder(fmt::format("{}/EfficiencySummary{}", inputFolder_, folderString));
    effVsVariable[iLayer] = booker.book1D(
        fmt::sprintf("eff%sLayer%s", folderString, lyrName),
        fmt::sprintf("Efficiency vs %s for layer %s;%s;SiStrip Hit efficiency", titleString, lyrName, titleXString),
        hfound->getNbinsX(),
        hfound->getAxisMin(),
        hfound->getAxisMax());

    LogDebug("SiStripHitEfficiencyHarvester")
        << " bin 0 " << hfound->getAxisMin() << " bin last: " << hfound->getAxisMax() << std::endl;

    for (int i = 0; i != hfound->getNbinsX() + 1; ++i) {
      const auto& den = htotal->getBinContent(i);
      const auto& num = hfound->getBinContent(i);

      // fill all modules efficiency
      if (den > 0.) {
        const auto& effPair = ::computeCPEfficiency(num, den);
        effVsVariable[iLayer]->setBinContent(i, effPair.value());
        effVsVariable[iLayer]->setBinError(i, effPair.error());

        LogDebug("SiStripHitEfficiencyHarvester")
            << __PRETTY_FUNCTION__ << " " << lyrName << " bin:" << i << " err:" << effPair.error() << std::endl;
      }
    }

    // graphics adjustment
    effVsVariable[iLayer]->getTH1F()->SetMinimum(tkMapMin_);

    // now do the profile
    TProfile* profile = ::computeEff(hfound->getTH1F(), htotal->getTH1F(), lyrName);
    TString title =
        fmt::sprintf("Efficiency vs %s for layer %s;%s;SiStrip Hit efficiency", titleString, lyrName, titleXString);
    profile->SetMinimum(tkMapMin_);

    profile->SetTitle(title.Data());
    booker.bookProfile(profile->GetName(), profile);

    delete profile;
  }  // loop on layers
}

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
  desc.add<std::string>("inputFolder", "AlCaReco/SiStripHitEfficiency");
  desc.add<bool>("isAtPCL", false);
  desc.add<bool>("doStoreOnDB", false);
  desc.add<std::string>("Record", "SiStripBadStrip");
  desc.add<double>("Threshold", 0.1);
  desc.add<std::string>("Title", "Hit Efficiency");
  desc.add<int>("nModsMin", 5);
  desc.addUntracked<bool>("doStoreOnTree", false);
  desc.addUntracked<bool>("AutoIneffModTagging", false);
  desc.addUntracked<double>("TkMapMin", 0.9);
  desc.addUntracked<double>("EffPlotMin", 0.9);
  desc.addUntracked<bool>("ShowRings", false);
  desc.addUntracked<bool>("ShowEndcapSides", true);
  desc.addUntracked<bool>("ShowTOB6TEC9", false);
  desc.addUntracked<bool>("ShowOnlyGoodModules", false);
  descriptions.addWithDefaultLabel(desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SiStripHitEfficiencyHarvester);
