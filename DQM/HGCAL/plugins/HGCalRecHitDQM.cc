#include <algorithm>
#include <limits>
#include <string>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DataFormats/HGCalReco/interface/HGCalSoARecHitsHostCollection.h"
#include "CondFormats/DataRecord/interface/HGCalDenseIndexInfoRcd.h"
#include "CondFormats/HGCalObjects/interface/HGCalMappingModuleIndexer.h"
#include "CondFormats/HGCalObjects/interface/HGCalMappingParameterHost.h"
#include "DQM/HGCAL/interface/HGCalDQMCommon.h"
#include "CondFormats/DataRecord/interface/HGCalElectronicsMappingRcd.h"

/**
 * \class HGCalRecHitDQM
 *
 * DQM client for HGCal rechits.
 * Reads rechits in SoA format and fills a per-module <E_nMIPs> vs channel profile from every hit.
 * Hits above 3 sigma noise go into per-layer energy and time histograms,
 * and into per-endcap summed energy and multiplicity vs layer.
 * It processes the first MinimumEvents events and then every PrescaleFactor-th event.
 */
class HGCalRecHitDQM : public DQMEDAnalyzer {
public:
  using MonitoredElement_t = hgcal::dqm::HGCalDQMModule;
  typedef std::pair<uint32_t, uint32_t> MonitoredElementKey_t;
  explicit HGCalRecHitDQM(const edm::ParameterSet&);
  ~HGCalRecHitDQM() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

  void analyze(const edm::Event&, const edm::EventSetup&) override;

  const edm::EDGetTokenT<HGCalSoARecHitsHostCollection> rechitsTkn_;
  edm::ESGetToken<hgcal::HGCalDenseIndexInfoHost, HGCalDenseIndexInfoRcd> denseIndexInfoTkn_;
  edm::ESGetToken<HGCalMappingModuleIndexer, HGCalElectronicsMappingRcd> moduleIdxTkn_;
  edm::ESGetToken<hgcal::HGCalMappingModuleParamHost, HGCalElectronicsMappingRcd> moduleInfoTkn_;
  const unsigned int minEvents_;
  const unsigned int prescaleFactor_;
  unsigned int nProcessed_;

  std::map<MonitoredElementKey_t, MonitoredElement_t> followedModules_;
  std::map<uint32_t, std::vector<std::pair<MonitoredElementKey_t, MonitoredElement_t>>> modulesByFED_;

  std::map<int, std::map<int, std::map<int, std::map<std::string, MonitoredElement_t>>>> HGCALMap;

  std::map<int, std::map<std::string, MonitorElement*>> recHitSummariesEndcaps_;
  std::map<int, std::map<int, std::map<std::string, MonitorElement*>>> recHitSummariesLayers_;
  // per-module TProfile: <E_nMIPs> vs channel; consumed by harvester as avgrechit_nmips.
  std::map<MonitoredElementKey_t, MonitorElement*> avgRechitNmips_;
  std::map<int, std::string> endCapKey = {{-1, "Minus"}, {1, "Plus"}};
};

//
// constructors and destructor
//

HGCalRecHitDQM::HGCalRecHitDQM(const edm::ParameterSet& iConfig)
    : rechitsTkn_(consumes<HGCalSoARecHitsHostCollection>(iConfig.getParameter<edm::InputTag>("RecHits"))),
      denseIndexInfoTkn_(esConsumes()),
      moduleIdxTkn_(esConsumes<edm::Transition::BeginRun>()),
      moduleInfoTkn_(esConsumes<edm::Transition::BeginRun>()),
      minEvents_(iConfig.getParameter<unsigned int>("MinimumEvents")),
      prescaleFactor_(std::max(1u, iConfig.getParameter<unsigned int>("PrescaleFactor"))),
      nProcessed_(0) {}

HGCalRecHitDQM::~HGCalRecHitDQM() {}

// ------------ method called for each event  ------------
void HGCalRecHitDQM::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  ++nProcessed_;

  bool toProcess = (nProcessed_ < minEvents_) || (nProcessed_ % prescaleFactor_ == 0);
  if (!toProcess)
    return;

  //read rechits and dense index info
  const auto& denseIndexInfo = iSetup.getData(denseIndexInfoTkn_);
  const auto& rechits = iEvent.getHandle(rechitsTkn_);
  if (!rechits.isValid())
    return;

  //number of hits and dense indices
  const auto& denseIndexInfo_view = denseIndexInfo.const_view();
  int32_t ndii = denseIndexInfo_view.metadata().size();
  const auto& rechits_view = rechits->const_view();
  int32_t nhits = rechits_view.metadata().size();
  assert(ndii >= nhits);

  //fill histograms
  constexpr float knoise_thr(3.0);

  //prepare sums
  std::map<int, std::map<int, float>> mipsum;
  std::map<int, std::map<int, int>> hitsum;
  for (const auto& [endcap, layerMap] : recHitSummariesLayers_) {
    for (const auto& [layer, _] : layerMap) {
      mipsum[endcap][layer] = 0.f;
      hitsum[endcap][layer] = 0;
    }
  }

  // Fill targets of the current (endcap, layer), refreshed only when it changes.
  int cur_endcap = std::numeric_limits<int>::min();
  int cur_layer = std::numeric_limits<int>::min();
  bool cur_ok = false;
  MonitorElement* h_energy = nullptr;
  MonitorElement* h_time = nullptr;
  MonitorElement* h_timevsE = nullptr;

  //loop over hits
  MonitoredElementKey_t cur_mod_key(std::numeric_limits<uint32_t>::max(), std::numeric_limits<uint32_t>::max());
  auto avgmips_it = avgRechitNmips_.end();
  for (int i = 0; i < nhits; i++) {
    const auto& rechit = rechits->const_view()[i];
    const auto& layer = rechit.layer();

    const auto& nmips = rechit.mipEnergy();
    const auto& time = rechit.time();

    const auto& denseIdx = rechit.recHitIndex();
    auto indexinfo = denseIndexInfo_view[denseIdx];

    // Filled before the noise/layer cut: the harvester expects an entry per channel per hit.
    MonitoredElementKey_t mod_key(indexinfo.fedId(), indexinfo.fedReadoutSeq());
    if (mod_key != cur_mod_key) {
      cur_mod_key = mod_key;
      avgmips_it = avgRechitNmips_.find(mod_key);
    }
    if (avgmips_it != avgRechitNmips_.end())
      avgmips_it->second->Fill(indexinfo.chNumber(), nmips);

    if (layer == 0)
      continue;
    if (rechit.energy() < knoise_thr * rechit.sigmaNoise())
      continue;

    auto endcap = indexinfo.z() < 0 ? -1 : 1;

    if (endcap != cur_endcap || static_cast<int>(layer) != cur_layer) {
      cur_endcap = endcap;
      cur_layer = static_cast<int>(layer);
      cur_ok = false;
      auto ec_it = recHitSummariesLayers_.find(endcap);
      if (ec_it == recHitSummariesLayers_.end())
        continue;
      auto lay_it = ec_it->second.find(layer);
      if (lay_it == ec_it->second.end())
        continue;
      const auto& mmap = lay_it->second;
      auto it_e = mmap.find("rechitenergy");
      auto it_t = mmap.find("rechittime");
      auto it_tvE = mmap.find("rechittimevsenergy");
      if (it_e == mmap.end() || it_t == mmap.end() || it_tvE == mmap.end())
        continue;
      h_energy = it_e->second;
      h_time = it_t->second;
      h_timevsE = it_tvE->second;
      cur_ok = true;
    }
    if (!cur_ok)
      continue;

    mipsum[endcap][layer] += nmips;
    hitsum[endcap][layer] += 1;
    h_energy->Fill(nmips);
    if (time > 0) {
      h_time->Fill(time);
      h_timevsE->Fill(nmips, time);
    }
  }

  //longitudinal profile versus layer
  for (const auto& [endcap, layerMap] : mipsum) {
    auto ec_it = recHitSummariesEndcaps_.find(endcap);
    if (ec_it == recHitSummariesEndcaps_.end())
      continue;
    const auto& mmap = ec_it->second;
    auto it_sum = mmap.find("rechitsumenergy");
    auto it_mult = mmap.find("rechitmultiplicity");
    if (it_sum == mmap.end() || it_mult == mmap.end())
      continue;
    for (const auto& [layer, _] : layerMap) {
      it_sum->second->Fill(layer, mipsum[endcap][layer]);
      it_mult->second->Fill(layer, hitsum[endcap][layer]);
    }
  }
}

void HGCalRecHitDQM::bookHistograms(DQMStore::IBooker& ibook, edm::Run const& run, edm::EventSetup const& iSetup) {
  const HGCalMappingModuleIndexer& moduleIndexer = iSetup.getData(moduleIdxTkn_);
  const hgcal::HGCalMappingModuleParamHost& moduleInfo = iSetup.getData(moduleInfoTkn_);

  for (auto ele : hgcal::dqm::readoutModules(moduleIndexer, moduleInfo)) {
    const uint32_t fedid = ele.fedid;
    const uint32_t imod = ele.modid;
    const std::string& typecode = ele.typecode;

    MonitoredElementKey_t key(fedid, imod);
    followedModules_[key] = ele;
    modulesByFED_[fedid].emplace_back(key, ele);

    auto& cassetteMap = HGCALMap[ele.endcap][ele.layer][ele.cassette];
    cassetteMap[typecode] = ele;
    cassetteMap[typecode].moduleIndex = cassetteMap.size() - 1;
  }

  // Per-module <E_nMIPs> vs channel TProfile. Folder must match HGCalDigiDQM's
  // per-module folder (harvester expects avgrechit_nmips alongside avgadc etc).
  for (const auto& [key, ele] : followedModules_) {
    std::string endcapStr = (ele.endcap == 1) ? "Plus" : "Minus";
    std::string folder = "HGCAL/EndCap_" + endcapStr + "/Layer_" + std::to_string(ele.layer) + "/Cassette_" +
                         std::to_string(ele.cassette) + "/(u" + std::to_string(ele.i1) + "-v" + std::to_string(ele.i2) +
                         ") " + ele.typecode;
    ibook.setCurrentFolder(folder);
    size_t nch = ele.nErx * 37;
    avgRechitNmips_[key] = ibook.bookProfile(
        "avgrechit_nmips", ele.typecode + ";Channel; <E_{nMIPs}>", nch, -0.5, nch - 0.5, 100, -100, 2000, "s");
  }

  for (const auto& [endcap, layerMap] : HGCALMap) {
    std::string endCapString = endCapKey[endcap];
    std::string endcapFolder = "HGCAL/EndCap_" + endCapString + "/";
    ibook.setCurrentFolder(endcapFolder);

    std::string label(endcap > 0 ? "CE+" : "CE-");
    recHitSummariesEndcaps_[endcap]["rechitsumenergy"] =
        ibook.book2D("rechitsumenergy", label + ";Layer; <Energy> [MIPs]", 47, 0.5, 47.5, 150, 0, 2500);

    recHitSummariesEndcaps_[endcap]["rechitmultiplicity"] =
        ibook.book2D("rechitmultiplicity", label + ";Layer; <#hits>", 47, 0.5, 47.5, 200, 0, 1500);

    for (const auto& [layer, _] : layerMap) {
      std::string layerFolder = endcapFolder + "Layer_" + std::to_string(layer) + "/";

      ibook.setCurrentFolder(layerFolder);

      std::string label("Layer " + std::to_string(layer));

      recHitSummariesLayers_[endcap][layer]["rechittime"] =
          ibook.book1D("rechittime", label + ";RecHit time [ps]; RecHits", 100, 0, 5000);
      recHitSummariesLayers_[endcap][layer]["rechitenergy"] =
          ibook.book1D("rechitenergy", label + ";RecHit energy [MIPs]; RecHits", 100, 0, 250);
      recHitSummariesLayers_[endcap][layer]["rechittimevsenergy"] =
          ibook.book2D("rechittimevsenergy", label + ";Energy [MIP]; RecHit time [ps]", 100, -10, 1000, 100, 0, 5000);
    }
  }
}

void HGCalRecHitDQM::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("RecHits", edm::InputTag("hgcalRecHits", ""));
  desc.add<unsigned int>("MinimumEvents", 5000);
  desc.add<unsigned int>("PrescaleFactor", 5000);
  desc.add<bool>("isSimulation", false);
  descriptions.add("hgcalrechitdqm", desc);
}

// define this as a plug-in
DEFINE_FWK_MODULE(HGCalRecHitDQM);
