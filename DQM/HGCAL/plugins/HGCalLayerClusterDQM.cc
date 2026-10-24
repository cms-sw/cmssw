#include <algorithm>
#include <map>
#include <numeric>
#include <string>
#include <vector>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DataFormats/HGCalReco/interface/HGCalSoAClustersHostCollection.h"
#include "DataFormats/HGCalReco/interface/HGCalSoARecHitsHostCollection.h"
#include "CondFormats/DataRecord/interface/HGCalElectronicsMappingRcd.h"
#include "CondFormats/HGCalObjects/interface/HGCalMappingModuleIndexer.h"
#include "CondFormats/HGCalObjects/interface/HGCalMappingParameterHost.h"
#include "DQM/HGCAL/interface/HGCalDQMCommon.h"

/**
 * \class HGCalLayerClusterDQM
 *
 * DQM client for HGCal layer clusters.
 * Reads layer clusters and rechits in SoA format
 * and uses each cluster's seed rechit for the layer and the MIP scale.
 * Clusters with one cell or fewer are skipped.
 * It fills per-layer energy (GeV and MIPs), size and position,
 * plus per-endcap summed energy, multiplicity and hits vs layer.
 * It processes the first MinimumEvents events and then every PrescaleFactor-th event.
 */
class HGCalLayerClusterDQM : public DQMEDAnalyzer {
public:
  using MonitoredElement_t = hgcal::dqm::HGCalDQMModule;
  typedef std::pair<uint32_t, uint32_t> MonitoredElementKey_t;

  explicit HGCalLayerClusterDQM(const edm::ParameterSet&);
  ~HGCalLayerClusterDQM() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

  void analyze(const edm::Event&, const edm::EventSetup&) override;
  std::string setEndCapFolder(DQMStore::IBooker& ibook, int endcap);

  // ------------ member data ------------
  const edm::EDGetTokenT<HGCalSoAClustersHostCollection> layerClustersToken_;
  const edm::EDGetTokenT<HGCalSoARecHitsHostCollection> rechitsTkn_;
  edm::ESGetToken<HGCalMappingModuleIndexer, HGCalElectronicsMappingRcd> moduleIdxTkn_;
  edm::ESGetToken<hgcal::HGCalMappingModuleParamHost, HGCalElectronicsMappingRcd> moduleInfoTkn_;
  const unsigned int minEvents_;
  const unsigned int prescaleFactor_;
  unsigned int nProcessed_;

  std::map<MonitoredElementKey_t, MonitoredElement_t> followedModules_;
  std::map<uint32_t, std::vector<std::pair<MonitoredElementKey_t, MonitoredElement_t>>> modulesByFED_;
  std::map<int, std::map<int, std::map<int, std::map<std::string, MonitoredElement_t>>>> HGCALMap;

  std::map<int, std::map<int, std::map<std::string, MonitorElement*>>> LCSummariesLayers_;
  std::map<int, std::map<std::string, MonitorElement*>> LCSummariesEndcaps_;
  std::map<int, std::string> endCapKey = {{-1, "Minus"}, {1, "Plus"}};
};

//
// constructors and destructor
//
HGCalLayerClusterDQM::HGCalLayerClusterDQM(const edm::ParameterSet& iConfig)
    : layerClustersToken_(
          consumes<HGCalSoAClustersHostCollection>(iConfig.getParameter<edm::InputTag>("layerClusters"))),
      rechitsTkn_(consumes<HGCalSoARecHitsHostCollection>(iConfig.getParameter<edm::InputTag>("RecHits"))),
      moduleIdxTkn_(esConsumes<edm::Transition::BeginRun>()),
      moduleInfoTkn_(esConsumes<edm::Transition::BeginRun>()),
      minEvents_(iConfig.getParameter<unsigned int>("MinimumEvents")),
      prescaleFactor_(std::max(1u, iConfig.getParameter<unsigned int>("PrescaleFactor"))),
      nProcessed_(0) {}

HGCalLayerClusterDQM::~HGCalLayerClusterDQM() {}

//
// member functions
//

// ------------ method called for each event  ------------
void HGCalLayerClusterDQM::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  ++nProcessed_;

  bool toProcess = (nProcessed_ < minEvents_) || (nProcessed_ % prescaleFactor_ == 0);
  if (!toProcess)
    return;

  //read LC and dense index info
  const auto& LCs = iEvent.getHandle(layerClustersToken_);
  if (!LCs.isValid())
    return;
  //number of hits and dense indices
  const auto& LCs_view = LCs->const_view();
  int32_t nLCs = LCs_view.metadata().size();
  const auto& rechits = iEvent.getHandle(rechitsTkn_);
  if (!rechits.isValid())
    return;
  const auto& rechits_view = rechits->const_view();
  int32_t nhits = rechits_view.metadata().size();
  //fill histograms

  //prepare sums
  std::map<int, std::map<int, float>> mipsum;
  std::map<int, std::map<int, int>> hitsum;
  std::map<int, std::map<int, int>> LCsum;
  std::map<int, std::map<int, int>> energysum;
  // For each layer has a vector of each LCs energy.
  std::map<int, std::map<int, std::vector<float>>> layerLCEnergy;

  for (const auto& [endcap, layerMap] : LCSummariesLayers_) {
    for (const auto& [layer, summaryMap] : layerMap) {
      mipsum[endcap][layer] = 0.f;
      hitsum[endcap][layer] = 0;
      LCsum[endcap][layer] = 0;
      energysum[endcap][layer] = 0;
    }
  }

  //loop over LCs
  for (int i = 0; i < nLCs; i++) {
    const auto& LC = LCs->const_view()[i];
    const int cells = LC.cells();
    if (cells <= 1) {
      continue;
    }
    // const auto& layer = LC.layer();
    const float x = LC.x();
    const float y = LC.y();
    const float z = LC.z();
    const int seed = LC.seed();
    if (seed < 0 || seed >= nhits)
      continue;
    const auto& rechit = rechits_view[seed];
    const int layer = rechit.layer();
    if (layer == 0)
      continue;

    //increment sums
    const float mip_scale = rechit.mipEnergy() / rechit.energy();
    auto& energy = LC.energy();
    const float nmips = energy * mip_scale;
    auto endcap = z < 0 ? -1 : 1;
    hitsum[endcap][layer] += cells;
    mipsum[endcap][layer] += nmips;
    energysum[endcap][layer] += energy;
    LCsum[endcap][layer] += 1;
    layerLCEnergy[endcap][layer].push_back(nmips);
    LCSummariesLayers_[endcap][layer]["LCenergy"]->Fill(energy);
    LCSummariesLayers_[endcap][layer]["LCMIPenergy"]->Fill(nmips);
    LCSummariesLayers_[endcap][layer]["LCnhits"]->Fill(cells);
    LCSummariesLayers_[endcap][layer]["LCx"]->Fill(x);
    LCSummariesLayers_[endcap][layer]["LCy"]->Fill(y);
    LCSummariesLayers_[endcap][layer]["LCz"]->Fill(z);
  }

  //longitudinal profile versus layer
  for (const auto& [endcap, layerMap] : LCSummariesLayers_) {
    for (const auto& [layer, _] : layerMap) {
      LCSummariesEndcaps_[endcap]["LCsumenergy"]->Fill(layer, energysum[endcap][layer]);
      LCSummariesEndcaps_[endcap]["LCMIPsumenergy"]->Fill(layer, mipsum[endcap][layer]);
      LCSummariesEndcaps_[endcap]["LCmultiplicity"]->Fill(layer, LCsum[endcap][layer]);
      LCSummariesEndcaps_[endcap]["LCnhits"]->Fill(layer, hitsum[endcap][layer]);
      auto& energyV = layerLCEnergy[endcap][layer];
      if (!energyV.empty()) {
        auto maxVal = *std::max_element(energyV.begin(), energyV.end());
        float sumVal = std::accumulate(energyV.begin(), energyV.end(), 0.f);
        LCSummariesLayers_[endcap][layer]["LCMaxenergy"]->Fill(maxVal);
        LCSummariesLayers_[endcap][layer]["LCMSumenergy"]->Fill(sumVal);
      }
    }
  }
}

void HGCalLayerClusterDQM::bookHistograms(DQMStore::IBooker& ibook,
                                          edm::Run const& run,
                                          edm::EventSetup const& iSetup) {
  const HGCalMappingModuleIndexer& moduleIndexer = iSetup.getData(moduleIdxTkn_);
  const hgcal::HGCalMappingModuleParamHost& moduleInfo = iSetup.getData(moduleInfoTkn_);

  for (auto ele : hgcal::dqm::readoutModules(moduleIndexer, moduleInfo)) {
    const uint32_t fedid = ele.fedid;
    const uint32_t imod = ele.modid;
    const std::string& typecode = ele.typecode;

    MonitoredElementKey_t key(fedid, imod);
    ele.fedModuleIndex = modulesByFED_[fedid].size();
    modulesByFED_[fedid].emplace_back(key, ele);
    followedModules_[key] = ele;

    auto& cassetteMap = HGCALMap[ele.endcap][ele.layer][ele.cassette];
    cassetteMap[typecode] = ele;
    cassetteMap[typecode].moduleIndex = cassetteMap.size() - 1;
  }

  //EndCap Level, no plots just for endcaps!
  for (const auto& endcapPair : HGCALMap) {
    int endcap = endcapPair.first;
    std::string endcapFolder = setEndCapFolder(ibook, endcap);
    std::string label(endcap > 0 ? "CE+" : "CE-");
    LCSummariesEndcaps_[endcap]["LCsumenergy"] =
        ibook.book2D("LCsumenergy", label + ";Layer; <Energy> [GeV]", 47, 0.5, 47.5, 250, 0, 50);
    LCSummariesEndcaps_[endcap]["LCMIPsumenergy"] =
        ibook.book2D("LCMIPsumenergy", label + ";Layer; <Energy> [MIPs]", 47, 0.5, 47.5, 150, 0, 3000);
    LCSummariesEndcaps_[endcap]["LCmultiplicity"] =
        ibook.book2D("LCmultiplicity", label + ";Layer; <#LC>", 47, 0.5, 47.5, 100, 0, 100);
    LCSummariesEndcaps_[endcap]["LCnhits"] =
        ibook.book2D("LCnhits", label + ";Layer; <#nHits>", 47, 0.5, 47.5, 100, 0, 2000);
    // Layer level, we need layer histograms with cassete on x axis: Basic plots reproduced
    const auto& layerMap = endcapPair.second;
    for (const auto& layerPair : layerMap) {
      int layer = layerPair.first;
      //const auto& cassetteMap = layerPair.second;
      std::string layerFolder = endcapFolder + "Layer_" + std::to_string(layer) + "/";
      ibook.setCurrentFolder(layerFolder);

      //rec hits per layer
      std::string label("Layer " + std::to_string(layer));
      LCSummariesLayers_[endcap][layer]["LCenergy"] =
          ibook.book1D("LCenergy", label + ";LC energy [GeV]; LCs", 500, 0, 50);
      LCSummariesLayers_[endcap][layer]["LCMIPenergy"] =
          ibook.book1D("LCMIPenergy", label + ";LC energy [MIPs]; LCs", 500, 0, 3000);
      // If there is more than one LC in a layer it will take the highest in energy.
      LCSummariesLayers_[endcap][layer]["LCMaxenergy"] =
          ibook.book1D("LCMaxenergy", label + ";Highest energy LC per Layer [GeV]; LCs", 500, 0, 50);
      // Sums all the LC energy in a layer.
      LCSummariesLayers_[endcap][layer]["LCMSumenergy"] =
          ibook.book1D("LCMSumenergy", label + ";Sum of all LC energy per Layer [GeV]; LCs", 500, 0, 50);
      LCSummariesLayers_[endcap][layer]["LCnhits"] = ibook.book1D("LCnhits", label + ";LC nHits; LCs", 250, 0, 250);
      LCSummariesLayers_[endcap][layer]["LCx"] = ibook.book1D("LCx", label + ";LC x; LCs", 1000, 50, 120);
      LCSummariesLayers_[endcap][layer]["LCy"] = ibook.book1D("LCy", label + ";LC y; LCs", 1000, 50, 120);
      LCSummariesLayers_[endcap][layer]["LCz"] = ibook.book1D("LCz", label + ";LC z; LCs", 100, -400, 400);
    }
  }
}

std::string HGCalLayerClusterDQM::setEndCapFolder(DQMStore::IBooker& ibook, int endcap) {
  std::string endCapString = endCapKey[endcap];
  std::string endcapFolder = std::string("HGCAL/") + "EndCap_" + endCapString + "/";
  ibook.setCurrentFolder(endcapFolder);
  return endcapFolder;
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void HGCalLayerClusterDQM::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("RecHits", edm::InputTag("hgcalRecHits", ""));
  desc.add<edm::InputTag>("layerClusters", edm::InputTag("hgcalSoALayerClusters", ""))
      ->setComment("Source of Rec Hit Layer Clusters info SoA");
  desc.add<unsigned int>("MinimumEvents", 10000);
  desc.add<unsigned int>("PrescaleFactor", 1);
  descriptions.add("hgcallayerclusterdqm", desc);
}

// define this as a plug-in
DEFINE_FWK_MODULE(HGCalLayerClusterDQM);
