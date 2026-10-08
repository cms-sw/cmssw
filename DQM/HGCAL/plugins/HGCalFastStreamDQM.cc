#include <atomic>
#include <string>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQM/HGCAL/interface/HGCalDQMCommon.h"
#include "DataFormats/HGCalDigi/interface/HGCalDigiHost.h"
#include "DataFormats/HGCalDigi/interface/HGCalECONDPacketInfoSoA.h"
#include "DataFormats/HGCalDigi/interface/HGCalECONDPacketInfoHost.h"
#include "DataFormats/HGCalDigi/interface/HGCalFEDPacketInfoSoA.h"
#include "DataFormats/HGCalDigi/interface/HGCalFEDPacketInfoHost.h"
#include "DataFormats/HGCalDigi/interface/HGCalRawDataDefinitions.h"
#include "DataFormats/HGCalReco/interface/HGCalSoARecHitsHostCollection.h"
#include "DataFormats/HGCalDigi/interface/HGCalDigiTriggerHost.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "CondFormats/HGCalObjects/interface/HGCalMappingModuleIndexer.h"
#include "CondFormats/DataRecord/interface/HGCalElectronicsMappingRcd.h"
#include "CondFormats/HGCalObjects/interface/HGCalMappingParameterHost.h"

#include <TFile.h>
#include <TTree.h>

using namespace edm;
using namespace hgcal::dqm;

/**
 * \class HGCalFastStreamDQM
 *
 * DQM client for the HGCal fast stream, run on every event.
 * Reads FED and ECON-D packet info and fills
 * - FED unpacking flags and payload,
 * - ECON-D quality and payload per cassette and per FED,
 * - per-module common-mode profiles,
 * - and BX/L1A/orbit comparisons between CB, ECON-D and S-link.
 * It also books event info and an ECON-D error vs layer histogram that it resets each lumisection.
 */
class HGCalFastStreamDQM : public DQMEDAnalyzer {
public:
  explicit HGCalFastStreamDQM(const edm::ParameterSet&);
  ~HGCalFastStreamDQM() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  using MonitoredElement_t = hgcal::dqm::HGCalDQMModule;

  struct HistogramConfig {
    std::string name, title;
    int nBins;
    double xMin, xMax;
    int nBinsY = 0;  // For 2D histograms
    double yMin = 0, yMax = 0;
    bool is2D = false;
  };

  typedef std::pair<uint32_t, uint32_t> MonitoredElementKey_t;

private:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void bookHistogramsFastStream(DQMStore::IBooker& ibook, edm::Run const& run, edm::EventSetup const& iSetup);
  void bookPerFEDComparisonHistograms(DQMStore::IBooker& ibook);
  void bookEventInfo(DQMStore::IBooker& ibooker);
  void bookLSSummary(DQMStore::IBooker& ibooker);

  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void analyzeECONDFlags(const edm::Event& iEvent, const edm::EventSetup& iSetup);
  void analyzeFEDFlags(const edm::Event& iEvent, const edm::EventSetup& iSetup);
  void analyzeBXComparison(const edm::Event& iEvent, const edm::EventSetup& iSetup);
  void analyzeLSFastStream(const edm::Event& iEvent, const edm::EventSetup& iSetup);

  void fillPerFEDComparisonHistograms(uint32_t fedid,
                                      const std::map<std::string, std::map<std::string, double>>& values);
  void recordEventInfo(const edm::Event& iEvent);
  void iterateEcondLSCounts(int layer, int flag);

  // ------------ member data ------------
  edm::LuminosityBlockNumber_t currentLS = -1;

  MonitorElement *runNumberME, *lumiSectionME, *eventNumberME, *runStartME, *timeStampME;
  MonitorElement* me_econd_quality_layer;
  MonitorElement* fedSummaryME;
  MonitorElement *fedQualityH_, *fedPayload2D_, *econdQualityH_, *econdPayload_;

  edm::ESGetToken<HGCalMappingModuleIndexer, HGCalElectronicsMappingRcd> moduleIdxTkn_;
  const edm::EDGetTokenT<hgcaldigi::HGCalECONDPacketInfoHost> econdInfoTkn_;
  const edm::EDGetTokenT<FEDRawDataCollection> fedRawToken_;

  std::map<MonitoredElementKey_t, MonitoredElement_t> followedModules_;
  std::map<uint32_t, std::vector<std::pair<MonitoredElementKey_t, MonitoredElement_t>>> modulesByFED_;

  std::map<uint32_t, int> fedIdToBinMap_;

  std::vector<uint32_t> followedFEDs_;

  std::map<int, std::map<int, std::map<int, MonitorElement*>>> econdQualityCassettes_;
  std::map<int, std::map<int, std::map<int, MonitorElement*>>> econdPayloadCassettes_;

  std::set<int> unique_directionallayers;

  std::map<int, std::map<int, std::map<int, std::map<std::string, MonitoredElement_t>>>> HGCALMap;

  std::map<std::string, std::map<MonitoredElementKey_t, MonitorElement*>> moduleHistos_;
  const edm::EDGetTokenT<hgcaldigi::HGCalFEDPacketInfoHost> fedInfoTkn_;
  edm::ESGetToken<hgcal::HGCalMappingModuleParamHost, HGCalElectronicsMappingRcd> moduleInfoTkn_;
  std::map<uint32_t, std::map<std::string, std::map<std::string, MonitorElement*>>> fedComparisonHistos_;
  std::map<uint32_t, MonitorElement*> econdQualityFEDs_;

  std::map<int, std::string> endCapKey = {{-1, "Minus"}, {1, "Plus"}};

  std::vector<std::pair<std::string, std::string>> comparisons = {{"CB", "ECOND"}, {"CB", "SLINK"}, {"ECOND", "SLINK"}};

  std::map<std::string, HistogramConfig> diffConfigs = {{"BxDiff", {"BxDiff", "BX", 11, -5.5, 5.5}},
                                                        {"L1aDiff", {"L1aDiff", "L1A", 11, -5.5, 5.5}},
                                                        {"OrbitDiff", {"OrbitDiff", "Orbit", 11, -5.5, 5.5}}};

  std::map<std::string, HistogramConfig> corrConfigs = {
      {"BxCorr", {"BxCorr", "BX", 64, -0.5, 4095.5, 64, -0.5, 4095.5, true}},
      {"L1aCorr", {"L1aCorr", "L1A", 64, -0.5, 63.5, 64, -0.5, 63.5, true}},
      {"OrbitCorr", {"OrbitCorr", "Orbit", 8, -0.5, 7.5, 8, -0.5, 7.5, true}}};
};

//
// constructors and destructor
//
HGCalFastStreamDQM::HGCalFastStreamDQM(const edm::ParameterSet& iConfig)
    : moduleIdxTkn_(esConsumes<edm::Transition::BeginRun>()),
      econdInfoTkn_(
          consumes<hgcaldigi::HGCalECONDPacketInfoHost>(iConfig.getParameter<edm::InputTag>("ECONDPacketInfo"))),
      fedRawToken_(consumes<FEDRawDataCollection>(iConfig.getParameter<edm::InputTag>("Raw"))),
      fedInfoTkn_(consumes<hgcaldigi::HGCalFEDPacketInfoHost>(iConfig.getParameter<edm::InputTag>("FEDPacketInfo"))),
      moduleInfoTkn_(esConsumes<edm::Transition::BeginRun>()) {}

HGCalFastStreamDQM::~HGCalFastStreamDQM() {}

//
// member functions
//

// ------------ method called for each event  ------------
void HGCalFastStreamDQM::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  recordEventInfo(iEvent);
  analyzeLSFastStream(iEvent, iSetup);
  analyzeECONDFlags(iEvent, iSetup);
  analyzeFEDFlags(iEvent, iSetup);
  analyzeBXComparison(iEvent, iSetup);
}

// Fast Stream: Fills the Fed error histos.
void HGCalFastStreamDQM::analyzeFEDFlags(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  //read module info flagged ECON-D list
  const auto& fedInfo = iEvent.getHandle(fedInfoTkn_);  // FED packet flags

  if (!fedInfo.isValid())
    return;

  for (auto fedid : followedFEDs_) {
    const auto fed = fedInfo->const_view()[fedid];

    int fedBinX = fedIdToBinMap_[fedid];
    fedPayload2D_->Fill(fedBinX, fed.FEDPayload() * 0.0625);

    auto flags = fed.FEDUnpackingFlag();
    if (hgcaldigi::isNotNormalFED(flags))
      fedQualityH_->Fill(fedBinX, 0);
    if (hgcaldigi::hasGenericUnpackError(flags))
      fedQualityH_->Fill(fedBinX, 1);
    if (hgcaldigi::hasHeaderUnpackError(flags))
      fedQualityH_->Fill(fedBinX, 2);
    if (hgcaldigi::hasPayloadUnpackError(flags))
      fedQualityH_->Fill(fedBinX, 3);
    if (hgcaldigi::hasCBHeaderError(flags))
      fedQualityH_->Fill(fedBinX, 4);
    if (hgcaldigi::hasCBActiveFlags(flags))
      fedQualityH_->Fill(fedBinX, 5);
    if (hgcaldigi::hasErrorECONDHeader(flags))
      fedQualityH_->Fill(fedBinX, 6);
    if (hgcaldigi::hasECONDPayloadLengthOverflow(flags))
      fedQualityH_->Fill(fedBinX, 7);
    if (hgcaldigi::hasECONDPayloadLengthMismatch(flags))
      fedQualityH_->Fill(fedBinX, 8);
    if (hgcaldigi::hasErrorSLinkTrailer(flags))
      fedQualityH_->Fill(fedBinX, 9);
    if (hgcaldigi::hasEarlySLinkEnd(flags))
      fedQualityH_->Fill(fedBinX, 10);
  }
}

// Fast Stream: Fills the EconD and CB error histos at layer + cassette level
void HGCalFastStreamDQM::analyzeECONDFlags(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  //read module info flagged ECON-D list
  const auto& econdInfo = iEvent.getHandle(econdInfoTkn_);  // ECON-D packet flags
  if (!econdInfo.isValid())
    return;
  const auto& raw_data = iEvent.getHandle(fedRawToken_);

  // first fill module level
  for (const auto& [key, mod] : followedModules_) {
    int endcap = mod.endcap;
    int layer = mod.layer;
    int directionalLayer = layer * endcap;
    int cassette = mod.cassette;
    std::string typecode = mod.typecode;

    const std::string& binlabels = typecode;
    auto binlabel = binlabels.c_str();
    int bin_id = HGCALMap[endcap][layer][cassette][typecode].moduleIndex;
    econdQualityH_ = econdQualityCassettes_[endcap][layer][cassette];
    econdPayload_ = econdPayloadCassettes_[endcap][layer][cassette];
    econdQualityH_->setBinLabel(bin_id + 1, binlabel, 1);

    const uint32_t imod = mod.dqmIndex;  // global/dense module index
    const auto econd = econdInfo->const_view()[imod];
    econdPayload_->Fill(econd.payloadLength());

    // error flags
    std::vector<int> errorBin = getErrorBinsForECONDCBFlags(
        (unsigned int)econd.cbFlag(), (unsigned int)econd.econdFlag(), (unsigned int)econd.exception());

    for (int errorID : errorBin) {
      econdQualityH_->Fill(bin_id, errorID - 0.5);
      iterateEcondLSCounts(directionalLayer, errorID - 0.5);
    }

    // HGCalUnpacker leaves the CM matrix unset when it rejects the payload.
    // Apply its payload-quality gate before reading the matrix, and fill a zero
    // sample per channel for these events.
    // A BCID/Orbit mismatch alone can still have decoded CM in passthrough mode.
    const bool hasCommonMode = hgcaldigi::htFlag(econd.econdFlag()) < 2 && hgcaldigi::eboFlag(econd.econdFlag()) < 2 &&
                               hgcaldigi::matchFlag(econd.econdFlag()) && econd.payloadLength() > 0 &&
                               econd.cbFlag() != hgcal::backend::ECONDPacketStatus::OfflinePayloadCRCError &&
                               econd.cbFlag() != hgcal::backend::ECONDPacketStatus::InactiveECOND;
    for (uint32_t erxIdx = 0; erxIdx < mod.nErx; ++erxIdx) {
      uint16_t cm0 = hasCommonMode ? econd.cm()(erxIdx, 0) : 0;
      uint16_t cm1 = hasCommonMode ? econd.cm()(erxIdx, 1) : 0;
      uint32_t idx = erxIdx * 2;
      moduleHistos_["cmChannels"][key]->Fill(idx, cm0);
      moduleHistos_["cmChannels"][key]->Fill(idx + 1, cm1);
    }

  }  // end of loop over followed modules
}

// validation of BX/L1A/Orbit numbers recorded in FED and ECON-D packets
void HGCalFastStreamDQM::analyzeBXComparison(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // Read ECOND information
  const auto& econdInfo = iEvent.getHandle(econdInfoTkn_);
  if (!econdInfo.isValid())
    return;

  // Read FED information
  const auto& fedInfo = iEvent.getHandle(fedInfoTkn_);
  if (!fedInfo.isValid())
    return;

  // Process each FED
  for (const auto& [fedid, modules] : modulesByFED_) {
    // Get FED/SLINK information once per FED
    const auto fed = fedInfo->const_view()[fedid];
    uint16_t slinkBx = fed.FEDBX();
    uint64_t slinkL1a = fed.FEDL1A() & hgcal::ECOND_FRAME::L1A_MASK;
    uint32_t slinkOrbit = fed.FEDOrbit() & hgcal::ECOND_FRAME::ORBIT_MASK;

    // Collect all CB and ECOND values for this FED
    std::vector<std::map<std::string, double>> fedValues;

    for (const auto& [key, mod] : modules) {
      const uint32_t imod = mod.dqmIndex;

      // Get ECOND/CB information for this module
      const auto econd = econdInfo->const_view()[imod];
      uint16_t econdBx = econd.BX();
      uint8_t econdL1a = econd.L1A();
      uint8_t econdOrbit = econd.Orbit();

      uint16_t cbBx = econd.CBBX();
      uint8_t cbL1a = econd.CBL1A();
      uint8_t cbOrbit = econd.CBOrbit();

      // Store values for this module
      std::map<std::string, std::map<std::string, double>> moduleValues = {
          {"CB", {{"Bx", cbBx}, {"L1a", cbL1a}, {"Orbit", cbOrbit}}},
          {"ECOND", {{"Bx", econdBx}, {"L1a", econdL1a}, {"Orbit", econdOrbit}}},
          {"SLINK", {{"Bx", slinkBx}, {"L1a", slinkL1a}, {"Orbit", slinkOrbit}}}};

      fillPerFEDComparisonHistograms(fedid, moduleValues);

      //--------------------------------------------------
      // Fill ECON-D Quality per FED
      //--------------------------------------------------
      std::string typecode = mod.typecode;
      auto binlabel = typecode.c_str();
      int bin_id = mod.fedModuleIndex;
      econdQualityFEDs_[fedid]->setBinLabel(bin_id + 1, binlabel, 1);

      std::vector<int> errorBin = getErrorBinsForECONDCBFlags(
          (unsigned int)econd.cbFlag(), (unsigned int)econd.econdFlag(), (unsigned int)econd.exception());
      for (int errorID : errorBin) {
        econdQualityFEDs_[fedid]->Fill(bin_id, errorID - 0.5);
      }
    }
  }
}

void HGCalFastStreamDQM::fillPerFEDComparisonHistograms(
    uint32_t fedid, const std::map<std::string, std::map<std::string, double>>& values) {
  // Check once per FED at the start of the comparison loop
  bool fedHistosExist = (fedComparisonHistos_.find(fedid) != fedComparisonHistos_.end());
  if (!fedHistosExist)
    return;  // Skip entire FED if histos don't exist

  // Fill per-FED histograms
  // IMPORTANT: Iterate in the same order as booking to match Y-labels
  int yBinCounter = 0;  // Track y-bin position
  for (const auto& [comp1, comp2] : comparisons) {
    std::string compKey = comp1 + comp2;

    for (const auto& [metricKey, config] : diffConfigs) {                // BxDiff, L1aDiff, OrbitDiff
      std::string metric = metricKey.substr(0, metricKey.length() - 4);  // "BxDiff" -> "Bx"
      double diff = values.at(comp1).at(metric) - values.at(comp2).at(metric);
      fedComparisonHistos_[fedid][compKey][metricKey]->Fill(diff);

      // Fill summary histogram if there's a mismatch
      if (diff != 0) {
        int fedBinX = fedIdToBinMap_[fedid];
        fedSummaryME->Fill(fedBinX, yBinCounter);
      }

      yBinCounter++;  // Increment for each metric in the same order as booking
    }

    // Fill correlation histograms
    std::vector<std::string> metric_list = {"Bx", "L1a", "Orbit"};
    for (const auto& metric : metric_list) {
      std::string corrKey = metric + "Corr";
      if (fedComparisonHistos_[fedid][compKey].find(corrKey) != fedComparisonHistos_[fedid][compKey].end()) {
        fedComparisonHistos_[fedid][compKey][corrKey]->Fill(values.at(comp1).at(metric), values.at(comp2).at(metric));
      }
    }
  }
}

void HGCalFastStreamDQM::bookHistograms(DQMStore::IBooker& ibook, edm::Run const& run, edm::EventSetup const& iSetup) {
  // Book event information
  bookEventInfo(ibook);

  // Fetch module index and module info from the EventSetup
  const HGCalMappingModuleIndexer& moduleIndexer = iSetup.getData(moduleIdxTkn_);
  const hgcal::HGCalMappingModuleParamHost& moduleInfo = iSetup.getData(moduleInfoTkn_);

  // Loop oveer available FEDs
  int binIndex = 0;
  for (const auto& fed : moduleIndexer.fedReadoutSequences()) {
    if (fed.totalECONs_ == 0)
      continue;
    followedFEDs_.push_back(fed.id);
    fedIdToBinMap_[fed.id] = binIndex;
    binIndex++;
  }

  // Loop over available modules and select those to track
  for (auto ele : hgcal::dqm::readoutModules(moduleIndexer, moduleInfo)) {
    const uint32_t fedid = ele.fedid;
    const uint32_t imod = ele.modid;
    const std::string& typecode = ele.typecode;
    unique_directionallayers.insert(ele.zside ? static_cast<int>(ele.layer) : -static_cast<int>(ele.layer));

    // Store in maps
    MonitoredElementKey_t key(fedid, imod);
    ele.fedModuleIndex = modulesByFED_[fedid].size();
    modulesByFED_[fedid].emplace_back(key, ele);
    followedModules_[key] = ele;

    // Build hierarchical HGCALMap: endcap -> layer -> cassette -> typecode
    auto& cassetteMap = HGCALMap[ele.endcap][ele.layer][ele.cassette];
    cassetteMap[typecode] = ele;
    cassetteMap[typecode].moduleIndex = cassetteMap.size() - 1;
  }

  bookPerFEDComparisonHistograms(ibook);
  bookHistogramsFastStream(ibook, run, iSetup);
}

void HGCalFastStreamDQM::bookHistogramsFastStream(DQMStore::IBooker& ibook,
                                                  edm::Run const& run,
                                                  edm::EventSetup const& iSetup) {
  size_t necondWithCBflags = hgcal::dqm::econdWithCBflags.size();

  //EndCap Level, no plots just for endcaps!
  for (const auto& endcapPair : HGCALMap) {
    int endcap = endcapPair.first;
    std::string endCapString = endCapKey[endcap];
    std::string endcapFolder = std::string("HGCAL/") + "EndCap_" + endCapString + "/";
    ibook.setCurrentFolder(endcapFolder);

    // Layer level, we need layer histograms with cassete on x axis: Basic plots reproduced
    const auto& layerMap = endcapPair.second;
    for (const auto& layerPair : layerMap) {
      int layer = layerPair.first;
      const auto& cassetteMap = layerPair.second;
      std::string layerFolder = endcapFolder + "Layer_" + std::to_string(layer) + "/";
      ibook.setCurrentFolder(layerFolder);

      //cassette level plots
      for (const auto& cassettePair : cassetteMap) {
        int cassette = cassettePair.first;
        // add bin to layer hist.
        const auto& econdMap = cassettePair.second;
        std::string cassetteFolder = layerFolder + "Cassette_" + std::to_string(cassette) + "/";
        int nModules = econdMap.size();
        ibook.setCurrentFolder(cassetteFolder);
        // Highest econd error per module for a given cassette either 1D or TH2 Poly
        // deatiled ECOND per module for given cassette
        econdQualityCassettes_[endcap][layer][cassette] =
            ibook.book2D("econdQualityCassette_" + std::to_string(cassette),
                         ";ECON-D;Header quality;",
                         nModules,
                         0,
                         nModules,
                         necondWithCBflags,
                         0,
                         necondWithCBflags);
        econdPayloadCassettes_[endcap][layer][cassette] =
            ibook.book1D("econdPayloadCassette_" + std::to_string(cassette), ";ECON-D;Payload", 480, 0, 480);
        hgcal::dqm::addBinLabels(econdWithCBflags, econdQualityCassettes_[endcap][layer][cassette], 2);
      }
    }
  }

  for (const auto& [key, mod] : followedModules_) {
    std::string typecode = mod.typecode;
    int cassette = mod.cassette;
    int layer = mod.layer;
    int endcap = mod.endcap;
    int u_coordinate = mod.i1;
    int v_coordinate = mod.i2;
    std::string uvStr = "(u" + std::to_string(u_coordinate) + "-v" + std::to_string(v_coordinate) + ") ";
    std::string endCapStr = endCapKey[endcap];
    std::string plotFolder = "HGCAL/EndCap_" + endCapStr + "/Layer_" + std::to_string(layer) + "/Cassette_" +
                             std::to_string(cassette) + "/" + uvStr + typecode;
    ibook.setCurrentFolder(plotFolder);

    //Per module histograms
    size_t ncm = 24;  // at most 24 common-mode channels

    // Use "s" option to show data spread (RMS) in TProfile
    // Other options: "" (sigma/sqrt{N}), "i" (integral), "g" (weighted)
    // See: https://root.cern.ch/doc/master/TProfileHelper_8h_source.html#l00693
    moduleHistos_["cmChannels"][key] =
        ibook.bookProfile("cmChannels", typecode + ";Channel; <CM>", ncm, -0.5, ncm - 0.5, 100, 0, 1024, "s");
  }

  // Book versus LS plots
  bookLSSummary(ibook);
}

void HGCalFastStreamDQM::recordEventInfo(const edm::Event& iEvent) {
  auto iRun = iEvent.id().run();
  auto iEvt = iEvent.id().event();
  auto iLumi = iEvent.luminosityBlock();
  runNumberME->Fill(iRun);
  lumiSectionME->Fill(iLumi);
  eventNumberME->Fill(iEvt);
}

// Fast Stream: Plots versus LS plots.
void HGCalFastStreamDQM::analyzeLSFastStream(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::LuminosityBlockNumber_t lumi = iEvent.luminosityBlock();
  if (lumi != currentLS) {
    me_econd_quality_layer->Reset();
  }
  currentLS = lumi;
}

void HGCalFastStreamDQM::bookEventInfo(DQMStore::IBooker& ibooker) {
  ibooker.setCurrentFolder("HGCAL/EventInfo");
  runNumberME = ibooker.bookInt("iRun");                // INT
  lumiSectionME = ibooker.bookInt("iLumiSection");      // INT
  eventNumberME = ibooker.bookInt("iEvent");            // INT
  runStartME = ibooker.bookFloat("runStartTimeStamp");  // REAL

  // Shared by all stream instances: only the first one records the run-start time stamp.
  static std::atomic<bool> timeRecorded{false};
  if (!timeRecorded.exchange(true)) {
    auto now = std::chrono::system_clock::now();
    std::time_t timestamp = std::chrono::system_clock::to_time_t(now);
    double unixTimestamp = static_cast<double>(timestamp);
    runStartME->Fill(unixTimestamp);

    TDatime dt(timestamp);
    TString timeString = dt.AsSQLString();
    timeStampME = ibooker.bookString("timeStamp", timeString);
  }
}

void HGCalFastStreamDQM::bookLSSummary(DQMStore::IBooker& ibooker) {
  // Set to HGCAL folder
  ibooker.setCurrentFolder("HGCAL");
  size_t necondWithCBflags = hgcal::dqm::econdWithCBflags.size();
  int nLayers = unique_directionallayers.size();
  std::vector<std::string> layer_labels;
  layer_labels.reserve(unique_directionallayers.size());
  for (int n : unique_directionallayers) {
    layer_labels.push_back(std::to_string(n));
  }
  // Book Error FLag versus Layer, this is cleared for each LS.
  me_econd_quality_layer =
      ibooker.book2D("econd_lastLS", ";Layer;Error;", nLayers, 0, nLayers, necondWithCBflags, 0, necondWithCBflags);
  hgcal::dqm::addBinLabels(layer_labels, me_econd_quality_layer, 1);
  hgcal::dqm::addBinLabels(hgcal::dqm::econdWithCBflags, me_econd_quality_layer, 2);
}

void HGCalFastStreamDQM::iterateEcondLSCounts(int layer, int flag) {
  // Adds an error flag to the flag versus layer plot.
  int layer_id = unique_directionallayers.count(layer)
                     ? std::distance(unique_directionallayers.begin(), unique_directionallayers.find(layer))
                     : -1;
  me_econd_quality_layer->Fill(layer_id, flag);
}

void HGCalFastStreamDQM::bookPerFEDComparisonHistograms(DQMStore::IBooker& ibook) {
  LogInfo("HGCalFastStreamDQM") << "Booking comparison histograms for " << followedFEDs_.size() << " FEDs with "
                                << followedModules_.size() << " total modules";

  std::vector<std::string> xlabels, ylabels;

  std::vector<std::string> fedflags = {"Error",
                                       "Generic",
                                       "S-link header",
                                       "Payload",
                                       "CB header",
                                       "CB flags",
                                       "ECON-D header",
                                       "ECON-D OF",
                                       "ECON-D Mismatch",
                                       "S-link trailer",
                                       "S-link end"};

  // Y-axis: Comparison types with readable labels
  for (const auto& [comp1, comp2] : comparisons) {
    for (const auto& [metricKey, config] : diffConfigs) {
      std::string readableLabel = config.title + "(" + comp1 + "/" + comp2 + ")";
      ylabels.push_back(readableLabel);
    }
  }

  // Book the FED summary histogram
  ibook.setCurrentFolder("HGCAL/FED");
  size_t nfedflags = fedflags.size();
  size_t nFEDs = followedFEDs_.size();
  size_t nComparisons = ylabels.size();
  size_t necondWithCBflags = hgcal::dqm::econdWithCBflags.size();

  fedSummaryME = ibook.book2D("fedSummaryMismatches",
                              "Counter Mismatches Summary;FED;Comparison Type;Mismatch Count",
                              nFEDs,
                              0,
                              nFEDs,
                              nComparisons,
                              0,
                              nComparisons);

  fedQualityH_ = ibook.book2D("fedQualityH", ";FED;Data quality;", nFEDs, 0, nFEDs, nfedflags, 0, nfedflags);
  fedPayload2D_ = ibook.book2D(
      "fedPayload", ";FED;Payload [128b words]", nFEDs, -0.5, static_cast<double>(nFEDs) - 0.5, 100, 0, 4000);

  // Book per-FED histograms
  for (uint32_t fedid : followedFEDs_) {
    // X-axis: collect FED IDs for labels
    xlabels.push_back("FED " + std::to_string(fedid));

    // Create FED-specific folder
    std::string fedFolder = "HGCAL/FED/FED_" + std::to_string(fedid) + "/";
    ibook.setCurrentFolder(fedFolder);

    // Register MEs for BX/L1A/Orb per FED
    for (const auto& [comp1, comp2] : comparisons) {
      std::string compKey = comp1 + comp2;
      std::string compKeyLower = compKey;
      std::transform(compKeyLower.begin(), compKeyLower.end(), compKeyLower.begin(), ::tolower);

      // Book difference histograms for this FED
      for (const auto& [metricKey, config] : diffConfigs) {
        std::string histName = compKeyLower + config.name + "_FED" + std::to_string(fedid);
        std::string histTitle = "FED " + std::to_string(fedid) + ";" + comp1 + " " + config.title + " - " + comp2 +
                                " " + config.title + ";Counts";

        fedComparisonHistos_[fedid][compKey][metricKey] =
            ibook.book1D(histName, histTitle, config.nBins, config.xMin, config.xMax);
      }

      // Book correlation histograms for this FED
      for (const auto& [metricKey, config] : corrConfigs) {
        std::string histName = compKeyLower + config.name + "_FED" + std::to_string(fedid);
        std::string histTitle = "FED " + std::to_string(fedid) + ";" + comp1 + " " + config.title + ";" + comp2 + " " +
                                config.title + ";Counts";

        fedComparisonHistos_[fedid][compKey][metricKey] = ibook.book2D(
            histName, histTitle, config.nBins, config.xMin, config.xMax, config.nBinsY, config.yMin, config.yMax);
      }
    }

    // Register ME for ECON-D Quality per FED
    ibook.setCurrentFolder(fedFolder);
    size_t nModules = modulesByFED_[fedid].size();
    econdQualityFEDs_[fedid] = ibook.book2D("econdQualityFED_" + std::to_string(fedid),
                                            ";ECON-D;Header quality;",
                                            nModules,
                                            0,
                                            nModules,
                                            necondWithCBflags,
                                            0,
                                            necondWithCBflags);
    hgcal::dqm::addBinLabels(econdWithCBflags, econdQualityFEDs_[fedid], 2);
  }

  // Set descriptive labels
  hgcal::dqm::addBinLabels(xlabels, fedSummaryME, 1);
  hgcal::dqm::addBinLabels(ylabels, fedSummaryME, 2);
  hgcal::dqm::addBinLabels(xlabels, fedQualityH_, 1);
  hgcal::dqm::addBinLabels(fedflags, fedQualityH_, 2);
  hgcal::dqm::addBinLabels(xlabels, fedPayload2D_, 1);
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void HGCalFastStreamDQM::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("Raw", edm::InputTag("rawDataCollector", ""));
  desc.add<edm::InputTag>("ECONDPacketInfo", edm::InputTag("hgcalDigis", ""));  //ECON-D flags
  desc.add<edm::InputTag>("FEDPacketInfo", edm::InputTag("hgcalDigis", ""));    //UnpackerFlags

  descriptions.add("hgcalfaststreamdqm", desc);
}

// define this as a plug-in
DEFINE_FWK_MODULE(HGCalFastStreamDQM);
