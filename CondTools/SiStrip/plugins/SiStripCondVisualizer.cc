// -*- C++ -*-
//
// Package:    CondTools/SiStrip
// Class:      SiStripNoiseVisualizer
//
/**\class SiStripNoiseVisualizer SiStripNoiseVisualizer.cc
 CondTools/SiStrip/plugins/SiStripNoiseVisualizer.cc

 Description:
    Creates a ROOT file with per-module profiles of Noise and Pedestals vs strip
 number

*/
//
// Original Author:  Marco Musich
//         Created:  Thu, 05 Apr 2018 15:32:25 GMT
//
//

// system include files
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

// user include files
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"
#include "CalibTracker/Records/interface/SiStripQualityRcd.h"
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "CommonTools/UtilAlgos/interface/DetIdSelector.h"
#include "CondFormats/DataRecord/interface/SiStripApvGainRcd.h"
#include "CondFormats/DataRecord/interface/SiStripCondDataRecords.h"
#include "CondFormats/DataRecord/interface/SiStripNoisesRcd.h"
#include "CondFormats/DataRecord/interface/SiStripPedestalsRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripApvGain.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "CondFormats/SiStripObjects/interface/SiStripPedestals.h"
#include "DataFormats/Common/interface/DetSetNew.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/SiStripCommon/interface/ConstantsForHardwareSystems.h" /* for STRIPS_PER_APV*/
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#define LOGERROR(x) edm::LogError(x)
#define LOGWARNING(x) edm::LogWarning(x)
#define LOGINFO(x) edm::LogInfo(x)
#define LOGVERBATIM(x) edm::LogVerbatim(x)
#define LOGDEBUG(x) LogDebug(x)

// ROOT includes
#include "TNamed.h"
#include "TObjString.h"
#include "TText.h"
#include "TTree.h"

//
// Auxiliary enum to enumerate the conditions types
//
namespace SiStripCondTypes {
  enum condformat { Noise = 0, Pedestals = 1, G1Gain = 2, G2Gain = 3, Quality = 4, EndOfTypes = 99 };
  static const std::array<std::string, 5> titles = {{"noise", "pedestals", "G1 gain", "G2 gain", "Quality"}};
  static const std::array<std::string, 5> units = {{"[ADC counts]", "[ADC counts]", "", "", ""}};

  // some magic from https://stackoverflow.com/questions/47801709/best-way-to-set-a-bitset-with-boolean-values
  template <typename... Args>
  std::bitset<sizeof...(Args)> makeBitSet(Args... as) {
    using unused = bool[];
    std::bitset<sizeof...(Args)> ret;
    std::size_t ui{ret.size()};
    (void)unused{true, (ret.set(--ui, as), true)...};
    return ret;
  }

}  // namespace SiStripCondTypes

//
// class declaration
//

using HistoMap = std::map<uint32_t, TH1F*>;

class SiStripCondVisualizer : public edm::one::EDAnalyzer<edm::one::SharedResources, edm::one::WatchRuns> {
public:
  explicit SiStripCondVisualizer(const edm::ParameterSet&);
  ~SiStripCondVisualizer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginRun(edm::Run const&, edm::EventSetup const&) override;
  void endRun(edm::Run const&, edm::EventSetup const&) override{};
  bool isDetIdSelected(const uint32_t detid);
  HistoMap bookModuleHistograms(const TrackerTopology* tTopo, const SiStripCondTypes::condformat& type);
  std::tuple<std::string, int, int, int> setTopoInfo(uint32_t detId, const TrackerTopology* tTopo);
  std::string module_location_type(const unsigned int& mod);
  void fillTheQualityMap(const SiStripQuality* obj, HistoMap& theMap);
  template <class Payload>
  void fillTheHistoMap(const Payload* obj, HistoMap& theMap);
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  // ----------member data ---------------------------
  edm::ESGetToken<SiStripNoises, SiStripNoisesRcd> noiseToken_;
  edm::ESGetToken<SiStripPedestals, SiStripPedestalsRcd> pedToken_;
  edm::ESGetToken<SiStripApvGain, SiStripApvGainRcd> g1Token_;
  edm::ESGetToken<SiStripApvGain, SiStripApvGain2Rcd> g2Token_;
  edm::ESGetToken<SiStripQuality, SiStripQualityRcd> qualToken_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> topoToken_;

  const bool doNoise_;
  const bool doPeds_;
  const bool doG1_;
  const bool doG2_;
  const bool doBadComps_;

  std::vector<DetIdSelector> detidsels_;
  edm::Service<TFileService> fs_;
  SiStripDetInfo detInfo_;
  std::map<std::string, TFileDirectory> outputFolders_;
  HistoMap NoiseMap_, PedMap_, G1Map_, G2Map_, QualMap_;
  std::bitset<5> plottedConditions_;
};

//
// constructors and destructor
//
SiStripCondVisualizer::SiStripCondVisualizer(const edm::ParameterSet& iConfig)
    : topoToken_(esConsumes<edm::Transition::BeginRun>()),
      doNoise_(iConfig.getParameter<bool>("doNoise")),
      doPeds_(iConfig.getParameter<bool>("doPeds")),
      doG1_(iConfig.getParameter<bool>("doG1")),
      doG2_(iConfig.getParameter<bool>("doG2")),
      doBadComps_(iConfig.getParameter<bool>("doBadComps")),
      detidsels_() {
  usesResource(TFileService::kSharedResource);
  // now do what ever initialization is needed
  detInfo_ = SiStripDetInfoFileReader::read(edm::FileInPath(SiStripDetInfoFileReader::kDefaultFile).fullPath());

  plottedConditions_ = SiStripCondTypes::makeBitSet(doNoise_, doPeds_, doG1_, doG2_, doBadComps_);

  // now do the consumes
  if (doNoise_)
    noiseToken_ = esConsumes();
  if (doPeds_)
    pedToken_ = esConsumes();
  if (doG1_)
    g1Token_ = esConsumes();
  if (doG2_)
    g2Token_ = esConsumes();
  if (doBadComps_)
    qualToken_ = esConsumes(edm::ESInputTag("", iConfig.getParameter<std::string>("StripQualityLabel")));

  // detid selection
  std::vector<edm::ParameterSet> selconfigs = iConfig.getParameter<std::vector<edm::ParameterSet>>("selections");
  for (std::vector<edm::ParameterSet>::const_iterator selconfig = selconfigs.begin(); selconfig != selconfigs.end();
       ++selconfig) {
    DetIdSelector selection(*selconfig);
    detidsels_.push_back(selection);
  }
}

//
// member functions
//

/*
 * special case for the SiStripQuality
 */
void SiStripCondVisualizer::fillTheQualityMap(const SiStripQuality* obj, HistoMap& theMap) {
  const auto& badComponentsList = obj->getBadComponentList();

  for (const auto& bc : badComponentsList) {
    LogDebug("SiStripCondVisualizer") << "Det:" << bc.detid << " location: " << this->module_location_type(bc.detid)
                                      << " bad APVs:" << bc.BadApvs << " bad Fibers:" << bc.BadFibers
                                      << " bad Module:" << bc.BadModule;
  }

  // now get the detids
  std::vector<uint32_t> activeDetIds;
  std::transform(badComponentsList.begin(),
                 badComponentsList.end(),
                 std::back_inserter(activeDetIds),
                 [](const SiStripQuality::BadComponent& bc) { return bc.detid; });

  for (const uint32_t& detid : activeDetIds) {
    if (!this->isDetIdSelected(detid))
      continue;

    unsigned int nStrip = detInfo_.getNumberOfApvsAndStripLength(detid).first * sistrip::STRIPS_PER_APV;

    for (unsigned int istrip_ = 0; istrip_ < nStrip; ++istrip_) {
      bool isStripBad = obj->IsStripBad(detid, istrip_);
      float quant_ = isStripBad ? 1.f : 0.f;
      if (!theMap.count(detid)) {
        LOGWARNING("SiStripCondVisualizer")
            << "@SUB=SiStripCondVisualizer::analyze(): " << detid << " was not found in the quality histogram map!!!";
      } else {
        theMap[detid]->SetBinContent(istrip_, quant_);
      }
    }  // loop on the strips
  }    // loop on the active detids
  return;
}

/*
 * Payload functor based method for all the other cases
 */
template <class Payload>
void SiStripCondVisualizer::fillTheHistoMap(const Payload* obj, HistoMap& theMap) {
  std::function<float(unsigned int, typename Payload::Range)> payloadFunctor = [&obj](unsigned int istrip,
                                                                                      typename Payload::Range range) {
    if constexpr (std::is_same_v<Payload, SiStripNoises>) {
      return obj->getNoise(istrip, range);
    } else if constexpr (std::is_same_v<Payload, SiStripPedestals>) {
      return obj->getPed(istrip, range);
    } else if constexpr (std::is_same_v<Payload, SiStripApvGain>) {
      return obj->getStripGain(istrip, range);
    }
  };

  std::vector<uint32_t> activeDetIds;
  obj->getDetIds(activeDetIds);

  for (const uint32_t& detid : activeDetIds) {
    if (!this->isDetIdSelected(detid))
      continue;

    typename Payload::Range condRange = obj->getRange(detid);
    unsigned int nStrip = detInfo_.getNumberOfApvsAndStripLength(detid).first * sistrip::STRIPS_PER_APV;

    for (unsigned int istrip_ = 0; istrip_ < nStrip; ++istrip_) {
      float quant_ = payloadFunctor(istrip_, condRange);

      if (!theMap.count(detid)) {
        LOGWARNING("SiStripCondVisualizer")
            << "@SUB=SiStripCondVisualizer::analyze(): " << detid << " was not found in the histogram map!!!";
      } else {
        theMap[detid]->SetBinContent(istrip_, quant_);
      }
    }  // loop on the strips
  }    // loop on the active detids
}

// ------------ method called for each event  ------------
void SiStripCondVisualizer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  if (doNoise_) {
    const SiStripNoises* noiseObj = &iSetup.getData(noiseToken_);
    this->fillTheHistoMap<SiStripNoises>(noiseObj, NoiseMap_);
  }
  if (doPeds_) {
    const SiStripPedestals* pedestalObj = &iSetup.getData(pedToken_);
    this->fillTheHistoMap<SiStripPedestals>(pedestalObj, PedMap_);
  }
  if (doG1_) {
    const SiStripApvGain* g1Obj = &iSetup.getData(g1Token_);
    this->fillTheHistoMap<SiStripApvGain>(g1Obj, G1Map_);
  }
  if (doG2_) {
    const SiStripApvGain* g2Obj = &iSetup.getData(g2Token_);
    this->fillTheHistoMap<SiStripApvGain>(g2Obj, G2Map_);
  }
  if (doBadComps_) { /* special case for the SiStripQuality */
    const SiStripQuality* siStripQualityObj = &iSetup.getData(qualToken_);
    this->fillTheQualityMap(siStripQualityObj, QualMap_);
  }
}

// ------------ method called for each run  ------------
void SiStripCondVisualizer::beginRun(const edm::Run& iRun, edm::EventSetup const& iSetup) {
  const TrackerTopology* tTopo_ = &iSetup.getData(topoToken_);
  const std::map<uint32_t, SiStripDetInfo::DetInfo> DetInfos = detInfo_.getAllData();

  for (const auto& it : DetInfos) {
    LogDebug("SiStripCondVisualizer") << "detid " << it.first << "isSelected  " << this->isDetIdSelected(it.first);

    if (!this->isDetIdSelected(it.first))
      continue;

    auto topolInfo = setTopoInfo(it.first, tTopo_);
    std::string thePart = std::get<0>(topolInfo);

    for (std::size_t i = 0; i < plottedConditions_.size(); ++i) {
      if (plottedConditions_.test(i)) {
        const std::string fname = Form("%s_%s", SiStripCondTypes::titles[i].c_str(), thePart.c_str());

        // book the TFileDirectory if it's not already done
        if (!outputFolders_.count(fname)) {
          outputFolders_[fname] = fs_->mkdir(fname);
        }
      }
    }  // loop on the bitset of plotted conditions
  }    // loop on modules

  if (doNoise_) {
    LOGINFO("SiStripCondVisualizer") << "@SUB=SiStripCondVisualizer::beginRun() \n Before booking NoisMap_.size(): "
                                     << NoiseMap_.size();
    NoiseMap_ = bookModuleHistograms(tTopo_, SiStripCondTypes::Noise);
    LOGINFO("SiStripCondVisualizer") << "@SUB=SiStripCondVisualizer::beginRun() \n After booking NoisMap_.size(): "
                                     << NoiseMap_.size();
  }
  if (doPeds_) {
    LOGINFO("SiStripCondVisualizer") << "@SUB=SiStripCondVisualizer::beginRun() \n Before booking PedMap_.size(): "
                                     << PedMap_.size();
    PedMap_ = bookModuleHistograms(tTopo_, SiStripCondTypes::Pedestals);
    LOGINFO("SiStripCondVisualizer") << "@SUB=SiStripCondVisualizer::beginRun() \n After booking PedMap_.size(): "
                                     << PedMap_.size();
  }
  if (doG1_) {
    LOGINFO("SiStripCondVisualizer") << "@SUB=SiStripCondVisualizer::beginRun() \n Before booking G1Map_.size(): "
                                     << G1Map_.size();
    G1Map_ = bookModuleHistograms(tTopo_, SiStripCondTypes::G1Gain);
    LOGINFO("SiStripCondVisualizer") << "@SUB=SiStripCondVisualizer::beginRun() \n After booking G1Map_.size(): "
                                     << G1Map_.size();
  }
  if (doG2_) {
    LOGINFO("SiStripCondVisualizer") << "@SUB=SiStripCondVisualizer::beginRun() \n Before booking G2Map_.size(): "
                                     << G2Map_.size();
    G2Map_ = bookModuleHistograms(tTopo_, SiStripCondTypes::G2Gain);
    LOGINFO("SiStripCondVisualizer") << "@SUB=SiStripCondVisualizer::beginRun() \n After booking G2Map_.size(): "
                                     << G2Map_.size();
  }
  if (doBadComps_) {
    LOGINFO("SiStripCondVisualizer") << "@SUB=SiStripCondVisualizer::beginRun() \n Before booking QualMap_.size(): "
                                     << QualMap_.size();
    QualMap_ = bookModuleHistograms(tTopo_, SiStripCondTypes::Quality);
    LOGINFO("SiStripCondVisualizer") << "@SUB=SiStripCondVisualizer::beginRun() \n After booking QualMap_.size(): "
                                     << QualMap_.size();
  }
}

// ------------ method called to determine the topology  ------------
std::tuple<std::string, int, int, int> SiStripCondVisualizer::setTopoInfo(uint32_t detId,
                                                                          const TrackerTopology* tTopo) {
  int subdetId_(-999), layer_(-999), side_(-999);
  std::string ret = "";
  std::tuple<std::string, int, int, int> def_tuple{ret, subdetId_, layer_, side_};

  subdetId_ = DetId(detId).subdetId();
  switch (subdetId_) {
    case SiStripSubdetector::TIB:  // TIB
      layer_ = tTopo->tibLayer(detId);
      side_ = 0;
      ret += Form("TIB_Layer%i", layer_);
      break;
    case SiStripSubdetector::TID:  // TID
      side_ = tTopo->tidSide(detId);
      layer_ = tTopo->tidWheel(detId);
      ret += ("TID_");
      ret += (side_ == 1) ? Form("P_disk%i", layer_) : Form("M_disk%i", layer_);
      break;
    case SiStripSubdetector::TOB:  // TOB
      layer_ = tTopo->tobLayer(detId);
      side_ = 0;
      ret += Form("TOB_Layer%i", layer_);
      break;
    case SiStripSubdetector::TEC:  // TEC
      side_ = tTopo->tecSide(detId);
      layer_ = tTopo->tecWheel(detId);
      ret += ("TEC_");
      ret += (side_ == 1) ? Form("P_disk%i", layer_) : Form("M_disk%i", layer_);
      break;
    default:
      edm::LogError("SiStripCondVisualizer") << "SUB=SiStripCondVisualizer::setTopoInfo() \n unrecognizer partition.";
      return def_tuple;
  }

  return std::make_tuple(ret, subdetId_, layer_, side_);
}

// name of the location of a given module and its type,
// e.g. TIB_L1s: stereo module at TIB layer 1
std::string SiStripCondVisualizer::module_location_type(const unsigned int& mod) {
  const SiStripDetId detid(mod);
  std::string subdet = "";
  if (detid.subDetector() == SiStripDetId::TIB)
    subdet = "TIB";
  if (detid.subDetector() == SiStripDetId::TOB)
    subdet = "TOB";
  if (detid.subDetector() == SiStripDetId::TID)
    subdet = "TID";
  if (detid.subDetector() == SiStripDetId::TEC)
    subdet = "TEC";

  // Barrel
  int layer = int((mod >> 14) & 0x7);
  std::string type = (detid.stereo() ? "s" : "a");
  std::string d_l_t = Form("%s_L%d%s", subdet.c_str(), layer, type.c_str());

  // Endcaps
  if (subdet == "TID" || subdet == "TEC") {
    unsigned int sideStartBit_{0};
    unsigned int wheelStartBit_{0};
    unsigned int ringStartBit_{0};
    unsigned int sideMask_{0};
    unsigned int wheelMask_{0};
    unsigned int ringMask_{0};

    // TEC
    if (subdet == "TEC") {
      sideStartBit_ = 18;
      wheelStartBit_ = 14;
      ringStartBit_ = 5;
      sideMask_ = 0x3;
      wheelMask_ = 0xF;
      ringMask_ = 0x7;
    }

    // TID
    if (subdet == "TID") {
      sideStartBit_ = 13;
      wheelStartBit_ = 11;
      ringStartBit_ = 9;
      sideMask_ = 0x3;
      wheelMask_ = 0x3;
      ringMask_ = 0x3;
    }

    // TEC+-, TID+- (see also at the bottom of this file
    int side = int((mod >> sideStartBit_) & sideMask_);
    int wheel = int((mod >> wheelStartBit_) & wheelMask_);
    int ring = int((mod >> ringStartBit_) & ringMask_);

    std::string s_side = (side == 1 ? "Plus" : "Minus");

    d_l_t = Form("%s%s_W%dR%d", subdet.c_str(), s_side.c_str(), wheel, ring);
  }
  return d_l_t;
}

// ------------ method called to determine if the detid is selected
bool SiStripCondVisualizer::isDetIdSelected(const uint32_t detid) {
  bool isSelected{false};
  for (std::vector<DetIdSelector>::const_iterator detidsel = detidsels_.begin(); detidsel != detidsels_.end();
       ++detidsel) {
    if (detidsel->isSelected(detid)) {
      isSelected = true;
      break;
    }
  }
  return isSelected;
}

// ------------ method called once to book all the module level histograms
HistoMap SiStripCondVisualizer::bookModuleHistograms(const TrackerTopology* tTopo_,
                                                     const SiStripCondTypes::condformat& type) {
  TH1F::SetDefaultSumw2(kTRUE);
  HistoMap h;

  const std::map<uint32_t, SiStripDetInfo::DetInfo> DetInfos = detInfo_.getAllData();

  for (const auto& it : DetInfos) {
    // check if det id is correct and if it is actually cabled in the detector
    if (it.first == 0 || it.first == 0xFFFFFFFF) {
      edm::LogError("DetIdNotGood") << "@SUB=analyze"
                                    << "Wrong det id: " << it.first << "  ... neglecting!";
      continue;
    }

    if (!this->isDetIdSelected(it.first))
      continue;

    auto topolInfo = setTopoInfo(it.first, tTopo_);
    const std::string thePart = std::get<0>(topolInfo);
    const std::string fname = Form("%s_%s", SiStripCondTypes::titles[type].c_str(), thePart.c_str());

    unsigned int nStrip = detInfo_.getNumberOfApvsAndStripLength(it.first).first * sistrip::STRIPS_PER_APV;

    h[it.first] =
        outputFolders_[fname].make<TH1F>(Form("%sProfile_%i", SiStripCondTypes::titles[type].c_str(), it.first),
                                         Form("%s for module %i (%s);n. strip; %s %s",
                                              SiStripCondTypes::titles[type].c_str(),
                                              it.first,
                                              thePart.c_str(),
                                              SiStripCondTypes::titles[type].c_str(),
                                              SiStripCondTypes::units[type].c_str()),
                                         nStrip,
                                         -0.5,
                                         nStrip + 0.5);
  }
  return h;
}

// ------------ method fills 'descriptions' with the allowed parameters for the module
void SiStripCondVisualizer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setComment("Creates a ROOT file with the per-moudle profiles of different SiStrip Database tag contents.");
  desc.add<bool>("doNoise", false);
  desc.add<bool>("doPeds", false);
  desc.add<bool>("doG1", false);
  desc.add<bool>("doG2", false);
  desc.add<bool>("doBadComps", false);
  desc.add<std::string>("StripQualityLabel", "MergedBadComponent");

  // for the DetId selection
  edm::ParameterSetDescription desc_detIdSelection;
  desc_detIdSelection.add<unsigned int>("detSelection");
  desc_detIdSelection.add<std::string>("detLabel");
  desc_detIdSelection.addUntracked<std::vector<std::string>>("selection");
  std::vector<edm::ParameterSet> default_detIdSelectionVector;
  edm::ParameterSet default_detIdSelector;
  default_detIdSelector.addParameter<unsigned int>("detSelection", 1);
  default_detIdSelector.addParameter<std::string>("detLabel", "Tracker");
  default_detIdSelector.addUntrackedParameter<std::vector<std::string>>("selection",
                                                                        {"0x1e000000-0x16000000",
                                                                         "0x1e006000-0x18002000",
                                                                         "0x1e006000-0x18004000",
                                                                         "0x1e000000-0x1a000000",
                                                                         "0x1e0c0000-0x1c040000",
                                                                         "0x1e0c0000-0x1c080000"});
  default_detIdSelectionVector.push_back(default_detIdSelector);
  desc.addVPSet("selections", desc_detIdSelection, default_detIdSelectionVector);

  descriptions.addWithDefaultLabel(desc);
}

// define this as a plug-in
DEFINE_FWK_MODULE(SiStripCondVisualizer);
