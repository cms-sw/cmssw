// -*- C++ -*-
//
// Package:    CondTools/SiStrip
// Class:      SiStripDB2Tree
//
/**\class SiStripDB2Tree SiStripDB2Tree.cc CondTools/SiStrip/plugins/SiStripDB2Tree.cc

 Description:
     Converts several SiStrip Payloads (Noise, Peds, APVGain, etc.) into a TTree
*/
//
// Original Author:  Mauro Verzetti
// Modified by:      Marco Musich
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
#include "CondFormats/DataRecord/interface/SiStripApvGainRcd.h"
#include "CondFormats/DataRecord/interface/SiStripCondDataRecords.h"
#include "CondFormats/DataRecord/interface/SiStripNoisesRcd.h"
#include "CondFormats/DataRecord/interface/SiStripPedestalsRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripApvGain.h"
#include "CondFormats/SiStripObjects/interface/SiStripLatency.h"
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
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#define LOGERROR(x) edm::LogError(x)
#define LOGINFO(x) edm::LogInfo(x)
#define LOGDEBUG(x) LogDebug(x)

// ROOT includes
#include "TNamed.h"
#include "TObjString.h"
#include "TText.h"
#include "TTree.h"

//**********************************************//
// Auxilliary class
//**********************************************//
class RecordInfo : public TNamed {
public:
  RecordInfo(const char* record, const char* tag) : TNamed(record, tag) {}

  void printInfo() const { LOGINFO("RecordInfo") << GetName() << " " << GetTitle(); }

  const char* getRecord() { return this->GetName(); }
  const char* getIOVSince() { return this->GetTitle(); }
};

class SiStripDB2Tree : public edm::one::EDAnalyzer<edm::one::SharedResources, edm::one::WatchRuns> {
public:
  explicit SiStripDB2Tree(const edm::ParameterSet&);
  ~SiStripDB2Tree() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginRun(edm::Run const&, edm::EventSetup const&) override;
  void endRun(edm::Run const&, edm::EventSetup const&) override{};
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void setTopoInfo(uint32_t detId, const TrackerTopology* tTopo);
  template <class Rcd>
  std::pair<const char*, std::string> getRecordInfo(const edm::EventSetup& iSetup) const;

  // ----------member data ---------------------------

  // ES tokens
  const edm::ESGetToken<SiStripPedestals, SiStripPedestalsRcd> pedToken_;
  const edm::ESGetToken<SiStripNoises, SiStripNoisesRcd> noiseToken_;
  const edm::ESGetToken<SiStripApvGain, SiStripApvGainRcd> g1Token_;
  const edm::ESGetToken<SiStripApvGain, SiStripApvGain2Rcd> g2Token_;
  const edm::ESGetToken<SiStripQuality, SiStripQualityRcd> qualToken_;
  const edm::ESGetToken<SiStripApvGain, SiStripApvGainSimRcd> gsimToken_;
  const edm::ESGetToken<SiStripLatency, SiStripLatencyRcd> latToken_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> topoToken_;

  const bool isMC_;

  TTree* tree_;
  std::string processGT_;

  //branches
  uint32_t detId_, ring_, istrip_, det_type_;
  Int_t layer_, side_, subdetId_;
  float noise_, gsim_, g1_, g2_, lenght_, pedestal_;
  bool isTIB_, isTOB_, isTEC_, isTID_, isBad_;
  TText* text_;

  // detInfo
  SiStripDetInfo detInfo_;
};

//
// constructors and destructor
//
SiStripDB2Tree::SiStripDB2Tree(const edm::ParameterSet& iConfig)
    : pedToken_(esConsumes()),
      noiseToken_(esConsumes()),
      g1Token_(esConsumes()),
      g2Token_(esConsumes()),
      qualToken_(esConsumes(edm::ESInputTag("", iConfig.getParameter<std::string>("StripQualityLabel")))),
      gsimToken_(esConsumes()),
      latToken_(esConsumes()),
      topoToken_(esConsumes()),
      isMC_(iConfig.getUntrackedParameter<bool>("isMC", false)),
      detId_(0),
      ring_(0),
      istrip_(0),
      det_type_(0),
      layer_(0),
      side_(0),
      subdetId_(0),
      noise_(0),
      gsim_(0),
      g1_(0),
      g2_(0),
      lenght_(0),
      isTIB_(false),
      isTOB_(false),
      isTEC_(false),
      isTID_(false) {
  usesResource(TFileService::kSharedResource);

  edm::Service<TFileService> fs;
  //now do what ever initialization is needed
  text_ = fs->make<TText>(0., 0., "");
  text_->SetName("RunMode");
  tree_ = fs->make<TTree>("StripDBTree", "Tree with DB SiStrip info");

  tree_->Branch("detId", &detId_, "detId/i");
  tree_->Branch("detType", &det_type_, "detType/i");
  tree_->Branch("noise", &noise_, "noise/F");
  tree_->Branch("pedestal", &pedestal_, "pedestal/F");
  tree_->Branch("istrip", &istrip_, "istrip/i");
  tree_->Branch("gsim", &gsim_, "gsim/F");
  tree_->Branch("g1", &g1_, "g1/F");
  tree_->Branch("g2", &g2_, "g2/F");
  tree_->Branch("layer", &layer_, "layer/I");
  tree_->Branch("side", &side_, "side/I");
  tree_->Branch("subdetId", &subdetId_, "subdetId/I");
  tree_->Branch("ring", &ring_, "ring/i");
  tree_->Branch("length", &lenght_, "length/F");
  tree_->Branch("isBad", &isBad_, "isBad/O");
  tree_->Branch("isTIB", &isTIB_, "isTIB/O");
  tree_->Branch("isTOB", &isTOB_, "isTOB/O");
  tree_->Branch("isTEC", &isTEC_, "isTEC/O");
  tree_->Branch("isTID", &isTID_, "isTID/O");

  detInfo_ = SiStripDetInfoFileReader::read(edm::FileInPath(SiStripDetInfoFileReader::kDefaultFile).fullPath());
}

//
// member functions
//

void SiStripDB2Tree::setTopoInfo(uint32_t detId, const TrackerTopology* tTopo) {
  subdetId_ = DetId(detId).subdetId();
  switch (subdetId_) {
    case SiStripSubdetector::TIB:  //TIB
      isTIB_ = true;
      isTOB_ = false;
      isTEC_ = false;
      isTID_ = false;
      layer_ = tTopo->tibLayer(detId);
      side_ = 0;
      ring_ = 0;
      break;
    case SiStripSubdetector::TID:  //TID
      isTIB_ = false;
      isTOB_ = false;
      isTEC_ = false;
      isTID_ = true;
      side_ = tTopo->tidSide(detId);
      layer_ = tTopo->tidWheel(detId);
      ring_ = 0;
      break;
    case SiStripSubdetector::TOB:  //TOB
      isTIB_ = false;
      isTOB_ = true;
      isTEC_ = false;
      isTID_ = false;
      layer_ = tTopo->tobLayer(detId);
      side_ = 0;
      ring_ = 0;
      break;
    case SiStripSubdetector::TEC:  //TEC
      isTIB_ = false;
      isTOB_ = false;
      isTEC_ = true;
      isTID_ = false;
      side_ = tTopo->tecSide(detId);
      layer_ = tTopo->tecWheel(detId);
      ring_ = 0;
      break;
  }
  return;
}

// ------------ auxilliary method for record info  ------------
template <class Rcd>
std::pair<const char*, std::string> SiStripDB2Tree::getRecordInfo(const edm::EventSetup& iSetup) const {
  const Rcd& record = iSetup.get<Rcd>();
  const edm::ValidityInterval& validity = record.validityInterval();
  const edm::IOVSyncValue first = validity.first();
  const edm::IOVSyncValue last = validity.last();
  if (first != edm::IOVSyncValue::beginOfTime() || last != edm::IOVSyncValue::endOfTime()) {
    LOGINFO("SiStripDB2Tree") << "@SUB=SiStripDB2Tree::getRecordInfo"
                              << "\nTrying to apply " << record.key().name() << " with multiple IOVs in tag.\n"
                              << "Validity range is " << first.eventID().run() << " - " << last.eventID().run() << "\n";
  } else {
    LOGINFO("SiStripDB2Tree") << "@SUB=SiStripDB2Tree::getRecordInfo"
                              << "\nTrying to apply " << record.key().name() << "Validity range is "
                              << first.eventID().run() << " - " << last.eventID().run() << "\n";
  }

  tree_->GetUserInfo()->Add(new RecordInfo(record.key().name(), std::to_string(first.eventID().run()).c_str()));

  return std::make_pair(record.key().name(), std::to_string(first.eventID().run()));
}

// ------------ method called for each run  ------------
void SiStripDB2Tree::beginRun(const edm::Run& iRun, edm::EventSetup const& iSetup) {
  char c_run[30];
  sprintf(c_run, "%i", iRun.run());
  tree_->GetUserInfo()->Add(new TObjString(c_run));

  auto pedHook = this->getRecordInfo<SiStripPedestalsRcd>(iSetup);
  auto noiseHook = this->getRecordInfo<SiStripNoisesRcd>(iSetup);
  auto g1Hook = this->getRecordInfo<SiStripApvGainRcd>(iSetup);
  auto g2Hook = this->getRecordInfo<SiStripApvGain2Rcd>(iSetup);
  auto qualityHook = this->getRecordInfo<SiStripQualityRcd>(iSetup);
  if (isMC_) {
    auto g1SimHook = this->getRecordInfo<SiStripApvGainSimRcd>(iSetup);
  }
}

// ------------ method called for each event  ------------
void SiStripDB2Tree::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // fill header information
  LogDebug("SiStrip") << edm::getProcessParameterSetContainingModule(moduleDescription()).dump();

  const edm::ParameterSet& globalTagPSet =
      edm::getProcessParameterSetContainingModule(moduleDescription()).getParameterSet("PoolDBESSource@GlobalTag");

  processGT_ = globalTagPSet.getParameter<std::string>("globaltag");

  RecordInfo* GTheader = new RecordInfo("GlobalTag", processGT_.c_str());
  tree_->GetUserInfo()->Add(GTheader);
  GTheader->printInfo();

  // handles
  const SiStripPedestals* pedestalObj = &iSetup.getData(pedToken_);
  const SiStripNoises* noiseObj = &iSetup.getData(noiseToken_);
  const SiStripApvGain* g1Obj = &iSetup.getData(g1Token_);
  const SiStripApvGain* g2Obj = &iSetup.getData(g2Token_);
  const SiStripQuality* siStripQualityObj = &iSetup.getData(qualToken_);
  const SiStripApvGain* gsimObj = nullptr;
  if (isMC_) {
    gsimObj = &iSetup.getData(gsimToken_);
  } else {
    LOGINFO("SiStripDB2Tree") << "We have determined this Data";
  }

  bool first = true;
  const SiStripLatency* latencyObj = &iSetup.getData(latToken_);

  std::vector<uint32_t> activeDetIds;
  noiseObj->getDetIds(activeDetIds);

  const TrackerTopology* tTopo_ = &iSetup.getData(topoToken_);

  for (uint32_t detid : activeDetIds) {
    setTopoInfo(detid, tTopo_);

    SiStripNoises::Range noiseRange = noiseObj->getRange(detid);
    SiStripApvGain::Range gsimRange;
    if (isMC_ && gsimObj != nullptr) {
      gsimObj->getRange(detid);
    }
    SiStripApvGain::Range g1Range = g1Obj->getRange(detid);
    SiStripApvGain::Range g2Range = g2Obj->getRange(detid);
    SiStripPedestals::Range pedestalsRange = pedestalObj->getRange(detid);

    unsigned int nStrip = detInfo_.getNumberOfApvsAndStripLength(detid).first * sistrip::STRIPS_PER_APV;
    lenght_ = detInfo_.getNumberOfApvsAndStripLength(detid).second;
    detId_ = detid;
    det_type_ = static_cast<unsigned int>(SiStripDetId(detid).moduleGeometry());
    for (istrip_ = 0; istrip_ < nStrip; ++istrip_) {
      if (first) {
        first = false;
        std::string run_op = ((latencyObj->latency(detid, 1) & READMODEMASK) == READMODEMASK) ? "PEAK" : "DECO";
        text_->SetText(0., 0., run_op.c_str());
        LOGINFO("SiStripDB2Tree") << "SiStripOperationModeRcd "
                                  << ". . . " << run_op;
      }
      gsim_ = isMC_ ? gsimObj->getStripGain(istrip_, gsimRange) : 1.;
      g1_ = g1Obj->getStripGain(istrip_, g1Range) ? g1Obj->getStripGain(istrip_, g1Range) : 1.;
      g2_ = g2Obj->getStripGain(istrip_, g2Range) ? g2Obj->getStripGain(istrip_, g2Range) : 1.;
      noise_ = noiseObj->getNoise(istrip_, noiseRange);
      pedestal_ = pedestalObj->getPed(istrip_, pedestalsRange);
      isBad_ = siStripQualityObj->IsStripBad(siStripQualityObj->getRange(detid), istrip_);
      tree_->Fill();
    }
  }
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void SiStripDB2Tree::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.setComment("Creates TTree with SiStrip Database tag content.");
  desc.add<std::string>("StripQualityLabel", "MergedBadComponent");
  desc.addUntracked<bool>("isMC", false);

  descriptions.add("SiStripDB2Tree", desc);
}

//define this as a plug-in
#include "FWCore/PluginManager/interface/ModuleDef.h"
DEFINE_FWK_MODULE(SiStripDB2Tree);
