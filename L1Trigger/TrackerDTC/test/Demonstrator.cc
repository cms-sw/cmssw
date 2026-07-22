#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/L1TrackTrigger/interface/TTDTC.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "L1Trigger/TrackerDTC/interface/Setup.h"
#include "L1Trigger/TrackerDTC/interface/SensorModule.h"
#include "L1Trigger/TrackerDTC/interface/DTC.h"
#include "L1Trigger/TrackerDTC/interface/StubFE.h"
#include "L1Trigger/TrackerDTC/interface/StubGL.h"
#include "L1Trigger/TrackerDTC/interface/StubDTC.h"

#include <TProfile.h>
#include <TH1F.h>
#include <TH2F.h>

#include <vector>
#include <deque>
#include <cmath>
#include <fstream>
#include <string>
#include <numeric>

namespace trackerDTC {

  /*! \class  trackerDTC::Demonstrator
   *  \brief  Class to fully emulate and test TrackerDTC f/w
   *  \author Thomas Schuh
   *  \date   2025, Aug
   */
  class Demonstrator : public edm::one::EDAnalyzer<edm::one::WatchRuns, edm::one::SharedResources> {
  public:
    Demonstrator(const edm::ParameterSet&);
    void beginJob() override {}
    void beginRun(const edm::Run&, const edm::EventSetup&) override;
    void analyze(const edm::Event&, const edm::EventSetup&) override;
    void endRun(const edm::Run&, const edm::EventSetup&) override;
    void endJob() override {}

  private:
    // ED input token of stubs
    edm::EDGetTokenT<TTStubDetSetVec> edGetToken_;
    // Setup token
    edm::ESGetToken<Setup, SetupRcd> esGetTokenSetup_;
    // stores, calculates and provides run-time constants
    const Setup* setup_;
    // configuration
    DTC::Config config_;
    // dtc ids to be tested
    std::vector<int> dtcIds_;
    // bx id
    int bx_ = 0;
    // single DTC emulators
    std::vector<DTC> dtcs_;
    std::vector<TH1F*> hisComb_;
    std::vector<TProfile*> profComb_;
    std::vector<std::vector<TH1F*>> his_;
    std::vector<std::vector<TProfile*>> prof_;
    std::vector<TH2F*> hisRZStubs_;
    std::vector<TH2F*> hisRZLost_;
    TH2F* hisRZStubsComb_;
    TH2F* hisRZLostComb_;
    TH2F* hisRZFrac_;
  };

  Demonstrator::Demonstrator(const edm::ParameterSet& iConfig) {
    usesResource("TFileService");
    const edm::InputTag& inputTag = iConfig.getParameter<edm::InputTag>("InputTag");
    edGetToken_ = consumes(inputTag);
    esGetTokenSetup_ = esConsumes<edm::Transition::BeginRun>();
    config_.enable = iConfig.getParameter<bool>("Enable");
    config_.runTime = iConfig.getParameter<double>("RunTime");
    config_.num8BX = iConfig.getParameter<int>("Num8BX");
    config_.pathIPBB = iConfig.getParameter<std::string>("PathIPBB");
    dtcIds_ = iConfig.getParameter<std::vector<int>>("IDs");
  }

  void Demonstrator::beginRun(const edm::Run& iEvent, const edm::EventSetup& iSetup) {
    // stores, calculates and provides run-time constants
    setup_ = &iSetup.getData(esGetTokenSetup_);
    if (dtcIds_.empty()) {
      dtcIds_ = std::vector<int>(setup_->sysNumRegion() * setup_->regNumDTC());
      std::iota(dtcIds_.begin(), dtcIds_.end(), 0);
    }
    config_.num18BX = config_.num8BX * setup_->cicNumBX() / setup_->regNumTFP();
    // Histograms
    const std::vector<std::string> names = {"In", "TMP8", "TMP12", "TMP18", "Out"};
    const std::vector<int> maxOcc = {73, 73, 109, 163, 157};
    const std::vector<int> numChannel = {72, 8, 4, 2, 2};
    hisComb_ = std::vector<TH1F*>(names.size());
    profComb_ = std::vector<TProfile*>(names.size());
    edm::Service<TFileService> fs;
    TFileDirectory dir = fs->mkdir("DTC");
    for (int i = 0; i < static_cast<int>(names.size()); i++) {
      hisComb_[i] = dir.make<TH1F>(("His Channel Occupancy" + names[i]).c_str(), ";", maxOcc[i], -.5, maxOcc[i] - .5);
      profComb_[i] = dir.make<TProfile>(
          ("Prof Channel Occupancy" + names[i]).c_str(), ";", numChannel[i], -.5, numChannel[i] - .5);
    }
    his_ = std::vector<std::vector<TH1F*>>(dtcIds_.size(), std::vector<TH1F*>(names.size()));
    prof_ = std::vector<std::vector<TProfile*>>(dtcIds_.size(), std::vector<TProfile*>(names.size()));
    // lost stub fraction in r-z
    constexpr int bins = 400;
    constexpr double maxZ = 300.;
    constexpr double maxR = 120.;
    hisRZStubsComb_ = dir.make<TH2F>("RZ Stubs", ";;", bins, -maxZ, maxZ, bins, 0., maxR);
    hisRZLostComb_ = dir.make<TH2F>("RZ Lost Stubs", ";;", bins, -maxZ, maxZ, bins, 0., maxR);
    hisRZFrac_ = dir.make<TH2F>("RZ Lost Stub Frac", ";;", bins, -maxZ, maxZ, bins, 0., maxR);
    hisRZStubs_.reserve(dtcIds_.size());
    hisRZLost_.reserve(dtcIds_.size());
    // prep all DTC boards
    dtcs_.reserve(dtcIds_.size());
    int id(0);
    for (int dtcId : dtcIds_) {
      TFileDirectory dir = fs->mkdir("DTC_" + std::to_string(dtcId));
      std::vector<TH1F*>& his = his_[id];
      std::vector<TProfile*>& prof = prof_[id];
      for (int i = 0; i < static_cast<int>(names.size()); i++) {
        his[i] = dir.make<TH1F>(("His Channel Occupancy" + names[i]).c_str(), ";", maxOcc[i], -.5, maxOcc[i] - .5);
        prof[i] = dir.make<TProfile>(
            ("Prof Channel Occupancy" + names[i]).c_str(), ";", numChannel[i], -.5, numChannel[i] - .5);
      }
      hisRZStubs_.emplace_back(dir.make<TH2F>("RZ Stubs", ";;", bins, -maxZ, maxZ, bins, 0., maxR));
      hisRZLost_.emplace_back(dir.make<TH2F>("RZ Lost Stubs", ";;", bins, -maxZ, maxZ, bins, 0., maxR));
      dtcs_.emplace_back(setup_, config_, dtcId, his, prof, hisRZStubs_.back(), hisRZLost_.back());
      id++;
    }
  }

  void Demonstrator::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
    // get stubs
    edm::Handle<TTStubDetSetVec> handle;
    iEvent.getByToken(edGetToken_, handle);
    // process single bx
    for (DTC& dtc : dtcs_)
      dtc.consume(handle, bx_ % setup_->cicNumBX());
    // process 8 bx boxcars
    if (++bx_ % setup_->cicNumBX() == 0)
      for (DTC& dtc : dtcs_)
        dtc.produce(bx_);
    // compare s/w with f/w
    if (bx_ == config_.num8BX * setup_->cicNumBX()) {
      if (config_.enable)
        std::cout << "Comparing DTCs in turn: " << std::flush;
      for (DTC& dtc : dtcs_)
        dtc.analyze();
      if (config_.enable)
        std::cout << " pass" << std::endl << std::flush;
      bx_ = 0;
    }
  }

  void Demonstrator::endRun(const edm::Run& iEvent, const edm::EventSetup& iSetup) {
    // combine Histograms
    for (int i = 0; i < static_cast<int>(hisComb_.size()); i++) {
      for (const std::vector<TH1F*>& his : his_)
        hisComb_[i]->Add(his[i]);
      for (const std::vector<TProfile*>& prof : prof_)
        profComb_[i]->Add(prof[i]);
    }
    for (TH2F* stubs : hisRZStubs_)
      hisRZStubsComb_->Add(stubs);
    for (TH2F* lost : hisRZLost_)
      hisRZLostComb_->Add(lost);
    hisRZFrac_->Add(hisRZLostComb_);
    hisRZFrac_->Divide(hisRZStubsComb_);
  }

}  // namespace trackerDTC

DEFINE_FWK_MODULE(trackerDTC::Demonstrator);
