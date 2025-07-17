#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/Common/interface/Handle.h"

#include "L1Trigger/TrackTrigger/interface/Setup.h"

#include <TProfile.h>
#include <TH1F.h>

#include <vector>
#include <deque>

namespace trackerDTC {

  /*! \class  trackerDTC::AnalyzerDAQ
   *  \brief  Class to analyze TTCluster Occupancies on DTCs, plots cluster occupancy
   *  \author Thomas Schuh
   *  \date   2020, Oct
   */
  class AnalyzerDAQ : public edm::one::EDAnalyzer<edm::one::WatchRuns, edm::one::SharedResources> {
  public:
    AnalyzerDAQ(const edm::ParameterSet& iConfig);
    void beginRun(const edm::Run& iEvent, const edm::EventSetup& iSetup) override;
    void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) override;
    void endRun(const edm::Run& iEvent, const edm::EventSetup& iSetup) override {}
    void endJob() override {}

  private:
    // ED input token of accepted TTClusters
    edm::EDGetTokenT<TTClusterDetSetVec> edGetToken_;
    // Setup token
    edm::ESGetToken<tt::Setup, tt::SetupRcd> esGetTokenSetup_;

    // Histograms

    TProfile* profModules_;
    TH1F* hisModules_;
    TProfile* profDTCs_;
    TH1F* hisDTCs_;
    TH1F* hisTracker_;
  };

  AnalyzerDAQ::AnalyzerDAQ(const edm::ParameterSet& iConfig) {
    usesResource("TFileService");
    // book input ED products
    const auto& inputTag = iConfig.getParameter<edm::InputTag>("InputTagTTClusterDetSetVec");
    edGetToken_ = consumes<TTClusterDetSetVec>(inputTag);
    // book ES products
    esGetTokenSetup_ = esConsumes<edm::Transition::BeginRun>();
  }

  void AnalyzerDAQ::beginRun(const edm::Run& iEvent, const edm::EventSetup& iSetup) {
    // helper class to store configurations
    const tt::Setup* setup = &iSetup.getData(esGetTokenSetup_);
    // book histograms
    edm::Service<TFileService> fs;
    TFileDirectory dir;
    dir = fs->mkdir("Modules");
    int maxOcc = 150;
    int numChannels = setup->numDTCs() * setup->numModulesPerDTC();
    hisModules_ = dir.make<TH1F>("His Module Occupancy", ";", maxOcc, -.5, maxOcc - .5);
    profModules_ = dir.make<TProfile>("Prof Module Occupancy", ";", numChannels, -.5, numChannels - .5);
    dir = fs->mkdir("DTCs");
    maxOcc = 3456;
    numChannels = setup->numDTCs();
    hisDTCs_ = dir.make<TH1F>("His DTC Occupancy", ";", maxOcc / 16, -.5, maxOcc - .5);
    profDTCs_ = dir.make<TProfile>("Prof DTC Occupancy", ";", numChannels, -.5, numChannels - .5);
    dir = fs->mkdir("Tracker");
    maxOcc = pow(2, 29);
    hisTracker_ = dir.make<TH1F>("His Tracker Occupancy", ";", maxOcc * pow(2., -12), -.5, maxOcc - .5);
  }

  void AnalyzerDAQ::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
    const tt::Setup* setup = &iSetup.getData(esGetTokenSetup_);
    // read in original TTCluster collection
    edm::Handle<TTClusterDetSetVec> handle;
    iEvent.getByToken<TTClusterDetSetVec>(edGetToken_, handle);
    // apply cabling map, reorganise cluster collections
    std::vector<std::vector<std::deque<TTClusterRef>>> dtcs(
        setup->numDTCs(), std::vector<std::deque<TTClusterRef>>(setup->numModulesPerDTC()));
    for (auto itModule = handle->begin(); itModule != handle->end(); itModule++) {
      // DetSetVec->detId - 1 or + 0 = tk layout det id depending from which of both sensor planes the cluster has been constructed
      const DetId& detIdModule = itModule->detId();
      const int offset = setup->trackerTopology()->isLower(detIdModule) ? 0 : setup->offsetDetIdTP();
      const DetId detId = detIdModule + offset;
      // corresponding sensor module
      tt::SensorModule* sm = setup->sensorModule(detId);
      // empty cluster collection
      std::deque<TTClusterRef>& module = dtcs[sm->dtcId()][sm->modId()];
      for (TTClusterDetSet::const_iterator itCluster = itModule->begin(); itCluster != itModule->end(); itCluster++)
        module.emplace_back(makeRefTo(handle, itCluster));
    }
    // analyze organized TTCluster collections
    int iDTC(0);
    int iModule(0);
    int nAll(0);
    for (const std::vector<std::deque<TTClusterRef>>& dtc : dtcs) {
      int nCluster(0);
      for (const std::deque<TTClusterRef>& module : dtc) {
        nCluster += module.size();
        hisModules_->Fill(module.size());
        profModules_->Fill(iModule++, module.size());
      }
      nAll += nCluster;
      hisDTCs_->Fill(nCluster);
      profDTCs_->Fill(iDTC++, nCluster);
    }
    hisTracker_->Fill(nAll);
  }

}  // namespace trackerDTC

DEFINE_FWK_MODULE(trackerDTC::AnalyzerDAQ);
