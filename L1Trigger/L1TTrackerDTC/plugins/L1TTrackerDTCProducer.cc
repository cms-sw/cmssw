#include "L1Trigger/L1TTrackerDTC/plugins/L1TTrackerDTCProducer.h"
#include "L1Trigger/L1TTrackerDTC/interface/Settings.h"
#include "L1Trigger/L1TTrackerDTC/interface/Module.h"
#include "L1Trigger/L1TTrackerDTC/interface/DTC.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "FWCore/Framework/interface/GenericHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include <memory>
#include <numeric>


using namespace std;
using namespace edm;

namespace L1TTrackerDTC {

  L1TTrackerDTCProducer::L1TTrackerDTCProducer(const ParameterSet& iConfig) {
    // helper class to store configurations
    settings_ = new Settings(iConfig);

    // writes ED products from disc to ram
    tokenTTStubDetSetVec_ = consumes<TTStubDetSetVec>(settings_->inputTagTTStubDetSetVec());

    // book ED output product
    produces<TTDTC>(settings_->productBranch());

    // collection of outer tracker sensor modules organised in DTCS [0-215][0-71]
    modules_ = vector<vector<Module*> >(settings_->numDTCs(), vector<Module*>(settings_->numModulesPerDTC(), nullptr));
  }

  L1TTrackerDTCProducer::~L1TTrackerDTCProducer() {
    for (auto& dtc : modules_)
      for (auto& module : dtc)
        delete module;
    delete settings_;
  }

  void L1TTrackerDTCProducer::beginRun(const Run& iRun, const EventSetup& iSetup) {
    // read in detector parameter
    settings_->beginRun(iRun, iSetup);

    // outer tracker sensor to DTC cabling map, index = module id [0-15551]
    const vector<DetId>& cablingMap = settings_->cablingMap();

    // prepare handy outer tracker geometry representation
    int modId(0);
    for (auto& dtc : modules_) {
      for (auto& module : dtc) {
        const DetId& detId = cablingMap[modId++];
        if (!detId.null())
          module = move(new Module(settings_, detId, modId));
      }
    }
  }

  void L1TTrackerDTCProducer::produce(Event& iEvent, const EventSetup& iSetup) {
    // empty DTC product
    TTDTC product(settings_->numRegions(), settings_->numOverlappingRegions(), settings_->numDTCsPerRegion());

    Handle<TTStubDetSetVec> handleTTStubDetSetVec;
    iEvent.getByToken(tokenTTStubDetSetVec_, handleTTStubDetSetVec);
    const TTStubDetSetVec::const_iterator end = handleTTStubDetSetVec->end();
    function<int(int&, TTStubDetSetVec::const_iterator)> acc =
        [handleTTStubDetSetVec, end](int& sum, const TTStubDetSetVec::const_iterator& channel) {
          return sum += channel != end ? channel->size() : 0;
        };

    // outer tracker sensor to DTC cabling map, value = det id [0-2**32-1], index = module id [0-15551]
    const vector<DetId>& cablingMap = settings_->cablingMap();

    // apply cabling map
    vector<vector<TTStubDetSetVec::const_iterator> > ttDTCs(
        settings_->numDTCs(), vector<TTStubDetSetVec::const_iterator>(settings_->numModulesPerDTC(), end));
    for (TTStubDetSetVec::const_iterator ttModule = handleTTStubDetSetVec->begin();
         ttModule != handleTTStubDetSetVec->end();
         ttModule++) {  // outer tracker modules

      const DetId detId = ttModule->detId() + settings_->offsetDetIdDSV();  // DetSetVec->detId + 1 = tk layout det id
      const int modId = distance(
          cablingMap.begin(), find(cablingMap.begin(), cablingMap.end(), detId));  // outer tracker module id [0-15551]

      if (modId == settings_->numModules()) {
        cms::Exception exception("Configuration", "Unknown DetID received from TTStub.");
        exception.addAdditionalInfo("Please check consistency between chosen cabling map and chosen tracker geometry.");
        exception.addContext("L1TTrackerDTC::Producer::produce");

        throw exception;
      }

      const int dtcId = modId / settings_->numModulesPerDTC();      // outer tracker dtc id [0-215]
      const int channelId = modId % settings_->numModulesPerDTC();  // outer tracker dtc channel id [0-71]

      ttDTCs[dtcId][channelId] = ttModule;
    }

    // read in and convert event content
    for (int dtcId = 0; dtcId < settings_->numDTCs(); dtcId++) {  // loop over outer tracker DTCs

      const vector<Module*>& modules = modules_[dtcId];
      const vector<TTStubDetSetVec::const_iterator>& ttDTC = ttDTCs[dtcId];

      const int nSubs = accumulate(ttDTC.begin(), ttDTC.end(), 0, acc);

      DTC dtc(settings_, dtcId, modules, nSubs);  // single outer tracker DTC board

      // fill incoming stubs
      for (int channelId = 0; channelId < settings_->numModulesPerDTC();
           channelId++) {  // loop over outer tracker DTC input channel

        const TTStubDetSetVec::const_iterator& ttModule = ttDTC[channelId];
        if (ttModule == end)
          continue;

        vector<TTStubRef> ttStubRefs;  // TTStubRefs from one module
        ttStubRefs.reserve(ttModule->size());
        for (TTStubDetSet::const_iterator ttStub = ttModule->begin(); ttStub != ttModule->end(); ttStub++)
          ttStubRefs.push_back(move(makeRefTo(handleTTStubDetSetVec, ttStub)));

        if (settings_->enableTruncation())  // truncate incoming stubs if desired
          ttStubRefs.resize(min((int)ttStubRefs.size(), settings_->maxFramesChannelInput()));

        dtc.consume(ttStubRefs, channelId);
      }

      // route stubs and fill product
      dtc.produce(product);
    }

    // store ED product
    iEvent.put(move(make_unique<TTDTC>(product)), settings_->productBranch());
  }

  DEFINE_FWK_MODULE(L1TTrackerDTCProducer);

}  // namespace L1TTrackerDTC