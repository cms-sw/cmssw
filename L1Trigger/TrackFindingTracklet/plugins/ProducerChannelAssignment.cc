#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "L1Trigger/TrackFindingTracklet/interface/ChannelAssignment.h"
#include "L1Trigger/TrackTrigger/interface/Setup.h"

#include <memory>

using namespace std;
using namespace edm;
using namespace tt;

namespace trklet {

  /*! \class  trklet::ProducerChannelAssignment
   *  \brief  Creates ChannelAssignment class that assigns tracklet tracks and stubs
   *          to output channel as well as DTC stubs to input channel
   *  \author Thomas Schuh
   *  \date   2020, Nov
   */
  class ProducerChannelAssignment : public ESProducer {
  public:
    ProducerChannelAssignment(const ParameterSet& iConfig);
    ~ProducerChannelAssignment() override {}
    unique_ptr<ChannelAssignment> produce(const ChannelAssignmentRcd& rcd);

  private:
    ChannelAssignment::Config iConfig_;
    ESGetToken<Setup, SetupRcd> esGetToken_;
  };

  ProducerChannelAssignment::ProducerChannelAssignment(const ParameterSet& iConfig) {
    auto cc = setWhatProduced(this);
    esGetToken_ = cc.consumes();
    iConfig_.seedTypeNames_ = iConfig.getParameter<vector<string>>("SeedTypes");
    iConfig_.channelEncoding_ = iConfig.getParameter<vector<int>>("IRChannelsIn");
    const edm::ParameterSet& pSetTM = iConfig.getParameter<ParameterSet>("TM");
    iConfig_.tmMuxOrder_ = pSetTM.getParameter<vector<string>>("MuxOrder");
    iConfig_.tmNumLayers_ = pSetTM.getParameter<int>("NumLayers");
    iConfig_.tmWidthStubId_ = pSetTM.getParameter<int>("WidthStubId");
    iConfig_.tmWidthCot_ = pSetTM.getParameter<int>("WidthCot");
    const edm::ParameterSet& pSetDR = iConfig.getParameter<ParameterSet>("DR");
    iConfig_.numComparisonModules_ = pSetDR.getParameter<int>("NumComparisonModules");
    iConfig_.minIdenticalStubs_ = pSetDR.getParameter<int>("MinIdenticalStubs");
    iConfig_.tmMuxOrderInt_.reserve(iConfig_.tmMuxOrder_.size());
    for (const string& s : iConfig_.tmMuxOrder_)
      iConfig_.tmMuxOrderInt_.push_back(
          distance(iConfig_.tmMuxOrder_.begin(), find(iConfig_.tmMuxOrder_.begin(), iConfig_.tmMuxOrder_.end(), s)));
    const ParameterSet& pSetSeedTypesSeedLayers = iConfig.getParameter<ParameterSet>("SeedTypesSeedLayers");
    const ParameterSet& pSetSeedTypesProjectionLayers = iConfig.getParameter<ParameterSet>("SeedTypesProjectionLayers");
    iConfig_.seedTypesSeedLayers_.reserve(iConfig_.seedTypeNames_.size());
    iConfig_.seedTypesProjectionLayers_.reserve(iConfig_.seedTypeNames_.size());
    for (const string& s : iConfig_.seedTypeNames_) {
      iConfig_.seedTypesSeedLayers_.emplace_back(pSetSeedTypesSeedLayers.getParameter<vector<int>>(s));
      iConfig_.seedTypesProjectionLayers_.emplace_back(pSetSeedTypesProjectionLayers.getParameter<vector<int>>(s));
    }
  }

  unique_ptr<ChannelAssignment> ProducerChannelAssignment::produce(const ChannelAssignmentRcd& rcd) {
    const Setup* setup = &rcd.get(esGetToken_);
    return make_unique<ChannelAssignment>(iConfig_, setup);
  }

}  // namespace trklet

DEFINE_FWK_EVENTSETUP_MODULE(trklet::ProducerChannelAssignment);
