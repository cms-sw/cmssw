#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "L1Trigger/TrackFindingTracklet/interface/ChannelAssignment.h"
#include "L1Trigger/TrackTrigger/interface/Setup.h"

#include <memory>

namespace trklet {

  /*! \class  trklet::ProducerChannelAssignment
   *  \brief  Creates ChannelAssignment class that assigns tracklet tracks and stubs
   *          to output channel as well as DTC stubs to input channel
   *  \author Thomas Schuh
   *  \date   2020, Nov
   */
  class ProducerChannelAssignment : public edm::ESProducer {
  public:
    ProducerChannelAssignment(const edm::ParameterSet& iConfig);
    ~ProducerChannelAssignment() override {}
    std::unique_ptr<ChannelAssignment> produce(const ChannelAssignmentRcd& rcd);

  private:
    ChannelAssignment::Config iConfig_;
    edm::ESGetToken<tt::Setup, tt::SetupRcd> esGetToken_;
  };

  ProducerChannelAssignment::ProducerChannelAssignment(const edm::ParameterSet& iConfig) {
    auto cc = setWhatProduced(this);
    esGetToken_ = cc.consumes();
    iConfig_.seedTypeNames_ = iConfig.getParameter<std::vector<std::string>>("SeedTypes");
    iConfig_.channelEncoding_ = iConfig.getParameter<std::vector<int>>("IRChannelsIn");
    const edm::ParameterSet& pSetTM = iConfig.getParameter<edm::ParameterSet>("TM");
    iConfig_.tmMuxOrder_ = pSetTM.getParameter<std::vector<std::string>>("MuxOrder");
    iConfig_.tmNumLayers_ = pSetTM.getParameter<int>("NumLayers");
    iConfig_.tmWidthStubId_ = pSetTM.getParameter<int>("WidthStubId");
    iConfig_.tmWidthCot_ = pSetTM.getParameter<int>("WidthCot");
    const edm::ParameterSet& pSetDR = iConfig.getParameter<edm::ParameterSet>("DR");
    iConfig_.numComparisonModules_ = pSetDR.getParameter<int>("NumComparisonModules");
    iConfig_.minIdenticalStubs_ = pSetDR.getParameter<int>("MinIdenticalStubs");
    iConfig_.tmMuxOrderInt_.reserve(iConfig_.tmMuxOrder_.size());
    for (const std::string& s : iConfig_.tmMuxOrder_)
      iConfig_.tmMuxOrderInt_.push_back(std::distance(
          iConfig_.tmMuxOrder_.begin(), find(iConfig_.tmMuxOrder_.begin(), iConfig_.tmMuxOrder_.end(), s)));
    const edm::ParameterSet& pSetSeedTypesSeedLayers = iConfig.getParameter<edm::ParameterSet>("SeedTypesSeedLayers");
    const edm::ParameterSet& pSetSeedTypesProjectionLayers =
        iConfig.getParameter<edm::ParameterSet>("SeedTypesProjectionLayers");
    iConfig_.seedTypesSeedLayers_.reserve(iConfig_.seedTypeNames_.size());
    iConfig_.seedTypesProjectionLayers_.reserve(iConfig_.seedTypeNames_.size());
    for (const std::string& s : iConfig_.seedTypeNames_) {
      iConfig_.seedTypesSeedLayers_.emplace_back(pSetSeedTypesSeedLayers.getParameter<std::vector<int>>(s));
      iConfig_.seedTypesProjectionLayers_.emplace_back(pSetSeedTypesProjectionLayers.getParameter<std::vector<int>>(s));
    }
  }

  std::unique_ptr<ChannelAssignment> ProducerChannelAssignment::produce(const ChannelAssignmentRcd& rcd) {
    const tt::Setup* setup = &rcd.get(esGetToken_);
    return std::make_unique<ChannelAssignment>(iConfig_, setup);
  }

}  // namespace trklet

DEFINE_FWK_EVENTSETUP_MODULE(trklet::ProducerChannelAssignment);
