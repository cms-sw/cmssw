#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "L1Trigger/TrackFindingTracklet/interface/TrackBuilderChannel.h"
#include "L1Trigger/TrackTrigger/interface/Setup.h"

#include <memory>

using namespace std;
using namespace edm;
using namespace tt;

namespace trackFindingTracklet {

  /*! \class  trackFindingTracklet::ProducerTrackBuilderChannel
   *  \brief  Creates TrackBuilderChannel class that assigns tracklet tracks to channel
   *  \author Thomas Schuh
   *  \date   2020, Nov
   */
  class ProducerTrackBuilderChannel : public ESProducer {
  public:
    ProducerTrackBuilderChannel(const ParameterSet& iConfig);
    ~ProducerTrackBuilderChannel() override {}
    unique_ptr<TrackBuilderChannel> produce(const TrackBuilderChannelRcd& rcd);

  private:
    const ParameterSet* iConfig_;
    ESGetToken<Setup, SetupRcd> esGetToken_;
  };

  ProducerTrackBuilderChannel::ProducerTrackBuilderChannel(const ParameterSet& iConfig) : iConfig_(&iConfig) {
    auto cc = setWhatProduced(this);
    esGetToken_ = cc.consumes();
  }

  unique_ptr<TrackBuilderChannel> ProducerTrackBuilderChannel::produce(const TrackBuilderChannelRcd& rcd) {
    return make_unique<TrackBuilderChannel>(*iConfig_);
  }

}  // namespace trackFindingTracklet

DEFINE_FWK_EVENTSETUP_MODULE(trackFindingTracklet::ProducerTrackBuilderChannel);