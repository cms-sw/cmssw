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
    const ParameterSet iConfig_;
    ESGetToken<Setup, SetupRcd> esGetToken_;
  };

  ProducerChannelAssignment::ProducerChannelAssignment(const ParameterSet& iConfig) : iConfig_(iConfig) {
    auto cc = setWhatProduced(this);
    esGetToken_ = cc.consumes();
  }

  unique_ptr<ChannelAssignment> ProducerChannelAssignment::produce(const ChannelAssignmentRcd& rcd) {
    const Setup* setup = &rcd.get(esGetToken_);
    return make_unique<ChannelAssignment>(iConfig_, setup);
  }

}  // namespace trklet

DEFINE_FWK_EVENTSETUP_MODULE(trklet::ProducerChannelAssignment);
