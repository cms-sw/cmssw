#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/ESInputTag.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "L1Trigger/TrackFindingTracklet/interface/ChannelAssignment.h"
#include "L1Trigger/TrackFindingTracklet/interface/DataFormats.h"

#include <memory>

namespace trklet {

  /*! \class  trklet::ProducerDataFormats
   *  \brief  Class to produce setup of Hybrid emulator data formats
   *  \author Thomas Schuh
   *  \date   2024, Sep
   */
  class ProducerDataFormats : public edm::ESProducer {
  public:
    ProducerDataFormats(const edm::ParameterSet& iConfig);
    ~ProducerDataFormats() override {}
    std::unique_ptr<DataFormats> produce(const ChannelAssignmentRcd& rcd);

  private:
    edm::ESGetToken<ChannelAssignment, ChannelAssignmentRcd> esGetToken_;
  };

  ProducerDataFormats::ProducerDataFormats(const edm::ParameterSet& iConfig) {
    auto cc = setWhatProduced(this);
    esGetToken_ = cc.consumes();
  }

  std::unique_ptr<DataFormats> ProducerDataFormats::produce(const ChannelAssignmentRcd& rcd) {
    const ChannelAssignment* ca = &rcd.get(esGetToken_);
    return std::make_unique<DataFormats>(ca);
  }

}  // namespace trklet

DEFINE_FWK_EVENTSETUP_MODULE(trklet::ProducerDataFormats);
