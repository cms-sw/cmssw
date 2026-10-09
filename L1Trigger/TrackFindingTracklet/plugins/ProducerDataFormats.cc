#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/ESInputTag.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"
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
    ~ProducerDataFormats() override = default;
    std::unique_ptr<DataFormats> produce(const trackerDTC::SetupRcd& rcd);

  private:
    edm::ESGetToken<Setup, trackerDTC::SetupRcd> esGetToken_;
  };

  ProducerDataFormats::ProducerDataFormats(const edm::ParameterSet& iConfig) {
    auto cc = setWhatProduced(this);
    esGetToken_ = cc.consumes();
  }

  std::unique_ptr<DataFormats> ProducerDataFormats::produce(const trackerDTC::SetupRcd& rcd) {
    const Setup* setup = &rcd.get(esGetToken_);
    return std::make_unique<DataFormats>(setup);
  }

}  // namespace trklet

DEFINE_FWK_EVENTSETUP_MODULE(trklet::ProducerDataFormats);
