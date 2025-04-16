#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/ESInputTag.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "L1Trigger/TrackerTFP/interface/DataFormats.h"

#include <memory>

namespace trackerTFP {

  /*! \class  trackerTFP::ProducerDataFormats
   *  \brief  Class to produce setup of Track Trigger emulator data formats
   *  \author Thomas Schuh
   *  \date   2020, June
   */
  class ProducerDataFormats : public edm::ESProducer {
  public:
    ProducerDataFormats(const edm::ParameterSet& iConfig);
    ~ProducerDataFormats() override {}
    std::unique_ptr<DataFormats> produce(const DataFormatsRcd& rcd);

  private:
    edm::ESGetToken<tt::Setup, tt::SetupRcd> esGetToken_;
  };

  ProducerDataFormats::ProducerDataFormats(const edm::ParameterSet& iConfig) {
    auto cc = setWhatProduced(this);
    esGetToken_ = cc.consumes();
  }

  std::unique_ptr<DataFormats> ProducerDataFormats::produce(const DataFormatsRcd& rcd) {
    const tt::Setup* setup = &rcd.get(esGetToken_);
    return std::make_unique<DataFormats>(setup);
  }

}  // namespace trackerTFP

DEFINE_FWK_EVENTSETUP_MODULE(trackerTFP::ProducerDataFormats);
