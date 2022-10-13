#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/ESInputTag.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "L1Trigger/TrackerTFP/interface/DataFormats.h"

#include <memory>

using namespace std;
using namespace edm;
using namespace tt;

namespace trackerTFP {

  /*! \class  trackerTFP::ProducerES
   *  \brief  Class to produce setup of Track Trigger emulator data formats
   *  \author Thomas Schuh
   *  \date   2020, June
   */
  class ProducerES : public ESProducer {
  public:
    ProducerES(const ParameterSet& iConfig);
    ~ProducerES() override {}
    unique_ptr<DataFormats> produce(const DataFormatsRcd& rcd);

  private:
    const ParameterSet iConfig_;
    ESGetToken<Setup, SetupRcd> esGetToken_;
  };

  ProducerES::ProducerES(const ParameterSet& iConfig) : iConfig_(iConfig) {
    auto cc = setWhatProduced(this);
    esGetToken_ = cc.consumes();
  }

  unique_ptr<DataFormats> ProducerES::produce(const DataFormatsRcd& rcd) {
    const Setup* setup = &rcd.get(esGetToken_);
    return make_unique<DataFormats>(iConfig_, setup);
  }

}  // namespace trackerTFP

DEFINE_FWK_EVENTSETUP_MODULE(trackerTFP::ProducerES);