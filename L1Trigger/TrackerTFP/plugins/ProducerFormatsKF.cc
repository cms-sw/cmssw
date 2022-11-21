#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/ESInputTag.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "L1Trigger/TrackerTFP/interface/DataFormats.h"
#include "L1Trigger/TrackerTFP/interface/KalmanFilterFormats.h"

#include <memory>

using namespace std;
using namespace edm;

namespace trackerTFP {

  /*! \class  trackerTFP::ProducerFormatsKF
   *  \brief  Class to produce setup of Kalman Filter emulator data formats
   *  \author Thomas Schuh
   *  \date   2020, July
   */
  class ProducerFormatsKF : public ESProducer {
  public:
    ProducerFormatsKF(const ParameterSet& iConfig);
    ~ProducerFormatsKF() override {}
    unique_ptr<KalmanFilterFormats> produce(const KalmanFilterFormatsRcd& rcd);

  private:
    const ParameterSet iConfig_;
    ESGetToken<DataFormats, DataFormatsRcd> esGetToken_;
  };

  ProducerFormatsKF::ProducerFormatsKF(const ParameterSet& iConfig) : iConfig_(iConfig) {
    auto cc = setWhatProduced(this);
    esGetToken_ = cc.consumes();
  }

  unique_ptr<KalmanFilterFormats> ProducerFormatsKF::produce(const KalmanFilterFormatsRcd& rcd) {
    const DataFormats* dataFormats = &rcd.get(esGetToken_);
    return make_unique<KalmanFilterFormats>(iConfig_, dataFormats);
  }

}  // namespace trackerTFP

DEFINE_FWK_EVENTSETUP_MODULE(trackerTFP::ProducerFormatsKF);