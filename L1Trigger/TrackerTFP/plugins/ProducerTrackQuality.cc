#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "L1Trigger/TrackerTFP/interface/TrackQuality.h"

#include <memory>

using namespace std;
using namespace edm;

namespace trackerTFP {

  /*! \class  trackerTFP::ProducerTrackQuality
   *  \brief  Class to produce TrackQuality of Track Trigger emulators
   *  \author Thomas Schuh
   *  \date   2024, July
   */
  class ProducerTrackQuality : public ESProducer {
  public:
    ProducerTrackQuality(const ParameterSet& iConfig) : iConfig_(iConfig) {
      auto cc = setWhatProduced(this);
      esGetToken_ = cc.consumes();
    }
    ~ProducerTrackQuality() override {}
    unique_ptr<TrackQuality> produce(const TrackQualityRcd& trackQualityRcd) {
      const DataFormats* dataFormats = &trackQualityRcd.get(esGetToken_);
      return make_unique<TrackQuality>(iConfig_, dataFormats);
    }

  private:
    const ParameterSet iConfig_;
    ESGetToken<DataFormats, DataFormatsRcd> esGetToken_;
  };

}  // namespace trackerTFP

DEFINE_FWK_EVENTSETUP_MODULE(trackerTFP::ProducerTrackQuality);
