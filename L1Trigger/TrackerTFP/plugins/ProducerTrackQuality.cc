#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "L1Trigger/TrackerTFP/interface/TrackQuality.h"

#include <memory>

namespace trackerTFP {

  /*! \class  trackerTFP::ProducerTrackQuality
   *  \brief  Class to produce TrackQuality of Track Trigger emulators
   *  \author Thomas Schuh
   *  \date   2024, July
   */
  class ProducerTrackQuality : public edm::ESProducer {
  public:
    ProducerTrackQuality(const edm::ParameterSet& iConfig) {
      auto cc = setWhatProduced(this);
      esGetToken_ = cc.consumes();
      iConfig_.model_ = iConfig.getParameter<edm::FileInPath>("Model");
      iConfig_.featureNames_ = iConfig.getParameter<std::vector<std::string>>("FeatureNames");
      iConfig_.baseShiftCot_ = iConfig.getParameter<int>("BaseShiftCot");
      iConfig_.baseShiftZ0_ = iConfig.getParameter<int>("BaseShiftZ0");
      iConfig_.baseShiftAPfixed_ = iConfig.getParameter<int>("BaseShiftAPfixed");
      iConfig_.chi2rphiConv_ = iConfig.getParameter<int>("Chi2rphiConv");
      iConfig_.chi2rzConv_ = iConfig.getParameter<int>("Chi2rzConv");
      iConfig_.weightBinFraction_ = iConfig.getParameter<int>("WeightBinFraction");
      iConfig_.dzTruncation_ = iConfig.getParameter<int>("DzTruncation");
      iConfig_.dphiTruncation_ = iConfig.getParameter<int>("DphiTruncation");
      iConfig_.widthM20_ = iConfig.getParameter<int>("WidthM20");
      iConfig_.widthM21_ = iConfig.getParameter<int>("WidthM21");
      iConfig_.widthInvV0_ = iConfig.getParameter<int>("WidthInvV0");
      iConfig_.widthInvV1_ = iConfig.getParameter<int>("WidthInvV1");
      iConfig_.widthchi2rphi_ = iConfig.getParameter<int>("Widthchi2rphi");
      iConfig_.widthchi2rz_ = iConfig.getParameter<int>("Widthchi2rz");
      iConfig_.baseShiftchi2rphi_ = iConfig.getParameter<int>("BaseShiftchi2rphi");
      iConfig_.baseShiftchi2rz_ = iConfig.getParameter<int>("BaseShiftchi2rz");
    }
    ~ProducerTrackQuality() override {}
    std::unique_ptr<TrackQuality> produce(const DataFormatsRcd& rcd) {
      const DataFormats* dataFormats = &rcd.get(esGetToken_);
      return std::make_unique<TrackQuality>(iConfig_, dataFormats);
    }

  private:
    ConfigTQ iConfig_;
    edm::ESGetToken<DataFormats, DataFormatsRcd> esGetToken_;
  };

}  // namespace trackerTFP

DEFINE_FWK_EVENTSETUP_MODULE(trackerTFP::ProducerTrackQuality);
