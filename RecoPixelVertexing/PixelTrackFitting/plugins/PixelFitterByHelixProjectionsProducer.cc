#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelFitter.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelFitterByHelixProjections.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

class PixelFitterByHelixProjectionsProducer: public edm::global::EDProducer<> {
public:
  explicit PixelFitterByHelixProjectionsProducer(const edm::ParameterSet& iConfig)
    : thescaleErrorsForBPix1(iConfig.getParameter<bool>("scaleErrorsForBPix1"))
    , thescaleFactor(iConfig.getParameter<double>("scaleFactor"))  {
    produces<PixelFitter>();
  }
  ~PixelFitterByHelixProjectionsProducer() override {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<bool>("scaleErrorsForBPix1", false);
    desc.add<double>("scaleFactor", 0.65)->setComment("The default value was derived for phase1 pixel");
    descriptions.add("pixelFitterByHelixProjectionsDefault", desc);
  }

private:
  void produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;
  const bool thescaleErrorsForBPix1;
  const float thescaleFactor;
};


void PixelFitterByHelixProjectionsProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  edm::ESHandle<MagneticField> fieldESH;
  iSetup.get<IdealMagneticFieldRecord>().get(fieldESH);

  auto impl = std::make_unique<PixelFitterByHelixProjections>(&iSetup, fieldESH.product(), thescaleErrorsForBPix1, thescaleFactor);
  auto prod = std::make_unique<PixelFitter>(std::move(impl));
  iEvent.put(std::move(prod));
}

DEFINE_FWK_MODULE(PixelFitterByHelixProjectionsProducer);
