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
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelFitterByRiemannParaboloid.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

class PixelFitterByRiemannParaboloidProducer: public edm::global::EDProducer<> {
public:
  explicit PixelFitterByRiemannParaboloidProducer(const edm::ParameterSet& iConfig)
    : useErrors_(iConfig.getParameter<bool>("useErrors")),
    useMultipleScattering_(iConfig.getParameter<bool>("useMultipleScattering"))
  {
    produces<PixelFitter>();
  }
  ~PixelFitterByRiemannParaboloidProducer() override {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<bool>("useErrors", true);
    desc.add<bool>("useMultipleScattering", true);
    descriptions.add("pixelFitterByRiemannParaboloidDefault", desc);
  }

private:
  bool useErrors_;
  bool useMultipleScattering_;
  void produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;
};


void PixelFitterByRiemannParaboloidProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  edm::ESHandle<MagneticField> fieldESH;
  iSetup.get<IdealMagneticFieldRecord>().get(fieldESH);

  auto impl = std::make_unique<PixelFitterByRiemannParaboloid>(&iSetup,
      fieldESH.product(), useErrors_, useMultipleScattering_);
  auto prod = std::make_unique<PixelFitter>(std::move(impl));
  iEvent.put(std::move(prod));
}

DEFINE_FWK_MODULE(PixelFitterByRiemannParaboloidProducer);
