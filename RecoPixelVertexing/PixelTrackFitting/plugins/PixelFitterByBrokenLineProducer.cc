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
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelFitterByBrokenLine.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

class PixelFitterByBrokenLineProducer: public edm::global::EDProducer<> {
public:
  explicit PixelFitterByBrokenLineProducer(const edm::ParameterSet& iConfig)
  {
    produces<PixelFitter>();
  }
  ~PixelFitterByBrokenLineProducer() {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    descriptions.add("pixelFitterByBrokenLine", desc);
  }

private:
  virtual void produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;
};


void PixelFitterByBrokenLineProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  edm::ESHandle<MagneticField> fieldESH;
  iSetup.get<IdealMagneticFieldRecord>().get(fieldESH);

  auto impl = std::make_unique<PixelFitterByBrokenLine>(fieldESH.product());
  auto prod = std::make_unique<PixelFitter>(std::move(impl));
  iEvent.put(std::move(prod));
}

DEFINE_FWK_MODULE(PixelFitterByBrokenLineProducer);
