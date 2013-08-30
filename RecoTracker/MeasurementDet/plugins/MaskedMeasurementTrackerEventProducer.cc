#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTrackerEvent.h"
#include "DataFormats/DetId/interface/DetIdCollection.h"

class MaskedMeasurementTrackerEventProducer : public edm::EDProducer {
public:
      explicit MaskedMeasurementTrackerEventProducer(const edm::ParameterSet &iConfig) ;
      ~MaskedMeasurementTrackerEventProducer() {}
private:
      virtual void produce(edm::Event&, const edm::EventSetup&);

      typedef edm::ContainerMask<edmNew::DetSetVector<SiStripCluster> > StripMask;
      typedef edm::ContainerMask<edmNew::DetSetVector<SiPixelCluster> > PixelMask;
      typedef edm::ContainerMask<edm::LazyGetter<SiStripCluster> >      StripLazyMask;

      edm::EDGetTokenT<MeasurementTrackerEvent> src_;
      bool onDemand_;

      edm::EDGetTokenT<StripMask> maskStrips_;
      edm::EDGetTokenT<PixelMask> maskPixels_;
      edm::EDGetTokenT<StripLazyMask> maskStripsLazy_;
};


MaskedMeasurementTrackerEventProducer::MaskedMeasurementTrackerEventProducer(const edm::ParameterSet &iConfig) :
    src_(consumes<MeasurementTrackerEvent>(iConfig.getParameter<edm::InputTag>("src"))),
    onDemand_(iConfig.getParameter<bool>("OnDemand"))
{
    edm::InputTag clustersToSkip = iConfig.getParameter<edm::InputTag>("clustersToSkip");
    if (onDemand_) {
        maskStripsLazy_ = consumes<StripLazyMask>(clustersToSkip);
    } else {
        maskStrips_ = consumes<StripMask>(clustersToSkip);
    }
    maskPixels_ = consumes<PixelMask>(clustersToSkip);

    produces<MeasurementTrackerEvent>();
}

void
MaskedMeasurementTrackerEventProducer::produce(edm::Event &iEvent, const edm::EventSetup& iSetup)
{
    edm::Handle<MeasurementTrackerEvent> mte;
    iEvent.getByToken(src_, mte);

    // prepare output
    std::auto_ptr<MeasurementTrackerEvent> out;

    edm::Handle<PixelMask> maskPixels;
    iEvent.getByToken(maskPixels_, maskPixels);

    if (onDemand_) {
        edm::Handle<StripLazyMask> maskStripsLazy;
        iEvent.getByToken(maskStripsLazy_, maskStripsLazy);
        out.reset(new MeasurementTrackerEvent(*mte, *maskStripsLazy, *maskPixels));
    } else {
        edm::Handle<StripMask> maskStrips;
        iEvent.getByToken(maskStrips_, maskStrips);
        out.reset(new MeasurementTrackerEvent(*mte, *maskStrips, *maskPixels));
    }

    // put into event
    iEvent.put(out);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(MaskedMeasurementTrackerEventProducer);
