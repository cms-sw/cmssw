// system include files
#include <memory>
#include <string>

// user include files
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include "CondFormats/HcalObjects/interface/HcalRespCorrs.h"
#include "CondFormats/DataRecord/interface/HcalRespCorrsRcd.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "RecoTracker/TrackProducer/interface/TrackProducerBase.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "FWCore/Utilities/interface/Exception.h"

//
// class declaration
//

namespace cms {

  class HitReCalibrator : public edm::one::EDProducer<> {
  public:
    explicit HitReCalibrator(const edm::ParameterSet&);
    ~HitReCalibrator() override;

    void beginJob() override;

    void produce(edm::Event&, const edm::EventSetup&) override;

  private:
    // ----------member data ---------------------------

    bool allowMissingInputs_;

    const edm::EDGetTokenT<HBHERecHitCollection> tok_hbhe_;
    const edm::EDGetTokenT<HORecHitCollection> tok_ho_;
    const edm::EDGetTokenT<HFRecHitCollection> tok_hf_;

    const edm::ESGetToken<HcalRespCorrs, HcalRespCorrsRcd> tok_resp_;
  };
}  // end namespace cms

namespace cms {

  HitReCalibrator::HitReCalibrator(const edm::ParameterSet& iConfig)
      : tok_hbhe_(consumes<HBHERecHitCollection>(iConfig.getParameter<edm::InputTag>("hbheInput"))),
        tok_ho_(consumes<HORecHitCollection>(iConfig.getParameter<edm::InputTag>("hoInput"))),
        tok_hf_(consumes<HFRecHitCollection>(iConfig.getParameter<edm::InputTag>("hfInput"))),
        tok_resp_(esConsumes<HcalRespCorrs, HcalRespCorrsRcd>()) {
    allowMissingInputs_ = true;

    //register your products

    produces<HBHERecHitCollection>("DiJetsHBHEReRecHitCollection");
    produces<HORecHitCollection>("DiJetsHOReRecHitCollection");
    produces<HFRecHitCollection>("DiJetsHFReRecHitCollection");
  }
  void HitReCalibrator::beginJob() {}

  HitReCalibrator::~HitReCalibrator() {}

  // ------------ method called to produce the data  ------------
  void HitReCalibrator::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
    auto miniDiJetsHBHERecHitCollection = std::make_unique<HBHERecHitCollection>();
    auto miniDiJetsHORecHitCollection = std::make_unique<HORecHitCollection>();
    auto miniDiJetsHFRecHitCollection = std::make_unique<HFRecHitCollection>();

    const HcalRespCorrs* jetRecalib = &iSetup.getData(tok_resp_);

    try {
      const edm::Handle<HBHERecHitCollection>& hbhe = iEvent.getHandle(tok_hbhe_);
      const HBHERecHitCollection Hithbhe = *(hbhe.product());
      for (HBHERecHitCollection::const_iterator hbheItr = Hithbhe.begin(); hbheItr != Hithbhe.end(); hbheItr++) {
        DetId id = hbheItr->detid();
        float recal;
        if (jetRecalib->exists(id))
          recal = jetRecalib->getValues(id)->getValue();
        else
          recal = 1.;
        float energy = hbheItr->energy();
        float time = hbheItr->time();
        HBHERecHit* hbhehit = new HBHERecHit(id, recal * energy, time);
        miniDiJetsHBHERecHitCollection->push_back(*hbhehit);
      }
    } catch (cms::Exception& e) {  // can't find it!
      if (!allowMissingInputs_) {
        edm::LogError("HitCalib") << "No HBHE collection ";
        throw e;
      }
    }

    try {
      const edm::Handle<HORecHitCollection>& ho = iEvent.getHandle(tok_ho_);
      const HORecHitCollection Hitho = *(ho.product());
      for (HORecHitCollection::const_iterator hoItr = Hitho.begin(); hoItr != Hitho.end(); hoItr++) {
        DetId id = hoItr->detid();
        float recal;
        if (jetRecalib->exists(id))
          recal = jetRecalib->getValues(id)->getValue();
        else
          recal = 1.;
        float energy = hoItr->energy();
        float time = hoItr->time();
        HORecHit* hohit = new HORecHit(id, recal * energy, time);
        miniDiJetsHORecHitCollection->push_back(*hohit);
      }
    } catch (cms::Exception& e) {  // can't find it!
      if (!allowMissingInputs_) {
        edm::LogError("HitCalib") << " No HO collection ";
        throw e;
      }
    }

    try {
      const edm::Handle<HFRecHitCollection>& hf = iEvent.getHandle(tok_hf_);
      const HFRecHitCollection Hithf = *(hf.product());
      for (HFRecHitCollection::const_iterator hfItr = Hithf.begin(); hfItr != Hithf.end(); hfItr++) {
        DetId id = hfItr->detid();
        float recal;
        if (jetRecalib->exists(id))
          recal = jetRecalib->getValues(id)->getValue();
        else
          recal = 1.;
        float energy = hfItr->energy();
        float time = hfItr->time();
        HFRecHit* hfhit = new HFRecHit(id, recal * energy, time);
        miniDiJetsHFRecHitCollection->push_back(*hfhit);
      }
    } catch (cms::Exception& e) {  // can't find it!
      if (!allowMissingInputs_)
        throw e;
    }

    //Put selected information in the event

    iEvent.put(std::move(miniDiJetsHBHERecHitCollection), "DiJetsHBHEReRecHitCollection");

    iEvent.put(std::move(miniDiJetsHORecHitCollection), "DiJetsHOReRecHitCollection");

    iEvent.put(std::move(miniDiJetsHFRecHitCollection), "DiJetsHFReRecHitCollection");
  }
}  // namespace cms

#include "FWCore/Framework/interface/MakerMacros.h"

using cms::HitReCalibrator;
DEFINE_FWK_MODULE(HitReCalibrator);
