#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/EgammaCandidates/interface/AODHIPhoton.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include "RecoEcal/EgammaCoreTools/interface/EcalClusterLazyTools.h"

#include "RecoHI/HiEgammaAlgos/interface/CxCalculator.h"
#include "RecoHI/HiEgammaAlgos/interface/RxCalculator.h"
#include "RecoHI/HiEgammaAlgos/interface/TxCalculator.h"

#ifndef AODHIPhotonProducer_h
#define AODHIPhotonProducer_h

class AODHIPhotonProducer : public edm::stream::EDProducer<> {

 public:

  explicit AODHIPhotonProducer (const edm::ParameterSet& ps);
  ~AODHIPhotonProducer();

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

 private:

  virtual void beginRun (edm::Run const& r, edm::EventSetup const & es) override final;
  virtual void endRun(edm::Run const&,  edm::EventSetup const&) override final;
  virtual void produce(edm::Event& evt, const edm::EventSetup& es);

  edm::EDGetTokenT<reco::PhotonCollection> photonProducer_;
  edm::EDGetTokenT<EcalRecHitCollection> barrelEcalHits_;
  edm::EDGetTokenT<EcalRecHitCollection> endcapEcalHits_;
  edm::EDGetTokenT<HBHERecHitCollection> hbhe_;
  edm::EDGetTokenT<HFRecHitCollection> hf_;
  edm::EDGetTokenT<HORecHitCollection> ho_;
  edm::EDGetTokenT<reco::BasicClusterCollection> barrelClusters_;
  edm::EDGetTokenT<reco::BasicClusterCollection> endcapClusters_;
  edm::EDGetTokenT<reco::TrackCollection> tracks_;

  std::string trackQuality_;

};

#endif

AODHIPhotonProducer::AODHIPhotonProducer(const edm::ParameterSet& config)
{
  photonProducer_   =
    consumes<reco::PhotonCollection>(config.getParameter<edm::InputTag>("photonProducer"));
  barrelEcalHits_   =
    consumes<EcalRecHitCollection>(config.getParameter<edm::InputTag>("ebRecHitCollection"));
  endcapEcalHits_   =
    consumes<EcalRecHitCollection>(config.getParameter<edm::InputTag>("eeRecHitCollection"));
  hbhe_ = consumes<HBHERecHitCollection>(config.getParameter<edm::InputTag>("hbhe"));
  hf_ = consumes<HFRecHitCollection>(config.getParameter<edm::InputTag>("hf"));
  ho_ = consumes<HORecHitCollection>(config.getParameter<edm::InputTag>("ho"));
  barrelClusters_ = consumes<reco::BasicClusterCollection>(config.getParameter<edm::InputTag>("basicClusterBarrel"));
  endcapClusters_ = consumes<reco::BasicClusterCollection>(config.getParameter<edm::InputTag>("basicClusterEndcap"));
  tracks_ = consumes<reco::TrackCollection>(config.getParameter<edm::InputTag>("trackCollection"));
  trackQuality_ = config.getParameter<std::string>("trackQuality");

  produces< aod::AODHIPhotonCollection >();
}

AODHIPhotonProducer::~AODHIPhotonProducer() {}

void
AODHIPhotonProducer::produce(edm::Event& evt, const edm::EventSetup& es)
{
  std::auto_ptr<aod::AODHIPhotonCollection> outputAODHIPhotonCollection (new aod::AODHIPhotonCollection);

  edm::Handle<reco::PhotonCollection> photons;
  evt.getByToken(photonProducer_, photons);
  edm::Handle<EcalRecHitCollection> barrelEcalHits;
  evt.getByToken(barrelEcalHits_, barrelEcalHits);
  edm::Handle<EcalRecHitCollection> endcapEcalHits;
  evt.getByToken(endcapEcalHits_, endcapEcalHits);
  edm::Handle<HBHERecHitCollection> hbhe;
  evt.getByToken(hbhe_, hbhe);
  edm::Handle<HFRecHitCollection> hf;
  evt.getByToken(hf_, hf);
  edm::Handle<HORecHitCollection> ho;
  evt.getByToken(ho_, ho);
  edm::Handle<reco::BasicClusterCollection> barrelClusters;
  evt.getByToken(barrelClusters_, barrelClusters);
  edm::Handle<reco::BasicClusterCollection> endcapClusters;
  evt.getByToken(endcapClusters_, endcapClusters);
  edm::Handle<reco::TrackCollection> trackCollection;
  evt.getByToken(tracks_, trackCollection);

  CxCalculator CxC(evt,es, barrelClusters, endcapClusters);
  RxCalculator RxC(evt,es, hbhe, hf, ho);
  TxCalculator TxC(evt, es, trackCollection, trackQuality_);
  EcalClusterLazyTools lazyTool(evt, es, barrelEcalHits_, endcapEcalHits_);

  for (reco::PhotonCollection::const_iterator phoItr = photons->begin(); phoItr != photons->end(); ++phoItr) {
    aod::AODHIPhoton newphoton(*phoItr);

    // HI-style isolation info
    newphoton.setcc1(CxC.getCCx(newphoton.superCluster(),1,0));
    newphoton.setcc2(CxC.getCCx(newphoton.superCluster(),2,0));
    newphoton.setcc3(CxC.getCCx(newphoton.superCluster(),3,0));
    newphoton.setcc4(CxC.getCCx(newphoton.superCluster(),4,0));
    newphoton.setcc5(CxC.getCCx(newphoton.superCluster(),5,0));

    newphoton.setcr1(RxC.getCRx(newphoton.superCluster(),1,0));
    newphoton.setcr2(RxC.getCRx(newphoton.superCluster(),2,0));
    newphoton.setcr3(RxC.getCRx(newphoton.superCluster(),3,0));
    newphoton.setcr4(RxC.getCRx(newphoton.superCluster(),4,0));
    newphoton.setcr5(RxC.getCRx(newphoton.superCluster(),5,0));

    newphoton.setct1PtCut20(TxC.getCTx(newphoton,1,2));
    newphoton.setct2PtCut20(TxC.getCTx(newphoton,2,2));
    newphoton.setct3PtCut20(TxC.getCTx(newphoton,3,2));
    newphoton.setct4PtCut20(TxC.getCTx(newphoton,4,2));
    newphoton.setct5PtCut20(TxC.getCTx(newphoton,5,2));

    // ecal spike rejection info (seed timing)
    const reco::CaloClusterPtr  seed = newphoton.superCluster()->seed();
    const DetId &id = lazyTool.getMaximum(*seed).first;
    float time  = -999.;
    const EcalRecHitCollection & rechits = ( newphoton.isEB() ? *barrelEcalHits : *endcapEcalHits);
    EcalRecHitCollection::const_iterator it = rechits.find( id );
    if( it != rechits.end() ) {
      time = it->time();
    }
    newphoton.seedTime(time);

    // ecal spike rejectino info (swiss cross)
    float eMax = lazyTool.eMax(*seed);
    float eRight = lazyTool.eRight(*seed);
    float eLeft = lazyTool.eLeft(*seed);
    float eTop = lazyTool.eTop(*seed);
    float eBottom = lazyTool.eBottom(*seed);
    newphoton.swissCrx( 1 - (eRight + eLeft + eTop + eBottom)/eMax );

    outputAODHIPhotonCollection->push_back(newphoton);
  }

  evt.put(outputAODHIPhotonCollection);
}

void AODHIPhotonProducer::beginRun (edm::Run const& r, edm::EventSetup const & es) {}
void AODHIPhotonProducer::endRun(edm::Run const&,  edm::EventSetup const&) {}

void
AODHIPhotonProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

DEFINE_FWK_MODULE(AODHIPhotonProducer);
