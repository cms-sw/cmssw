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

#include "DataFormats/EgammaCandidates/interface/HIPhotonIsolation.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include "RecoEcal/EgammaCoreTools/interface/EcalClusterLazyTools.h"

#include "RecoHI/HiEgammaAlgos/interface/EcalClusterIsoCalculator.h"
#include "RecoHI/HiEgammaAlgos/interface/HcalRechitIsoCalculator.h"
#include "RecoHI/HiEgammaAlgos/interface/TrackIsoCalculator.h"

class photonIsolationHIProducer : public edm::stream::EDProducer<> {

 public:

  explicit photonIsolationHIProducer (const edm::ParameterSet& ps);
  ~photonIsolationHIProducer();

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

 private:

  virtual void produce(edm::Event& evt, const edm::EventSetup& es) override;

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

photonIsolationHIProducer::photonIsolationHIProducer(const edm::ParameterSet& config)
  :
  photonProducer_   (
    consumes<reco::PhotonCollection>(config.getParameter<edm::InputTag>("photonProducer"))),
  barrelEcalHits_   (
    consumes<EcalRecHitCollection>(config.getParameter<edm::InputTag>("ebRecHitCollection"))),
  endcapEcalHits_   (
    consumes<EcalRecHitCollection>(config.getParameter<edm::InputTag>("eeRecHitCollection"))),
  hbhe_ ( consumes<HBHERecHitCollection>(config.getParameter<edm::InputTag>("hbhe"))),
  hf_ ( consumes<HFRecHitCollection>(config.getParameter<edm::InputTag>("hf"))),
  ho_ ( consumes<HORecHitCollection>(config.getParameter<edm::InputTag>("ho"))),
  barrelClusters_ ( consumes<reco::BasicClusterCollection>(config.getParameter<edm::InputTag>("basicClusterBarrel"))),
  endcapClusters_ ( consumes<reco::BasicClusterCollection>(config.getParameter<edm::InputTag>("basicClusterEndcap"))),
  tracks_ ( consumes<reco::TrackCollection>(config.getParameter<edm::InputTag>("trackCollection"))),
  trackQuality_ ( config.getParameter<std::string>("trackQuality"))
{
  produces< reco::HIPhotonIsolationMap >();
}

photonIsolationHIProducer::~photonIsolationHIProducer() {}

void
photonIsolationHIProducer::produce(edm::Event& evt, const edm::EventSetup& es)
{
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

  std::auto_ptr<reco::HIPhotonIsolationMap> outputMap (new reco::HIPhotonIsolationMap);
  reco::HIPhotonIsolationMap::Filler filler(*outputMap);
  std::vector<reco::HIPhotonIsolation> isoVector;

  EcalClusterIsoCalculator CxC(evt,es, barrelClusters, endcapClusters);
  HcalRechitIsoCalculator RxC(evt,es, hbhe, hf, ho);
  TrackIsoCalculator TxC(evt, es, trackCollection, trackQuality_);
  EcalClusterLazyTools lazyTool(evt, es, barrelEcalHits_, endcapEcalHits_);

  for (reco::PhotonCollection::const_iterator phoItr = photons->begin(); phoItr != photons->end(); ++phoItr)
  {
    reco::HIPhotonIsolation iso;
    // HI-style isolation info
    iso.ecalClusterIsoR1(CxC.getBkgSubEcalClusterIso(phoItr->superCluster(),1,0));
    iso.ecalClusterIsoR2(CxC.getBkgSubEcalClusterIso(phoItr->superCluster(),2,0));
    iso.ecalClusterIsoR3(CxC.getBkgSubEcalClusterIso(phoItr->superCluster(),3,0));
    iso.ecalClusterIsoR4(CxC.getBkgSubEcalClusterIso(phoItr->superCluster(),4,0));
    iso.ecalClusterIsoR5(CxC.getBkgSubEcalClusterIso(phoItr->superCluster(),5,0));

    iso.hcalRechitIsoR1(RxC.getBkgSubHcalRechitIso(phoItr->superCluster(),1,0));
    iso.hcalRechitIsoR2(RxC.getBkgSubHcalRechitIso(phoItr->superCluster(),2,0));
    iso.hcalRechitIsoR3(RxC.getBkgSubHcalRechitIso(phoItr->superCluster(),3,0));
    iso.hcalRechitIsoR4(RxC.getBkgSubHcalRechitIso(phoItr->superCluster(),4,0));
    iso.hcalRechitIsoR5(RxC.getBkgSubHcalRechitIso(phoItr->superCluster(),5,0));

    iso.trackIsoR1PtCut20(TxC.getBkgSubTrackIso(*phoItr,1,2));
    iso.trackIsoR2PtCut20(TxC.getBkgSubTrackIso(*phoItr,2,2));
    iso.trackIsoR3PtCut20(TxC.getBkgSubTrackIso(*phoItr,3,2));
    iso.trackIsoR4PtCut20(TxC.getBkgSubTrackIso(*phoItr,4,2));
    iso.trackIsoR5PtCut20(TxC.getBkgSubTrackIso(*phoItr,5,2));

    // ecal spike rejection info (seed timing)
    const reco::CaloClusterPtr  seed = phoItr->superCluster()->seed();
    const DetId &id = lazyTool.getMaximum(*seed).first;
    float time  = -999.;
    const EcalRecHitCollection & rechits = ( phoItr->isEB() ? *barrelEcalHits : *endcapEcalHits);
    EcalRecHitCollection::const_iterator it = rechits.find( id );
    if( it != rechits.end() ) {
      time = it->time();
    }
    iso.seedTime(time);

    // ecal spike rejectino info (swiss cross)
    float eMax = lazyTool.eMax(*seed);
    float eRight = lazyTool.eRight(*seed);
    float eLeft = lazyTool.eLeft(*seed);
    float eTop = lazyTool.eTop(*seed);
    float eBottom = lazyTool.eBottom(*seed);
    iso.swissCrx( 1 - (eRight + eLeft + eTop + eBottom)/eMax );

    isoVector.push_back(iso);
  }
  filler.insert(photons, isoVector.begin(), isoVector.end() );
  filler.fill();
  evt.put(outputMap);
}

void
photonIsolationHIProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("photonProducer",edm::InputTag("photons"));
  desc.add<edm::InputTag>("ebRecHitCollection",edm::InputTag("ecalRecHit:EcalRecHitsEB"));
  desc.add<edm::InputTag>("eeRecHitCollection",edm::InputTag("ecalRecHit:EcalRecHitsEE"));
  desc.add<edm::InputTag>("hbhe",edm::InputTag("hbhereco"));
  desc.add<edm::InputTag>("hf",edm::InputTag("hfreco"));
  desc.add<edm::InputTag>("ho",edm::InputTag("horeco"));
  desc.add<edm::InputTag>("basicClusterBarrel",edm::InputTag("islandBasicClusters:islandBarrelBasicClusters"));
  desc.add<edm::InputTag>("basicClusterEndcap",edm::InputTag("islandBasicClusters:islandEndcapBasicClusters"));
  desc.add<edm::InputTag>("trackCollection",edm::InputTag("hiGeneralTracks"));
  desc.add<std::string>("trackQuality","highPurity");

  descriptions.add("photonIsolationHIProducer", desc);
}

DEFINE_FWK_MODULE(photonIsolationHIProducer);
