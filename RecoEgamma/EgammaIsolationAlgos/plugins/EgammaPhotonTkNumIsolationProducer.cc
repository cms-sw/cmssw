//*****************************************************************************
// File:      EgammaPhotonTkNumIsolationProducer.cc
// ----------------------------------------------------------------------------
// OrigAuth:  Matthias Mozer
// Institute: IIHE-VUB
//=============================================================================
//*****************************************************************************

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Candidate/interface/CandAssociation.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/PhotonTkIsolation.h"

class EgammaPhotonTkNumIsolationProducer : public edm::global::EDProducer<> {
public:
  explicit EgammaPhotonTkNumIsolationProducer(const edm::ParameterSet&);
  ~EgammaPhotonTkNumIsolationProducer() override;

  void produce(edm::StreamID sid, edm::Event&, const edm::EventSetup&) const override;

private:
  const edm::InputTag photonProducer_;
  const edm::InputTag trackProducer_;
  const edm::InputTag beamspotProducer_;

  const double ptMin_;
  const double intRadiusBarrel_;
  const double intRadiusEndcap_;
  const double stripBarrel_;
  const double stripEndcap_;
  const double extRadius_;
  const double maxVtxDist_;
  const double drb_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(EgammaPhotonTkNumIsolationProducer);

EgammaPhotonTkNumIsolationProducer::EgammaPhotonTkNumIsolationProducer(const edm::ParameterSet& config)
    :

      // use configuration file to setup input/output collection names
      photonProducer_(config.getParameter<edm::InputTag>("photonProducer")),

      trackProducer_(config.getParameter<edm::InputTag>("trackProducer")),
      beamspotProducer_(config.getParameter<edm::InputTag>("BeamspotProducer")),

      ptMin_(config.getParameter<double>("ptMin")),
      intRadiusBarrel_(config.getParameter<double>("intRadiusBarrel")),
      intRadiusEndcap_(config.getParameter<double>("intRadiusEndcap")),
      stripBarrel_(config.getParameter<double>("stripBarrel")),
      stripEndcap_(config.getParameter<double>("stripEndcap")),
      extRadius_(config.getParameter<double>("extRadius")),
      maxVtxDist_(config.getParameter<double>("maxVtxDist")),
      drb_(config.getParameter<double>("maxVtxDistXY"))

{
  //register your products
  produces<edm::ValueMap<int>>();
}

EgammaPhotonTkNumIsolationProducer::~EgammaPhotonTkNumIsolationProducer() {}

//
// member functions
//

// ------------ method called to produce the data  ------------
void EgammaPhotonTkNumIsolationProducer::produce(edm::StreamID sid,
                                                 edm::Event& iEvent,
                                                 const edm::EventSetup& iSetup) const {
  // Get the  filtered objects
  edm::Handle<edm::View<reco::Candidate>> photonHandle;
  iEvent.getByLabel(photonProducer_, photonHandle);

  //get the tracks
  edm::Handle<reco::TrackCollection> tracks;
  iEvent.getByLabel(trackProducer_, tracks);
  const reco::TrackCollection* trackCollection = tracks.product();

  //get beamspot
  edm::Handle<reco::BeamSpot> beamSpotH;
  iEvent.getByLabel(beamspotProducer_, beamSpotH);
  reco::TrackBase::Point beamspot = beamSpotH->position();

  //prepare product
  auto isoMap = std::make_unique<edm::ValueMap<int>>();
  edm::ValueMap<int>::Filler filler(*isoMap);
  std::vector<int> retV(photonHandle->size(), 0);

  PhotonTkIsolation myTkIsolation(extRadius_,
                                  intRadiusBarrel_,
                                  intRadiusEndcap_,
                                  stripBarrel_,
                                  stripEndcap_,
                                  ptMin_,
                                  maxVtxDist_,
                                  drb_,
                                  trackCollection,
                                  beamspot);

  for (unsigned int i = 0; i < photonHandle->size(); ++i) {
    int isoValue = myTkIsolation.getIso(&(photonHandle->at(i))).first;
    retV[i] = isoValue;
  }

  //fill and insert valuemap
  filler.insert(photonHandle, retV.begin(), retV.end());
  filler.fill();
  iEvent.put(std::move(isoMap));
}
