//*****************************************************************************
// File:      EgammaElectronTkIsolationProducer.cc
// ----------------------------------------------------------------------------
// OrigAuth:  Matthias Mozer
// Institute: IIHE-VUB
//=============================================================================
//*****************************************************************************

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Candidate/interface/CandAssociation.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/ElectronTkIsolation.h"

class EgammaElectronTkIsolationProducer : public edm::stream::EDProducer<> {
public:
  explicit EgammaElectronTkIsolationProducer(const edm::ParameterSet&);
  ~EgammaElectronTkIsolationProducer() override;

  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  edm::InputTag electronProducer_;
  edm::InputTag trackProducer_;
  edm::InputTag beamspotProducer_;

  double ptMin_;
  double intRadiusBarrel_;
  double intRadiusEndcap_;
  double stripBarrel_;
  double stripEndcap_;
  double extRadius_;
  double maxVtxDist_;
  double drb_;

  edm::ParameterSet conf_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(EgammaElectronTkIsolationProducer);

EgammaElectronTkIsolationProducer::EgammaElectronTkIsolationProducer(const edm::ParameterSet& config) : conf_(config) {
  // use configuration file to setup input/output collection names
  electronProducer_ = conf_.getParameter<edm::InputTag>("electronProducer");

  trackProducer_ = conf_.getParameter<edm::InputTag>("trackProducer");
  beamspotProducer_ = conf_.getParameter<edm::InputTag>("BeamspotProducer");

  ptMin_ = conf_.getParameter<double>("ptMin");
  intRadiusBarrel_ = conf_.getParameter<double>("intRadiusBarrel");
  intRadiusEndcap_ = conf_.getParameter<double>("intRadiusEndcap");
  stripBarrel_ = conf_.getParameter<double>("stripBarrel");
  stripEndcap_ = conf_.getParameter<double>("stripEndcap");
  extRadius_ = conf_.getParameter<double>("extRadius");
  maxVtxDist_ = conf_.getParameter<double>("maxVtxDist");
  drb_ = conf_.getParameter<double>("maxVtxDistXY");

  //register your products
  produces<edm::ValueMap<double>>();
}

EgammaElectronTkIsolationProducer::~EgammaElectronTkIsolationProducer() {}

void EgammaElectronTkIsolationProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // Get the  filtered objects
  edm::Handle<reco::GsfElectronCollection> electronHandle;
  iEvent.getByLabel(electronProducer_, electronHandle);

  //get the tracks
  edm::Handle<reco::TrackCollection> tracks;
  iEvent.getByLabel(trackProducer_, tracks);
  const reco::TrackCollection* trackCollection = tracks.product();

  //prepare product
  auto isoMap = std::make_unique<edm::ValueMap<double>>();
  edm::ValueMap<double>::Filler filler(*isoMap);
  std::vector<double> retV(electronHandle->size(), 0);

  edm::Handle<reco::BeamSpot> beamSpotH;
  iEvent.getByLabel(beamspotProducer_, beamSpotH);
  reco::TrackBase::Point beamspot = beamSpotH->position();

  ElectronTkIsolation myTkIsolation(extRadius_,
                                    intRadiusBarrel_,
                                    intRadiusEndcap_,
                                    stripBarrel_,
                                    stripEndcap_,
                                    ptMin_,
                                    maxVtxDist_,
                                    drb_,
                                    trackCollection,
                                    beamspot);

  for (unsigned int i = 0; i < electronHandle->size(); ++i) {
    double isoValue = myTkIsolation.getPtTracks(&(electronHandle->at(i)));
    retV[i] = isoValue;
  }

  //fill and insert valuemap
  filler.insert(electronHandle, retV.begin(), retV.end());
  filler.fill();
  iEvent.put(std::move(isoMap));
}
