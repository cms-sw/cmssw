//*****************************************************************************
// File:      EgammaElectronTkNumIsolationProducer.cc
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
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/ElectronTkIsolation.h"

class EgammaElectronTkNumIsolationProducer : public edm::global::EDProducer<> {
public:
  explicit EgammaElectronTkNumIsolationProducer(const edm::ParameterSet&);
  ~EgammaElectronTkNumIsolationProducer() override;

  void produce(edm::StreamID sid, edm::Event&, const edm::EventSetup&) const override;

private:
  const edm::InputTag electronProducer_;
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
DEFINE_FWK_MODULE(EgammaElectronTkNumIsolationProducer);

EgammaElectronTkNumIsolationProducer::EgammaElectronTkNumIsolationProducer(const edm::ParameterSet& config)
    :  // use configuration file to setup input/output collection names
      electronProducer_(config.getParameter<edm::InputTag>("electronProducer")),

      trackProducer_(config.getParameter<edm::InputTag>("trackProducer")),
      beamspotProducer_(config.getParameter<edm::InputTag>("BeamspotProducer")),

      ptMin_(config.getParameter<double>("ptMin")),
      intRadiusBarrel_(config.getParameter<double>("intRadiusBarrel")),
      intRadiusEndcap_(config.getParameter<double>("intRadiusEndcap")),
      stripBarrel_(config.getParameter<double>("stripBarrel")),
      stripEndcap_(config.getParameter<double>("stripEndcap")),
      extRadius_(config.getParameter<double>("extRadius")),
      maxVtxDist_(config.getParameter<double>("maxVtxDist")),
      drb_(config.getParameter<double>("maxVtxDistXY")) {
  //register your products
  produces<edm::ValueMap<int>>();
}

EgammaElectronTkNumIsolationProducer::~EgammaElectronTkNumIsolationProducer() {}

void EgammaElectronTkNumIsolationProducer::produce(edm::StreamID sid,
                                                   edm::Event& iEvent,
                                                   const edm::EventSetup& iSetup) const {
  // Get the  filtered objects
  edm::Handle<reco::GsfElectronCollection> electronHandle;
  iEvent.getByLabel(electronProducer_, electronHandle);

  //get the tracks
  edm::Handle<reco::TrackCollection> tracks;
  iEvent.getByLabel(trackProducer_, tracks);
  const reco::TrackCollection* trackCollection = tracks.product();

  //prepare product
  auto isoMap = std::make_unique<edm::ValueMap<int>>();
  edm::ValueMap<int>::Filler filler(*isoMap);
  std::vector<int> retV(electronHandle->size(), 0);

  //get beamspot
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
    int isoValue = myTkIsolation.getNumberTracks(&(electronHandle->at(i)));
    retV[i] = isoValue;
  }

  //fill and insert valuemap
  filler.insert(electronHandle, retV.begin(), retV.end());
  filler.fill();
  iEvent.put(std::move(isoMap));
}
