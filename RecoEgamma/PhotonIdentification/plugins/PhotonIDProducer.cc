#include "RecoEgamma/PhotonIdentification/plugins/PhotonIDProducer.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/Common/interface/ValueMap.h"



PhotonIDProducer::PhotonIDProducer(const edm::ParameterSet& conf) : conf_(conf) {
  photonToken_ = consumes<reco::PhotonCollection>(edm::InputTag(conf_.getParameter<std::string>("photonProducer"),
								conf_.getParameter<std::string>("photonLabel")));
 
  photonCutBasedIDLooseLabel_ = conf.getParameter<std::string>("photonCutBasedIDLooseLabel");
  photonCutBasedIDTightLabel_ = conf.getParameter<std::string>("photonCutBasedIDTightLabel");
  photonCutBasedIDLooseEMLabel_ = conf.getParameter<std::string>("photonCutBasedIDLooseEMLabel");

  doCutBased_ = conf_.getParameter<bool>("doCutBased");
  cutBasedAlgo_ = new CutBasedPhotonIDAlgo();
  cutBasedAlgo_->setup(conf);
  produces<edm::ValueMap<Bool_t> > (photonCutBasedIDLooseLabel_);
  produces<edm::ValueMap<Bool_t> > (photonCutBasedIDTightLabel_);
  produces<edm::ValueMap<Bool_t> > (photonCutBasedIDLooseEMLabel_);

}

PhotonIDProducer::~PhotonIDProducer() {

  //if (doCutBased_)
  delete cutBasedAlgo_;

}

void PhotonIDProducer::produce(edm::Event& e, const edm::EventSetup& c) {

   // Read in photons
  edm::Handle<reco::PhotonCollection> photons;
  e.getByToken(photonToken_,photons);


  // Loop over photons and calculate photon ID using specified technique(s)
  reco::PhotonCollection::const_iterator photon;
  std::vector <Bool_t> Loose;
  std::vector <Bool_t> Tight;
  std::vector <Bool_t> LooseEM;
  for (photon = (*photons).begin();
       photon != (*photons).end(); ++photon) {
    bool LooseQual;
    bool TightQual;
    bool LooseEMQual;
    if (photon->isEB())
      cutBasedAlgo_->decideEB(&(*photon),LooseEMQual,LooseQual, TightQual);
    else
      cutBasedAlgo_->decideEE(&(*photon),LooseEMQual,LooseQual, TightQual);
    LooseEM.push_back(LooseEMQual);
    Loose.push_back(LooseQual);
    Tight.push_back(TightQual);
    
  }
  

  auto outlooseEM = std::make_unique<edm::ValueMap<Bool_t>>();
  edm::ValueMap<Bool_t>::Filler fillerlooseEM(*outlooseEM);
  fillerlooseEM.insert(photons, LooseEM.begin(), LooseEM.end());
  fillerlooseEM.fill();
  // and put it into the event
  e.put(std::move(outlooseEM), photonCutBasedIDLooseEMLabel_);

  auto outloose = std::make_unique<edm::ValueMap<Bool_t>>();
  edm::ValueMap<Bool_t>::Filler fillerloose(*outloose);
  fillerloose.insert(photons, Loose.begin(), Loose.end());
  fillerloose.fill();
  // and put it into the event
  e.put(std::move(outloose), photonCutBasedIDLooseLabel_);
  
  auto outtight = std::make_unique<edm::ValueMap<Bool_t>>();
  edm::ValueMap<Bool_t>::Filler fillertight(*outtight);
  fillertight.insert(photons, Tight.begin(), Tight.end());
  fillertight.fill();
  // and put it into the event
  e.put(std::move(outtight), photonCutBasedIDTightLabel_);
  
  
}
