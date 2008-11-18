#include "RecoEgamma/PhotonIdentification/plugins/PhotonIDProducer.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/Common/interface/ValueMap.h"



PhotonIDProducer::PhotonIDProducer(const edm::ParameterSet& conf) : conf_(conf) {

  photonProducer_ = conf_.getParameter<std::string>("photonProducer");
  photonLabel_ = conf_.getParameter<std::string>("photonLabel");
 
  photonCutBasedIDLooseLabel_ = conf.getParameter<std::string>("photonCutBasedIDLooseLabel");
  photonCutBasedIDTightLabel_ = conf.getParameter<std::string>("photonCutBasedIDTightLabel");


  doCutBased_ = conf_.getParameter<bool>("doCutBased");
  cutBasedAlgo_ = new CutBasedPhotonIDAlgo();
  cutBasedAlgo_->setup(conf);
  produces<edm::ValueMap<Bool_t> > (photonCutBasedIDLooseLabel_);
  produces<edm::ValueMap<Bool_t> > (photonCutBasedIDTightLabel_);

}

PhotonIDProducer::~PhotonIDProducer() {

  //if (doCutBased_)
  delete cutBasedAlgo_;

}

void PhotonIDProducer::produce(edm::Event& e, const edm::EventSetup& c) {

   // Read in photons
  edm::Handle<reco::PhotonCollection> photons;
  e.getByLabel(photonProducer_,photonLabel_,photons);


  // Loop over photons and calculate photon ID using specified technique(s)
  reco::PhotonCollection::const_iterator photon;
  std::vector <Bool_t> Loose;
  std::vector <Bool_t> Tight;
  for (photon = (*photons).begin();
       photon != (*photons).end(); ++photon) {
    bool LooseQual;
    bool TightQual;
    if (photon->isEB())
      cutBasedAlgo_->decideEB(&(*photon),LooseQual, TightQual);
    else
      cutBasedAlgo_->decideEE(&(*photon),LooseQual, TightQual);
    Loose.push_back(LooseQual);
    Tight.push_back(TightQual);
    
  }
  

  std::auto_ptr<edm::ValueMap<Bool_t> > outloose(new edm::ValueMap<Bool_t>());
  edm::ValueMap<Bool_t>::Filler fillerloose(*outloose);
  fillerloose.insert(photons, Loose.begin(), Loose.end());
  fillerloose.fill();
  // and put it into the event
  e.put(outloose, photonCutBasedIDLooseLabel_);
  
  std::auto_ptr<edm::ValueMap<Bool_t> > outtight(new edm::ValueMap<Bool_t>());
  edm::ValueMap<Bool_t>::Filler fillertight(*outtight);
  fillertight.insert(photons, Tight.begin(), Tight.end());
  fillertight.fill();
  // and put it into the event
  e.put(outtight, photonCutBasedIDTightLabel_);
  
  
}
