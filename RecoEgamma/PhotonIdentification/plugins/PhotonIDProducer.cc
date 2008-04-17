#include "RecoEgamma/PhotonIdentification/plugins/PhotonIDProducer.h"
#include "DataFormats/EgammaCandidates/interface/PhotonID.h"
#include "DataFormats/EgammaCandidates/interface/PhotonIDAssociation.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/BasicClusterShapeAssociation.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
void PhotonIDProducer::beginJob(edm::EventSetup const& iSetup) {
  //Otherwise we're just going to do our calculations.  This will
  //set a bool for passing the cut based criteria, nothing else.
  if (doCutBased_) cutBasedAlgo_->setup(conf_);

}


PhotonIDProducer::PhotonIDProducer(const edm::ParameterSet& conf) : conf_(conf) {

  photonProducer_ = conf_.getParameter<std::string>("photonProducer");
  photonLabel_ = conf_.getParameter<std::string>("photonLabel");
  photonIDLabel_ = conf_.getParameter<std::string>("photonIDLabel");
  photonIDAssociation_ = conf_.getParameter<std::string>("photonIDAssociationLabel");

  doCutBased_ = conf_.getParameter<bool>("doCutBased");
  cutBasedAlgo_ = new CutBasedPhotonIDAlgo();
  cutBasedAlgo_->setup(conf);
  produces<reco::PhotonIDCollection>(photonIDLabel_);
  produces<reco::PhotonIDAssociationCollection>(photonIDAssociation_);

}

PhotonIDProducer::~PhotonIDProducer() {

  if (doCutBased_)
    delete cutBasedAlgo_;

}

void PhotonIDProducer::produce(edm::Event& e, const edm::EventSetup& c) {

  // Read in photons
  edm::Handle<reco::PhotonCollection> photons;
  e.getByLabel(photonProducer_,photonLabel_,photons);

  // Initialize output photon ID collection
  reco::PhotonIDCollection photonIDCollection;
  std::auto_ptr<reco::PhotonIDCollection> photonIDCollection_p(new reco::PhotonIDCollection);

  // Loop over photons and calculate photon ID using specified technique(s)
  reco::PhotonCollection::const_iterator photon;
  for (photon = (*photons).begin();
       photon != (*photons).end(); ++photon) {
    
    reco::PhotonID phoID = cutBasedAlgo_->calculate(&(*photon),e);
    photonIDCollection.push_back(phoID);
  }
  
  // Add output electron ID collection to the event
  photonIDCollection_p->assign(photonIDCollection.begin(),
			       photonIDCollection.end());
  edm::OrphanHandle<reco::PhotonIDCollection> PhotonIDHandle = e.put(photonIDCollection_p,photonIDLabel_);
  
  // Add photon ID AssociationMap to the event
  std::auto_ptr<reco::PhotonIDAssociationCollection> photonIDAssocs_p(new reco::PhotonIDAssociationCollection);
  for (unsigned int i = 0; i < photons->size(); i++){
    photonIDAssocs_p->insert(edm::Ref<reco::PhotonCollection>(photons,i),edm::Ref<reco::PhotonIDCollection>(PhotonIDHandle,i));
  }
  e.put(photonIDAssocs_p,photonIDAssociation_);
  
}
