#include "RecoTauTag/HLTProducers/interface/HLTTauProducer.h"
#include "DataFormats/TrackReco/interface/Track.h"
//
// class decleration
//


HLTTauProducer::HLTTauProducer(const edm::ParameterSet& iConfig)
{
  emIsolatedJetsL2_ = iConfig.getParameter<edm::InputTag>("L2EcalIsoJets");
  trackIsolatedJetsL25_ = iConfig.getParameter<edm::InputTag>("L25TrackIsoJets");
  trackIsolatedJetsL3_ = iConfig.getParameter<edm::InputTag>("L3TrackIsoJets");
 
  produces<reco::HLTTauCollection>();
}

HLTTauProducer::~HLTTauProducer(){ }

void HLTTauProducer::produce(edm::Event& iEvent, const edm::EventSetup& iES)
{

  using namespace reco;
  using namespace edm;
  using namespace std;
  

  HLTTauCollection * jetCollection = new HLTTauCollection;
 
  edm::Handle<L2TauInfoAssociation> tauL2Jets;
  iEvent.getByLabel(emIsolatedJetsL2_ , tauL2Jets );

  edm::Handle<PFTauCollection> tauL25Jets;
  iEvent.getByLabel(trackIsolatedJetsL25_, tauL25Jets );
  
  edm::Handle<PFTauCollection> tauL3Jets;
  iEvent.getByLabel(trackIsolatedJetsL3_, tauL3Jets );

  PFTauCollection tauL25 = *(tauL25Jets.product());
  PFTauCollection tauL3 = *(tauL3Jets.product());
  
  int i=0;
  
 for(L2TauInfoAssociation::const_iterator p = tauL2Jets->begin();p!=tauL2Jets->end();++p)
	   {
	     //Retrieve The L2TauIsolationInfo Class from the AssociationMap
	     const L2TauIsolationInfo l2info = p->val;
	     //Retrieve the Jet
	     const CaloJet& jet =*(p->key);
	     

	     double emIsol  = l2info.ecalIsolEt();

    float eta_ = jet.eta();
    float phi_ = jet.phi();
    float pt_ = jet.pt();

   
    HLTTau pippo(eta_,phi_,pt_);
    /*
    pippo.setNL3TrackIsolation(nTracksL3);
    pippo.setSumPtTracksL3(sumPtTracksL3);
    pippo.setSeedEcalHitEt(l2info.seedEcalHitEt());
    pippo.setEcalClusterShape(l2info.ecalClusterShape());
    pippo.setNEcalHits(l2info.nEcalHits());		       
    pippo.setHcalIsolEt(l2info.hcalIsolEt());
    pippo.setSeedHcalHitEt(l2info.seedHcalHitEt());
    pippo.setHcalClusterShape(l2info.hcalClusterShape());
    pippo.setNHcalHits(l2info.nHcalHits());
    */
    jetCollection->push_back(pippo);
      i++;
  }
  
  auto_ptr<reco::HLTTauCollection> selectedTaus(jetCollection);
  
  iEvent.put(selectedTaus);
  



}
