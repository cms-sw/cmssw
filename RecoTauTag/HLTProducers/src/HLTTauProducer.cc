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

  
  
  int i=0;
  
 for(L2TauInfoAssociation::const_iterator p = tauL2Jets->begin();p!=tauL2Jets->end();++p)
	   {
	     //Retrieve The L2TauIsolationInfo Class from the AssociationMap
	     const L2TauIsolationInfo l2info = p->val;
	     //Retrieve the Jet
	     const CaloJet& jet =*(p->key);
	     
	     float eta_ = jet.eta();
	     float phi_ = jet.phi();
	     float pt_ = jet.pt();
	     HLTTau pippo(eta_,phi_,pt_);
	     pippo.setEMIsolationValue(l2info.ecalIsolEt());
	     pippo.setSeedEcalHitEt(l2info.seedEcalHitEt());
	     pippo.setEcalClusterShape(l2info.ecalClusterShape());
	     pippo.setNEcalHits(l2info.nEcalHits());		       
	     pippo.setHcalIsolEt(l2info.hcalIsolEt());
	     pippo.setSeedHcalHitEt(l2info.seedHcalHitEt());
	     pippo.setHcalClusterShape(l2info.hcalClusterShape());
	     pippo.setNHcalHits(l2info.nHcalHits());
	     //getting the pftau reference
	     PFTauRef thePFTauRef(tauL25Jets,i);
	     //setting up L2.5 quantities
	     pippo.setL25TauPt(thePFTauRef->pt());
	     if(thePFTauRef->leadPFChargedHadrCand().isNonnull()) {
	       pippo.setL25LeadTrackPtValue(thePFTauRef->leadPFChargedHadrCand()->pt());
	     }else{
	       pippo.setL25LeadTrackPtValue(0.);
	     }

	     if(thePFTauRef->leadPFCand().isNonnull()) {
	       pippo.setL25LeadPionPtValue(thePFTauRef->leadPFCand()->pt());
	     }else{
	       pippo.setL25LeadPionPtValue(0.);
	     }
	     
       //setting up L3 quantities
	     PFTauRef thePFTauL3Ref(tauL3Jets,i);
	     pippo.setNL3TrackIsolation(thePFTauL3Ref->isolationTracks().size());
	     float sumPtL3 = 0.;
	     for(unsigned int j=0;j<thePFTauL3Ref->isolationTracks().size();j++){
	       sumPtL3 = sumPtL3 + thePFTauL3Ref->isolationTracks()[j]->pt();
	     }
	     pippo.setSumPtTracksL3(sumPtL3);

	     jetCollection->push_back(pippo);
	     i++;
  }
  
  auto_ptr<reco::HLTTauCollection> selectedTaus(jetCollection);
  
  iEvent.put(selectedTaus);
  



}
