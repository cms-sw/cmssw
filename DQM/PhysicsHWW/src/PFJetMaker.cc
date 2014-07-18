#include "DQM/PhysicsHWW/interface/PFJetMaker.h"
#include "DataFormats/BTauReco/interface/JetTag.h"

typedef math::XYZTLorentzVectorF LorentzVector;

PFJetMaker::PFJetMaker(const edm::ParameterSet& iConfig, edm::ConsumesCollector iCollector){

  PFJetCollection_  = iCollector.consumes<edm::View<reco::Jet> >(iConfig.getParameter<edm::InputTag>("pfJetsInputTag"));
  BJetTags_         = iCollector.consumes<reco::JetFloatAssociation::Container>(iConfig.getParameter<edm::InputTag>("trackCountingHighEffBJetTags"));

}

void PFJetMaker::SetVars(HWW& hww, const edm::Event& iEvent, const edm::EventSetup& iSetup){

  hww.Load_pfjets_p4();
  hww.Load_pfjets_area();
  hww.Load_pfjets_trackCountingHighEffBJetTag();

  bool validToken;

  edm::Handle<edm::View<reco::Jet> >    pfjetsHandle;
  validToken = iEvent.getByToken(PFJetCollection_,  pfjetsHandle	);
  if(!validToken) return;

  edm::Handle<reco::JetFloatAssociation::Container> trackCountingHighEffBJetTags;
  validToken = iEvent.getByToken(BJetTags_, trackCountingHighEffBJetTags);
  if(!validToken) return;
    
  for(edm::View<reco::Jet>::const_iterator jet_it = pfjetsHandle->begin(); jet_it != pfjetsHandle->end(); jet_it++) {
       
    if(jet_it->pt() <= 0.0 ) continue;

    hww.pfjets_p4()     .push_back( LorentzVector( jet_it->p4() )      );
    hww.pfjets_area()   .push_back(jet_it->jetArea()                   );

    unsigned int idx = jet_it-pfjetsHandle->begin();
    edm::RefToBase<reco::Jet> jetRef = pfjetsHandle->refAt(idx);

    hww.pfjets_trackCountingHighEffBJetTag().push_back( (*trackCountingHighEffBJetTags)[jetRef]	); 

  }

}
