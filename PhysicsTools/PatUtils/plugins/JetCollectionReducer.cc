#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"

template<typename T>
class JetCollectionReducerT : public edm::stream::EDProducer<> {

public:
  explicit JetCollectionReducerT(const edm::ParameterSet & iConfig);
  virtual ~JetCollectionReducerT() {}

  virtual void produce(edm::Event & iEvent, const edm::EventSetup & iSetup) override;

private:

  edm::EDGetTokenT<std::vector<T>> jetColToken_;            
  bool writeEmptyCollection_;
  std::vector<edm::EDGetTokenT<edm::View<reco::Candidate> > > collections_;
  
};


template<typename T>
JetCollectionReducerT<T>::JetCollectionReducerT(const edm::ParameterSet& iConfig) :
  jetColToken_(consumes<std::vector<T> >( iConfig.getParameter<edm::InputTag>("jetCollection") )),
  writeEmptyCollection_(iConfig.getParameter<bool>("writeEmptyCollection"))
{
  
  std::vector<edm::InputTag> filtersDecTags = iConfig.getParameter<std::vector<edm::InputTag> >("triggeringCollections");
  for(std::vector<edm::InputTag>::const_iterator inputTag=filtersDecTags.begin();
      inputTag!=filtersDecTags.end();++inputTag) {
    collections_.push_back(consumes<edm::View<reco::Candidate> >(*inputTag));
  }
  
  produces<std::vector<T> >();

}

// ------------ method called to produce the data  ------------
template<typename T>
void
JetCollectionReducerT<T>::produce(edm::Event& iEvent, const edm::EventSetup&)
{  

  std::unique_ptr<std::vector<T> > outJets(new std::vector<T>());
  
  bool filterDecision=false;
  edm::Handle<edm::View<reco::Candidate> > tmpCol;
  for(std::vector<edm::EDGetTokenT<edm::View<reco::Candidate> > >::const_iterator filter=collections_.begin();
      filter!=collections_.end();filter++) {
    iEvent.getByToken(*filter,tmpCol);
    if(tmpCol->size()!=0) {
      filterDecision=true;
      break;
    }
  }

  if(!filterDecision) {
    if (writeEmptyCollection_) iEvent.put(std::move(outJets));
    return;
  }
  
  edm::Handle< std::vector<T> > jetColHandle;
  iEvent.getByToken( jetColToken_, jetColHandle );
  
  //MM: probably a better way to do it...
  for(size_t ij=0;ij<jetColHandle->size();ij++) {
    outJets->push_back( (*jetColHandle)[ij] );
  }

  iEvent.put(std::move(outJets));

}

//define this as a plug-in
typedef JetCollectionReducerT<pat::Jet> PATJetCollectionReducer;
typedef JetCollectionReducerT<reco::PFJet> PFJetCollectionReducer;
DEFINE_FWK_MODULE(PATJetCollectionReducer);
DEFINE_FWK_MODULE(PFJetCollectionReducer);
