#include "RecoTauTag/HLTProducers/interface/EMIsolatedTauJetsSelector.h"
#include "DataFormats/JetReco/interface/JetTracksAssociation.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "Math/GenVector/VectorUtil.h"
//
// class decleration
//


EMIsolatedTauJetsSelector::EMIsolatedTauJetsSelector(const edm::ParameterSet& iConfig)
{
  tauSrc      = iConfig.getParameter<std::vector< edm::InputTag > >("TauSrc");
  typedef std::vector<edm::InputTag> vtag;
  for(vtag::const_iterator it = tauSrc.begin(); it != tauSrc.end(); ++it) {
    edm::EDGetTokenT<reco::EMIsolatedTauTagInfoCollection> aToken = consumes<reco::EMIsolatedTauTagInfoCollection>(*it);
    tauSrc_token.push_back(aToken);
  }

  produces<reco::CaloJetCollection>("Isolated");
  produces<reco::CaloJetCollection>("NotIsolated");
}

EMIsolatedTauJetsSelector::~EMIsolatedTauJetsSelector(){ }

void EMIsolatedTauJetsSelector::produce(edm::Event& iEvent, const edm::EventSetup& iES)
{

  using namespace reco;
  using namespace edm;
  using namespace std;
  
  std::auto_ptr<reco::CaloJetCollection> isolatedTaus(new CaloJetCollection); 
  std::auto_ptr<reco::CaloJetCollection> notIsolatedTaus(new CaloJetCollection);
 
  typedef std::vector<edm::EDGetTokenT<reco::EMIsolatedTauTagInfoCollection> > vtag_token;
  for( vtag_token::const_iterator s = tauSrc_token.begin(); s != tauSrc_token.end(); ++ s ) {
    edm::Handle<EMIsolatedTauTagInfoCollection> tauJets;
    iEvent.getByToken( * s, tauJets );
    EMIsolatedTauTagInfoCollection::const_iterator i = tauJets->begin();
    for(;i !=tauJets->end(); i++ ) {
      double discriminator = (*i).discriminator();
      if(discriminator > 0) {
	const CaloJet* pippo = dynamic_cast<const CaloJet*>(i->jet().get());
	isolatedTaus->push_back(*pippo );
      }else{
	const CaloJet* notPippo =dynamic_cast<const CaloJet*>(i->jet().get());

	notIsolatedTaus->push_back(*notPippo );
      }
    }
  }
  
  

  iEvent.put(isolatedTaus, "Isolated");
  iEvent.put(notIsolatedTaus,"NotIsolated");
  
}
