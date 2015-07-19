/* -*- C++ -*-
 * Package:    HLTrigger/JetMET
 * Class:      PUFilter
 * \class PUFilter PUFilter.cc HLTrigger/JetMET/plugins/PUFilter.cc
 *
 *  Description: [one line class summary]
 *
 *   Implementation:
 *        [Notes on implementation]
 *         Original Author:  Silvio DONATO
 *                  Created:  Fri, 17 Jul 2015 12:22:46 GMT
 *
 *                  */


#include <memory>

#include "RecoJets/JetProducers/plugins/PUFilter.h"

PUFilter::PUFilter(const edm::ParameterSet& iConfig):
jetsToken_( consumes<edm::View<reco::PFJet> > (iConfig.getParameter<edm::InputTag>("Jets") ) ),
jetPuIdToken_( consumes<edm::ValueMap<int> > (iConfig.getParameter<edm::InputTag>("JetPUID") ) )
{
   produces<std::vector<reco::PFJet> > ();
}


PUFilter::~PUFilter()
{
 
   

}


void
PUFilter::produce(edm::StreamID sid, edm::Event& iEvent, const edm::EventSetup & iSetup) const
{
   using namespace edm;

  Handle<edm::View<reco::PFJet> > jetsH;
  Handle< edm::ValueMap<int> > id_decisions;
  
  iEvent.getByToken( jetsToken_, jetsH );
  iEvent.getByToken( jetPuIdToken_, id_decisions );
  
  std::auto_ptr<std::vector<reco::PFJet> > goodjets(new std::vector<reco::PFJet> );
  for( size_t i = 0; i < jetsH->size(); ++i ) {
    auto jet = jetsH->refAt(i);
/*    if (jet->pt()>40){
        std::cout << "jet pt = " << jet->pt() ;
        std::cout << " ID = " << (*id_decisions)[jet] << std::endl;
    }
  */  if((*id_decisions)[jet]) goodjets->push_back(*jet);
  }
  iEvent.put(goodjets);
}

void
PUFilter::beginJob()
{
}

void
PUFilter::endJob() {
}
void
PUFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("Jets", edm::InputTag("hltAK4PFJetsCorrected"));
  desc.add<edm::InputTag>("JetPUID", edm::InputTag("MVAJetPuIdProducer","CATEv0Id"));
  descriptions.add("PUFilter",desc);
  desc.setUnknown();
}
DEFINE_FWK_MODULE(PUFilter);

