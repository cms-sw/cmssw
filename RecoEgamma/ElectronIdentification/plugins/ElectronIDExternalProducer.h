#ifndef RecoEgamma_ElectronIdentification_ElectronIDExternalProducer_h
#define RecoEgamma_ElectronIdentification_ElectronIDExternalProducer_h

#include <memory>
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"

#include "DataFormats/Common/interface/ValueMap.h"

template<class algo>
class ElectronIDExternalProducer : public edm::EDProducer {
 public:
   explicit ElectronIDExternalProducer(const edm::ParameterSet& iConfig) :
            src_(iConfig.getParameter<edm::InputTag>("src")),
            select_(iConfig)
   {
        produces<edm::ValueMap<float> >();
   }

   virtual ~ElectronIDExternalProducer() {}

   void produce(edm::Event & iEvent, const edm::EventSetup & iSetup) ;
	
 private:	
   edm::InputTag src_ ;
   algo select_ ;
  
};

template<typename algo>
void ElectronIDExternalProducer<algo>::produce(edm::Event & iEvent, const edm::EventSetup & iSetup) {
     // read input collection
     edm::Handle<reco::GsfElectronCollection> electrons;
     iEvent.getByLabel(src_, electrons);

     // initialize common selector
     select_.newEvent(iEvent, iSetup);

     // prepare room for output
     std::vector<float> values; values.reserve(electrons->size());
     for ( reco::GsfElectronCollection::const_iterator eleIt = electrons->begin () ;
             eleIt != electrons->end () ;
             ++eleIt ) {
         values.push_back( float( select_((*eleIt),iEvent,iSetup) ) );
     }

     // fill in the ValueMap
     std::auto_ptr<edm::ValueMap<float> > out(new edm::ValueMap<float>());
     edm::ValueMap<float>::Filler filler(*out);
     filler.insert(electrons, values.begin(), values.end());
     filler.fill();
     // and put it into the event
     iEvent.put(out);

}
#endif
