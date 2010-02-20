#include "PhysicsTools/TagAndProbe/interface/eidCandProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/Common/interface/ValueMap.h"


eidCandProducer::eidCandProducer(const edm::ParameterSet &params)
{

  const edm::InputTag allelectrons("gsfElectrons");
  electronCollection_ = 
    params.getUntrackedParameter<edm::InputTag>("electronCollection", 
						allelectrons);
  electronLabel_= 
    params.getParameter<edm::InputTag>("electronLabelLoose");
  
  produces<std::vector<reco::GsfElectron> >();
  produces< edm::PtrVector<reco::GsfElectron> >();
  produces< edm::RefToBaseVector<reco::GsfElectron> >();
}




eidCandProducer::~eidCandProducer()
{

}


//
// member functions
//


// ------------ method called to produce the data  ------------

void eidCandProducer::produce(edm::Event &event, 
			      const edm::EventSetup &eventSetup)
{
   // Create the output collection
  std::auto_ptr<std::vector<reco::GsfElectron> >  outCol(new std::vector<reco::GsfElectron> );
  std::auto_ptr< edm::RefToBaseVector<reco::GsfElectron> > 
    outColRef( new edm::RefToBaseVector<reco::GsfElectron> );
  std::auto_ptr< edm::PtrVector<reco::GsfElectron> > 
    outColPtr( new edm::PtrVector<reco::GsfElectron> );


   // Read electrons
   edm::Handle<edm::View<reco::GsfElectron> > electrons;
   event.getByLabel(electronCollection_, electrons);
   


  //Read electron ID results
  edm::Handle<edm::ValueMap<float> > eIDValueMap; 
  //Robust-Loose 
  event.getByLabel( electronLabel_ , eIDValueMap); 
  const edm::ValueMap<float> & eIDmapL = *eIDValueMap;





   // Loop over electrons
  const edm::PtrVector<reco::GsfElectron>& ptrVect = electrons->ptrVector();
  const edm::RefToBaseVector<reco::GsfElectron>& refs = electrons->refVector();
  unsigned int counter=0;

   for(edm::View<reco::GsfElectron>::const_iterator  elec = electrons->begin(); 
       elec != electrons->end();  ++elec, ++counter) {
     // Get cut decision for each electron
     if( eIDmapL[ refs[counter] ] == 1.0 ) {
       outCol->push_back(*elec);
       outColRef->push_back( refs[counter] );
       outColPtr->push_back(  ptrVect[counter] );
     }
   }

   event.put(outCol);
   event.put(outColRef);
   event.put(outColPtr);
}




// ------ method called once each job just before starting event loop  ---



void eidCandProducer::beginJob() {}




void eidCandProducer::endJob() {}



//define this as a plug-in
DEFINE_FWK_MODULE( eidCandProducer );

