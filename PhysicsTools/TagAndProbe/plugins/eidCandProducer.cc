#include "PhysicsTools/TagAndProbe/interface/eidCandProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/Common/interface/ValueMap.h"


eidCandProducer::eidCandProducer(const edm::ParameterSet &params)
{

  electronCollection_ = 
    params.getUntrackedParameter<std::string>("electronCollection", 
					      "pixelMatchGsfElectrons");
  electronLabelLoose_= 
    params.getUntrackedParameter<std::string>("electronLabelLoose", 
					    "eidRobustLoose");

   produces<reco::GsfElectronCollection>();
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
   std::auto_ptr<reco::GsfElectronCollection> 
     outCol(new reco::GsfElectronCollection);


   // Read electrons
   edm::Handle<reco::GsfElectronCollection> electrons;
   event.getByLabel(electronCollection_, electrons);
   


  //Read electron ID results
  edm::Handle<edm::ValueMap<float> > eIDValueMap; 
  //Robust-Loose 
  event.getByLabel( electronLabelLoose_ , eIDValueMap); 
  const edm::ValueMap<float> & eIDmapL = *eIDValueMap;





   // Loop over electrons
   for(unsigned int i = 0; i < electrons->size(); i++) {
     // Get cut decision for each electron
     edm::Ref<reco::GsfElectronCollection> electronRef(electrons, i);
     if( eIDmapL[electronRef] == 1.0 ) outCol->push_back(*electronRef);
   } 

   event.put(outCol);
}



// ------ method called once each job just before starting event loop  ---



void eidCandProducer::beginJob(const edm::EventSetup &eventSetup) {
}




void eidCandProducer::endJob() {
}



//define this as a plug-in
DEFINE_FWK_MODULE( eidCandProducer );

