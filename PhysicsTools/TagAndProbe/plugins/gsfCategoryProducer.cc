#include "PhysicsTools/TagAndProbe/interface/gsfCategoryProducer.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include <string>


gsfCategoryProducer::gsfCategoryProducer(const edm::ParameterSet &params)
{

  _inputProducer = params.getParameter<std::string>("InputProducer");
  _gsfCategory   = params.getParameter<std::string>("GsfCategory");
  _isInBarrel    = params.getParameter<bool>("isInBarrel");
  _isInEndCap    = params.getParameter<bool>("isInEndCap");
  _isInCrack     = params.getParameter<bool>("isInCrack");

   componentsBit = 0;
   if(_gsfCategory.find("golden") != std::string::npos ) 
     componentsBit += Golden;
   if(_gsfCategory.find("bigbrem") != std::string::npos ) 
     componentsBit += Bigbrem;
   if(_gsfCategory.find("narrow") != std::string::npos ) 
     componentsBit += Narrow;
   if(_gsfCategory.find("shower") != std::string::npos ) 
     componentsBit += Showering;

   produces<reco::GsfElectronCollection>();
}






gsfCategoryProducer::~gsfCategoryProducer()
{

}


//
// member functions
//


// ------------ method called to produce the data  ------------

void gsfCategoryProducer::produce(edm::Event &event, const edm::EventSetup &eventSetup)
{
   // Create the output collection
   std::auto_ptr<reco::GsfElectronCollection> outCol(new reco::GsfElectronCollection);


   // Run the algoritm and store in the event
   edm::Handle<reco::GsfElectronCollection> eleCandidatesHandle;
   try
   {
      event.getByLabel(_inputProducer, eleCandidatesHandle);
   }
   catch(cms::Exception &ex)
   {
      edm::LogError("GsfElectron ") << 
	"Error! Can't get collection " << _inputProducer;
      throw ex;
   }




   // Loop over electrons
   for(unsigned int i = 0; i < eleCandidatesHandle->size(); ++i) {
     // Get cut decision for each electron
     edm::Ref<reco::GsfElectronCollection> electronRef(eleCandidatesHandle, i);

     bool isGsfGolden    = false;
     bool isGsfBigBrem   = false;
     bool isGsfNarrow    = false;
     bool isGsfShowering = false;
     bool isGsfInBarrel  = false;
     bool isGsfInEndCap  = false;
     bool isGsfInCrack   = false;


     int gsfclass = electronRef->classification();
     if( gsfclass == 0  || gsfclass == 100) isGsfGolden  = true;
     if( gsfclass == 10 || gsfclass == 110) isGsfBigBrem = true;
     if( gsfclass == 20 || gsfclass == 120) isGsfNarrow  = true;
     if( (gsfclass >= 30 && gsfclass <= 34) || 
	 (gsfclass >= 130 && gsfclass <= 134) ) isGsfShowering = true;
     if( gsfclass >= 0 && gsfclass <= 34 ) isGsfInBarrel  = true;
     if( gsfclass >= 100 && gsfclass <= 134 ) isGsfInEndCap  = true;
     if( gsfclass == 40 ) isGsfInCrack   = true;


     bool cutDecision = false;
     if(_isInCrack && isGsfInCrack) cutDecision = true;
     if((_isInBarrel && isGsfInBarrel) || (_isInEndCap && isGsfInEndCap)) {
       if( (componentsBit & Golden) && isGsfGolden ) cutDecision = true; 
       if( (componentsBit & Bigbrem) && isGsfBigBrem ) cutDecision = true; 
       if( (componentsBit & Narrow) && isGsfNarrow ) cutDecision = true; 
       if( (componentsBit & Showering) && isGsfShowering ) cutDecision = true; 
     }

     if(cutDecision) outCol->push_back(*electronRef);
   } 

   event.put(outCol);
}



// ------------ method called once each job just before starting event loop  ---



void gsfCategoryProducer::beginJob() {
}




void gsfCategoryProducer::endJob() {
}



//define this as a plug-in
DEFINE_FWK_MODULE( gsfCategoryProducer );
