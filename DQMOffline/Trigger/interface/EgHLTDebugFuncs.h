#ifndef DQMOFFLINE_TRIGGER_DEBUGFUNCS
#define DQMOFFLINE_TRIGGER_DEBUGFUNCS

//collection of usefull functions adding to debug namespace
//
//author: Sam Harper (July 2008)
//
//
//aim: to store common functions which are usefull for debuging my code,
//     in theory any function here doesnt contribute to the program
//     therefore the program should still work when all functions here are     
//     removed from it

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Provenance/interface/Provenance.h"

#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/Event.h"
namespace egHLT {
  namespace debug {
    
    //function prints to LogInfo the names of all products of type PROD in event
    //it returns the number of products it finds
    template<typename PROD> int listAllProducts(const edm::Event& iEvent,const char* moduleName);
    
  }
  
  
  template<typename PROD> int debug::listAllProducts(const edm::Event& iEvent,const char* moduleName)
  {
    std::vector<edm::Handle<PROD> > products;
    iEvent.getManyByType(products);
    
    for(size_t i=0;i<products.size();i++){
      //  edm::LogInfo(moduleName) <<"for product "<<i+1<<"/"<<products.size()<<" "<<products[i].provenance()->moduleLabel()<<" "<<products[i].provenance()->moduleName()<<std::endl;
      std::cout <<"for product "<<i+1<<"/"<<products.size()<<" "<<products[i].provenance()->moduleLabel()<<" "<<products[i].provenance()->moduleName()<<std::endl; 
     
    }
    return products.size();
  }
}




#endif
