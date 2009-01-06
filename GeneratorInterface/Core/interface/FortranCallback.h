#ifndef gen_FortranCallback_h
#define gen_FortranCallback_h

/*
namespace lhef
{
class LHEEvent;
}
*/

//class LHERunInfoProduct;
//class LHEEventProduct;

#include "SimDataFormats/GeneratorProducts/interface/LHECommonBlocks.h"

#include "SimDataFormats/GeneratorProducts/interface/LHERunInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/LHEEventProduct.h"


namespace HepMC
{
class GenEvent;
}

namespace gen {

  class FortranCallback 
  {

    public:
    
      static FortranCallback* getInstance() ;
    
      //void setLHEEvent(lhef::LHEEvent* lhev) { fPartonLevel = lhev; return ; }
      void setLHERunProd( LHERunInfoProduct* lherp ) { fRunBlock=lherp; return; }
      void setLHEEventProd( LHEEventProduct* lheep ) { fEventBlock=lheep; return ; }
      
      void fillHeader();
      void fillEvent();  

   private:
   
     // ctor
     
     FortranCallback();

     // data member(s)
     
     //lhef::LHEEvent*  fPartonLevel;
     LHERunInfoProduct* fRunBlock;
     LHEEventProduct*   fEventBlock;
     
     static FortranCallback* fInstance;

	
  };

// --** Implementation **---

FortranCallback* FortranCallback::fInstance = 0;

FortranCallback::FortranCallback()
//   : fPartonLevel(0)
   : fRunBlock(0), fEventBlock(0)
{   
}

FortranCallback* FortranCallback::getInstance()
{
   if ( fInstance == 0 ) fInstance = new FortranCallback;
   return fInstance;
}

void FortranCallback::fillHeader()
{

   // if ( fPartonLevel == 0 ) return;  
   // const lhef::HEPRUP* heprup = fPartonLevel->getHEPRUP();
   
   if ( fRunBlock == 0 ) return;
      
   const lhef::HEPRUP* heprup = &(fRunBlock->heprup());

   lhef::CommonBlocks::fillHEPRUP(heprup);   
   
   return;

}

void FortranCallback::fillEvent()
{

   //if ( fPartonLevel == 0 ) return;    
   //const lhef::HEPEUP* hepeup = fPartonLevel->getHEPEUP();
   
   if ( fEventBlock == 0 ) return;
   
   const lhef::HEPEUP* hepeup = &(fEventBlock->hepeup()); 

        //if (iterations++) {
                if (hepeup_.nup = 0)
                return;
        //}

   lhef::CommonBlocks::fillHEPEUP(hepeup);

   return;
    
}


} // end namespace

#endif



