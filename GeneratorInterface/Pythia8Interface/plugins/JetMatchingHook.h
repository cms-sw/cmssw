#ifndef gen_JetMatchinhHook_h
#define gen_JetMatchingHook_h

#include <Pythia.h>

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/GeneratorProducts/interface/LHECommonBlocks.h"

#include "GeneratorInterface/LHEInterface/interface/LHERunInfo.h"
#include "GeneratorInterface/LHEInterface/interface/LHEEvent.h"

#include "GeneratorInterface/PartonShowerVeto/interface/JetMatching.h"

class JetMatchingHook : public Pythia8::UserHooks
{

public:

  JetMatchingHook( const edm::ParameterSet&, Pythia8::Info* );
  virtual ~JetMatchingHook();
  
  virtual bool canVetoPartonLevel() { return true; }  
  virtual bool doVetoPartonLevel( const Pythia8::Event& event );
    
  void setEventNumber( int ievt ) { fEventNumber = ievt; return ; }
  
  void init( lhef::LHERunInfo* runInfo );
  void resetMatchingStatus() { fJetMatching->resetMatchingStatus(); return; }
  void beforeHadronization( lhef::LHEEvent* lhee );
  
protected:

  void setLHERunInfo( lhef::LHERunInfo* lheri ) { 
     fRunBlock=lheri;
     if ( fRunBlock == 0 ) return;
     const lhef::HEPRUP* heprup = fRunBlock->getHEPRUP();
     lhef::CommonBlocks::fillHEPRUP(heprup); 
     return;
  }
  void setLHEEvent( lhef::LHEEvent* lhee ) { 
     fEventBlock=lhee; 
     if ( fEventBlock == 0 ) return;
     const lhef::HEPEUP* hepeup = fEventBlock->getHEPEUP();
     lhef::CommonBlocks::fillHEPEUP(hepeup);
     return;
  }
    
private:

     lhef::LHERunInfo*       fRunBlock;
     lhef::LHEEvent*         fEventBlock;
     int                     fEventNumber;
     Pythia8::Info*          fInfoPtr;
     gen::JetMatching*       fJetMatching;
          
     void setHEPEVT( const Pythia8::Event& );
     // void setHEPEVT();
     int getAncestor( int, const Pythia8::Event& );
 
};

#endif
