#ifndef gen_Pythia6Hadronizer_h
#define gen_Pythia6Hadronizer_h

// -*- C++ -*-

// class Pythia6Hadronizer is an example of a class that models the
// Hadronizer concept.

#include <memory>

#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"

#include "SimDataFormats/GeneratorProducts/interface/GenRunInfoProduct.h"

namespace lhef
{
class LHERunInfo;
}

class LHEEventProduct;

namespace HepMC
{
class GenEvent;
}

namespace CLHEP {
class HepRandomEngine;
class RandFlat;
}


namespace gen
{

class Pythia6Service;
class JetMatching;

  class Pythia6Hadronizer 
  {
  
  public:
     Pythia6Hadronizer(edm::ParameterSet const& ps);
     ~Pythia6Hadronizer();

     // bool generatePartons();
     bool generatePartonsAndHadronize();
     bool hadronize();
     bool decay();
     bool residualDecay();
     bool initializeForExternalPartons();
     bool initializeForInternalPartons();
     bool declareStableParticles( const std::vector<int> );
     
     static JetMatching* getJetMatching() { return fJetMatching; }
          
     void finalizeEvent();

     void statistics();

     const char* classname() const;
     
     void setLHERunInfo( lhef::LHERunInfo* lheri ) ;
     void setLHEEventProd( LHEEventProduct* lheep ); 
     
     void resetEvent( HepMC::GenEvent* );
     
     HepMC::GenEvent* getGenEvent() { return fGenEvent.release(); }
     const GenRunInfoProduct& getGenRunInfo() const { return fGenRunInfo; }
             
  private:
     
     Pythia6Service* fPy6Service;
           
     // the following 7 params are common for all generators(interfaces)
     // probably better to wrap them up in a class and reuse ?
     //
     double fCOMEnergy ;  // this one is irrelevant for setting py6 as hadronizer
                          // or if anything, it should be picked up from LHERunInfoProduct !   
     std::auto_ptr<HepMC::GenEvent> fGenEvent; 
     GenRunInfoProduct              fGenRunInfo;
     int                            fEventCounter;
     
     lhef::LHERunInfo*              fRunInfo;
     LHEEventProduct*               fEventInfo;

     static JetMatching*            fJetMatching; 

     bool            fHepMCVerbosity;
     unsigned int    fMaxEventsToPrint ;
           
     // this is the only one specific to Pythia6
     //
     unsigned int    fPythiaListVerbosity ;
               
  };
}

#endif
