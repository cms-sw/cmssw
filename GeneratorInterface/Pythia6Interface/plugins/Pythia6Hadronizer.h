// -*- C++ -*-

// class Pythia6Hadronizer is an example of a class that models the
// Hadronizer concept.


#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"

#include "SimDataFormats/GeneratorProducts/interface/GenInfoProduct.h"

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
  class Pythia6Hadronizer 
  {
  public:
     Pythia6Hadronizer(edm::ParameterSet const& ps);
     ~Pythia6Hadronizer();

     // bool generatePartons();
     bool generatePartonsAndHadronize();
     bool hadronize();
     bool decay();
     bool initializeForExternalPartons();
     bool initializeForInternalPartons();
     bool declareStableParticles();
     
     void statistics();

     const char* classname() const;
     
     void setLHERunInfo( lhef::LHERunInfo* lheri ) ;
     void setLHEEventProd( LHEEventProduct* lheep ); 
     HepMC::GenEvent* getGenEvent() { return fGenEvent; }
     const edm::GenInfoProduct& getGenInfoProduct() const { return fGenInfoProduct; }
  
  protected:
  
     bool doEvent();
     
  private:
           
     std::vector<std::string> paramGeneral;
     std::vector<std::string> paramCSA;
     std::vector<std::string> paramSLHA;
     
     // the following 7 params are common for all generators(interfaces)
     // probably better to wrap them up in a class and reuse ?
     //
     double fCOMEnergy ;  // this one is irrelevant for setting py6 as hadronizer
                          // or if anything, it should be picked up from LHERunInfoProduct !   
     HepMC::GenEvent*     fGenEvent; 
     edm::GenInfoProduct  fGenInfoProduct;
     int                  fEventCounter;
     
     lhef::LHERunInfo*    fRunInfo;
     
     CLHEP::HepRandomEngine& fRandomEngine;
     CLHEP::RandFlat*        fRandomGenerator; 

     bool            fHepMCVerbosity;
     unsigned int    fMaxEventsToPrint ;
           
     // this is the only one specific to Pythia6
     //
     unsigned int    fPythiaListVerbosity ;
     
     void setGeneralParams();
     void setCSAParams();
     void setSLHAParams();
          
  };
}
