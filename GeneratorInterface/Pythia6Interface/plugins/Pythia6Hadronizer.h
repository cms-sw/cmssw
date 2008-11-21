// -*- C++ -*-

// class MockHadronizer is an example of a class that models the
// Hadronizer concept.

#include "HadronizerFtn.h"

#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"


namespace gen
{
  class Pythia6Hadronizer : public HadronizerFtn
  {
  public:
     explicit Pythia6Hadronizer(edm::ParameterSet const& ps);

     // bool generatePartons();
     virtual bool generatePartonsAndHadronize();
     virtual bool hadronize();
     virtual bool decay();
     virtual bool initializeForExternalPartons();
     virtual bool initializeForInternalPartons();
     virtual bool declareStableParticles();

     const char* classname() const;
  
  protected:
  
     virtual bool doEvent();
     
  private:
      
     unsigned int    fPythiaListVerbosity ;
     
     
  };
}
