#ifndef gen_HadronizerFtn_h
#define gen_HadronizerFtn_h

// -*- C++ -*-

// class HadronizerFtn models the Hadronizer concept,
// plus has Fortran call backs

// #include <boost/shared_ptr.hpp>
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"

namespace lhef
{
class LHEEvent;
}

namespace HepMC
{
class GenEvent;
}

namespace gen
{

  class HadronizerFtn
  {
  public:
     explicit HadronizerFtn(edm::ParameterSet const& ps);
     virtual ~HadronizerFtn() {};

     // bool generatePartons();
     virtual bool generatePartonsAndHadronize() = 0 ;
     virtual bool hadronize() = 0 ;
     virtual bool decay() = 0 ;
     virtual bool initializeForExternalPartons() = 0;
     virtual bool initializeForInternalPartons() = 0;
     virtual bool declareStableParticles() = 0 ;

     virtual const char* classname() const = 0 ;
     
     void setLHEEvent(lhef::LHEEvent* lhev) { fPartonLevel = lhev; return ; }
     
     HepMC::GenEvent* getGenEvent() { return fGenEvent; }

  protected:
     virtual bool doEvent() = 0;
     
     friend struct FortranCallback;

     void fillHeader();
     void fillEvent();  
    
     double fCOMEnergy ;
     
     bool   fHepMCVerbosity;
     unsigned int    fMaxEventsToPrint ; 
     
     lhef::LHEEvent*  fPartonLevel;
     HepMC::GenEvent* fGenEvent; 
     
  };
  
  struct FortranCallback 
  {

      FortranCallback() : fInstance(0) {} ;

      void upinit() { if (fInstance) fInstance->fillHeader(); }
      void upevnt() { if (fInstance) fInstance->fillEvent(); }

      HadronizerFtn* fInstance;
	
  } static fortranCallback;


} // end namespace

#endif
