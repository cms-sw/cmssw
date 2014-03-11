#ifndef gen_ExhumeHadronizer_h
#define gen_ExhumeHadronizer_h

#include <memory>

#include <boost/shared_ptr.hpp>

#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"

#include "GeneratorInterface/Core/interface/BaseHadronizer.h"

namespace lhef
{
  class LHERunInfo;
  class LHEEvent;
}

class LHEEventProduct;

namespace HepMC
{
  class GenEvent;
}

namespace Exhume{
  class Event;
  class CrossSection;
}

namespace CLHEP
{
  class HepRandomEngine;
}

namespace gen
{
  //class Pythia6Hadronizer;
  class Pythia6Service;

  class ExhumeHadronizer : public BaseHadronizer
  {
  
  public:
     ExhumeHadronizer(edm::ParameterSet const& ps);
     ~ExhumeHadronizer();

     // bool generatePartons();
     bool generatePartonsAndHadronize();
     bool hadronize();
     bool decay();
     bool residualDecay();
     bool readSettings( int );
     bool initializeForExternalPartons();
     bool initializeForInternalPartons();
     bool declareStableParticles( const std::vector<int>& );
     bool declareSpecialSettings( const std::vector<std::string>& );
     
     void finalizeEvent();

     void statistics();

     const char* classname() const;
     
  private:
     Pythia6Service* pythia6Service_;

     CLHEP::HepRandomEngine* randomEngine_;

     double comEnergy_;
                    
     //edm::ParameterSet processPSet_;
     //edm::ParameterSet paramsPSet_;
     edm::ParameterSet myPSet_;
    
     bool            hepMCVerbosity_;
     unsigned int    maxEventsToPrint_;
     unsigned int    pythiaListVerbosity_;
     
     bool convertToPDG_;

     //Pythia6Hadronizer* pythia6Hadronizer_;
     Exhume::Event* exhumeEvent_;	
     Exhume::CrossSection* exhumeProcess_;
  };
}

#endif
