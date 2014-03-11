#ifndef HijingHadronizer_h
#define HijingHadronizer_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "GeneratorInterface/Core/interface/BaseHadronizer.h"
#include "CLHEP/Random/RandomEngine.h"

#include <map>
#include <string>
#include <vector>
#include <math.h>

namespace HepMC {
  class GenEvent;
  class GenParticle;
  class GenVertex;
}

extern CLHEP::HepRandomEngine* hijRandomEngine;

namespace gen
{
  extern "C" {
    float hijran_(int*);
  }

  class HijingHadronizer : public BaseHadronizer {
  public:
    HijingHadronizer(const edm::ParameterSet &);
    virtual ~HijingHadronizer();

    bool generatePartonsAndHadronize();
    bool hadronize();
    bool decay();
    bool residualDecay();
    bool readSettings( int ) { return true; }
    bool initializeForExternalPartons();
    bool initializeForInternalPartons();
    bool declareStableParticles( const std::vector<int>& );
    bool declareSpecialSettings( const std::vector<std::string>& ) { return true; }
    
    void finalizeEvent();
    void statistics();
    const char* classname() const;

  private:
    
    void					add_heavy_ion_rec(HepMC::GenEvent *evt);
    HepMC::GenParticle*	                        build_hijing( int index, int barcode );	
    HepMC::GenVertex*                           build_hijing_vertex(int i, int id);
    bool					get_particles(HepMC::GenEvent* evt);
    bool                                        call_hijset(double efrm, std::string frame, std::string proj, 
                                                            std::string targ, int iap, int izp, int iat, int izt);
    //    inline double			        nuclear_radius() const;
    void                                        rotateEvtPlane();

    HepMC::GenEvent   *evt;
    edm::ParameterSet pset_;
    double            bmax_;                  // max impact param; 
                                              // units of nucl radius
    double            bmin_;                  // min impact param; 
                                              // units of nucl radius
    double            efrm_;                  // collision energy  
    std::string       frame_;
    std::string       proj_;
    std::string       targ_;
    int               iap_;
    int               izp_;
    int               iat_;
    int               izt_;    

//    unsigned int      maxEventsToPrint_;      // Events to print if verbosity  
//    unsigned int      pythiaPylistVerbosity_; // pythia verbosity; def=1 

    double            phi0_;                  // Event plane angle
    double            sinphi0_;
    double            cosphi0_;
    bool              rotate_;                // Switch to rotate event plane

    //    unsigned int      shadowingswitch_;       // shadowing switcher
                                              // 1-ON, 0-OFF
    //    double            signn_;                 // inelastic nucleon nucleon cross section [mb]
                                              // DEFAULT= 58 mb
    //    CLHEP::HepRandomEngine* fRandomEngine;
//    Pythia6Service* pythia6Service_;
  };

} /*end namespace*/

#endif
