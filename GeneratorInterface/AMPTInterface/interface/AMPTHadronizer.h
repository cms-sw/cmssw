#ifndef AMPTHadronizer_h
#define AMPTHadronizer_h

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

extern CLHEP::HepRandomEngine* _amptRandomEngine;

namespace gen
{

  extern "C" {
    float ranart_(int*);
  }

  extern "C" {
    float ran1_(int*);
  }

  class AMPTHadronizer : public BaseHadronizer {
  public:
    AMPTHadronizer(const edm::ParameterSet &);
    virtual ~AMPTHadronizer();

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
    HepMC::GenParticle*	                        build_ampt( int index, int barcode );	
    HepMC::GenVertex*                           build_ampt_vertex(int i, int id);
    bool					get_particles(HepMC::GenEvent* evt);
    bool                                        ampt_init(const edm::ParameterSet &pset);
    bool                                        call_amptset(double efrm, std::string frame, std::string proj, std::string targ, int iap, int izp, int iat, int izt);
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
    int               amptmode_;
    int               ntmax_;
    double            dt_;
    double            stringFragA_;
    double            stringFragB_;       
    bool              popcornmode_;
    double            popcornpar_;
    bool              shadowingmode_;
    bool              quenchingmode_;
    double            quenchingpar_;
    double            pthard_;
    double            mu_;
    int               izpc_;
    double            alpha_;
    double            dpcoal_;
    double            drcoal_;
    bool              ks0decay_;
    bool              phidecay_;
    int               deuteronmode_;
    int               deuteronfactor_;
    int               deuteronxsec_;
    double            minijetpt_;
    int               maxmiss_;
    int               doInitialAndFinalRadiation_;
    int               ktkick_;
    int               diquarkembedding_;
    double            diquarkpx_;
    double            diquarkpy_;
    double            diquarkx_;
    double            diquarky_;
    int               nsembd_;
    double            psembd_;
    double            tmaxembd_;
    bool              shadowingmodflag_; 
    double            shadowingfactor_;   
    double            phi0_;                  // Event plane angle
    double            sinphi0_;
    double            cosphi0_;
    bool              rotate_;                // Switch to rotate event plane
  };
} /*end namespace*/

#endif
