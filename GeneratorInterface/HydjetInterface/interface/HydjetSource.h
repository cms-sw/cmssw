#ifndef HydjetSource_h
#define HydjetSource_h

// $Id: HydjetSource.h,v 1.9 2007/08/15 14:50:28 mironov Exp $

/** \class HydjetSource
*
* Generates HYDJET ==> HepMC events
*
* Camelia Mironov
*   for the Generator Interface. April 2007
*********************************************/

#define PYCOMP pycomp_

#include "FWCore/Framework/interface/GeneratedInputSource.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/HiGenData/interface/SubEvent.h"
#include <map>
#include <string>
#include <vector>
#include <math.h>

namespace HepMC {
  class GenEvent;
  class GenParticle;
  class GenVertex;
}

namespace edm
{
  class HydjetSource : public GeneratedInputSource {
  public:
    HydjetSource(const ParameterSet &, const InputSourceDescription &);
    virtual ~HydjetSource();

  private:
    void						add_heavy_ion_rec(HepMC::GenEvent *evt);
    HepMC::GenVertex*                                   build_hyjet_vertex(int i, int id);
    HepMC::GenVertex*                                   build_lujet_vertex(int i, int id);
    HepMC::GenParticle*	                                build_hyjet( int index, int barcode );	
    HepMC::GenParticle*                                 build_lujet( int index, int barcode );
    bool						call_pygive(const std::string& iParm);
    void						clear();
    bool						get_hard_particles(HepMC::GenEvent* evt, std::vector<SubEvent>& subs);
    bool                                                get_soft_particles(HepMC::GenEvent* evt, std::vector<SubEvent>& subs);
    bool                                                call_hyinit(float energy);
    bool						hyjhydro_init(const ParameterSet &pset);
    bool						hyjpythia_init(const ParameterSet &pset);
    inline double			                nuclear_radius() const;
    virtual bool                                        produce(Event & e);
    
    HepMC::GenEvent *evt;
    float            abeamtarget_;            // beam/target atomic mass number 
    float            bfixed_;                 // fixed impact param (fm); valid only if cflag_=0
    float            bmax_;                   // max impact param; 
                                              // units of nucl radius
    float            bmin_;                   // min impact param; 
                                              // units of nucl radius
    int              cflag_;                  // centrality flag 
                                              // =  0 fixed impact param, 
                                              // <> 0 between bmin and bmax
    float            comenergy;               // collision energy   
    float            fracsoftmult_;           // fraction of soft hydro induced hadronic multiplicity
                                              // proportional to no of nucleon participants
                                              // (1-fracsoftmult_)--- fraction of soft 
                                              // multiplicity proportional to the numebr 
                                              // of nucleon-nucleon binary collisions
                                              // DEFAULT=1., allowed range [0.01,1]
    float            hadfreeztemp_;           // hadron freez-out temperature
                                              // DEFAULT=0.14MeV, allowed ranges [0.08,0.2]MeV
    std::string      hymode_;                 // Hydjet running mode
    unsigned int     maxEventsToPrint_;       // Events to print if verbosity  
    float            maxlongy_;               // max longitudinal collective rapidity: 
                                              // controls width of eta-spectra
                                              // DEFAULT=5, allowed range [0.01,7.0]
    float            maxtrany_;               // max transverse collective rapidity: 
                                              // controls slope of low-pt spectra
                                              // DEFAULT=1, allowed range [0.01,3.0]
    int              nsub_;
    int              nhard_;                  // multiplicity of PYTHIA(+PYQUEN)-induced particles in event              
    int              nparton_;                // number of initial partons involved 
    int              nmultiplicity_;          // mean soft multiplicity in central PbPb
                                              // automatically calculated for other centralitie and beams         
    int              nsoft_;                  // multiplicity of HYDRO-induced particles in event 
    unsigned int     nquarkflavor_;           //! number of active quark flavors in qgp
                                              //! DEFAULT=0; allowed values: 0,1,2,3.    
    unsigned int     pythiaPylistVerbosity_;  // pythia verbosity; def=1 
    double           qgpt0_;                  // initial temperature of QGP
                                              // DEFAULT = 1GeV; allowed range [0.2,2.0]GeV; 
    double           qgptau0_;                // proper time of QGP formation
                                              // DEFAULT = 0.1 fm/c; allowed range [0.01,10.0]fm/
    double           signn_;                  // inelastic nucleon nucleon cross section [mb]
                                              // DEFAULT= 58 mb
  };

double HydjetSource::nuclear_radius() const
{
  // Return the nuclear radius derived from the 
  // beam/target atomic mass number.

  return 1.15 * pow((double)abeamtarget_, 1./3.);
}

} /*end namespace*/

#endif
