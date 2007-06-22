#ifndef HydjetSource_h
#define HydjetSource_h

// $Id: HydjetSource.h,v 1.7 2007/05/21 14:49:07 mironov Exp $

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
#include <map>
#include <string>
#include <vector>
#include <math.h>

namespace HepMC {
  class GenEvent;
  class GenParticle;
}

namespace edm
{
  class HydjetSource : public GeneratedInputSource {
  public:
    HydjetSource(const ParameterSet &, const InputSourceDescription &);
    virtual ~HydjetSource();

  private:
    void						add_heavy_ion_rec(HepMC::GenEvent *evt);
    bool						build_vertices(int i, std::vector<HepMC::GenParticle*>& luj_entries,
                                                                       HepMC::GenEvent* evt);
    HepMC::GenParticle*	build_particle( int index );	
    bool						call_hyjgive(const std::string& iParm);
    bool						call_pygive(const std::string& iParm);
    void						clear();
    bool						get_hydjet_particles(HepMC::GenEvent* evt);
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
    std::string      hymode_;                 // Hydjet running mode
    unsigned int     maxEventsToPrint_;       // Events to print if verbosity      
    int              nhard_;                  //!multiplicity of PYTHIA(+PYQUEN)-induced particles in event              
    int              nmultiplicity_;          // mean soft multiplicity in central PbPb
                                              // automatically calculated for other centralitie and beams         
    int              nsoft_;                  //!multiplicity of HYDRO-induced particles in event     
    double           ptmin_;                  // min transverse  mom of the hard scattering
    unsigned int     pythiaPylistVerbosity_;  // pythia verbosity; def=1 

  };

double HydjetSource::nuclear_radius() const
{
  // Return the nuclear radius derived from the 
  // beam/target atomic mass number.

  return 1.15 * pow((double)abeamtarget_, 1./3.);
}

} /*end namespace*/

#endif
