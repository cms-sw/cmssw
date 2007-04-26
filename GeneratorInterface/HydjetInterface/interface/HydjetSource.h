#ifndef Hydjet_Source_h
#define Hydjet_Source_h


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
#include "CLHEP/HepMC/GenEvent.h"

namespace edm
{
  class HydjetSource : public GeneratedInputSource {
  public:

    /// Constructor
    HydjetSource(const ParameterSet &, const InputSourceDescription &);
    /// Destructor
    virtual ~HydjetSource();

    enum EHydjetMode{  
      kHydroOnly  = 0, //jet production off (pure HYDRO event)
      kHydroJets  = 1, //jet production on, jet quenching off (HYDRO+njet*PYTHIA events)
      kHydroQJets = 2, //jet production & jet quenching on (HYDRO+njet*PYQUEN events)
      kJetsOnly   = 3, //jet production on, jet quenching off, HYDRO off (njet*PYTHIA events)
      kQJetsOnly  = 4  //jet production & jet quenching on, HYDRO off (njet*PYQUEN events)
    };


  private:
    bool build_vertices(int i,
	std::vector<HepMC::GenParticle*>& luj_entries, HepMC::GenEvent* evt);
    HepMC::GenParticle* build_particle( int index );	
    bool call_hyjgive(const std::string& iParm);
    bool call_pygive(const std::string& iParm);
    void clear();
    bool get_hydjet_particles(HepMC::GenEvent* evt);
    bool hyjhydro_init();
    bool hyjpythia_init();
    virtual bool produce(Event & e);
    
    HepMC::GenEvent *evt;
    double           abeamtarget_;            // beam/target atomic mass number 
    double           bfixed_;                 // fixed impact param (fm); valid only if cflag_=0
    double           bmax_;                   // max impact param; 
                                              // units of nucl radius
    double           bmin_;                   // min impact param; 
                                              // units of nucl radius
    int              cflag_;                  // centrality flag 
                                              // =  0 fixed impact param, 
                                              // <> 0 between bmin and bmax
    double           comenergy;               // collision energy          


    EHydjetMode      hyMode;                  // Hydjet running mode
    unsigned int     maxEventsToPrint_;       // Events to print if verbosity                     
    int              nhard_;                  //!multiplicity of PYTHIA(+PYQUEN)-induced particles in event  
    int              nmultiplicity_;          // mean soft multiplicity in central PbPb
                                              // automatically calculated for other centralities and beams         
    int              nsoft_;                  //!multiplicity of HYDRO-induced particles in event            
    unsigned int     pythiaPylistVerbosity_;  // pythia verbosity; default=1 
  };
} 


#endif
