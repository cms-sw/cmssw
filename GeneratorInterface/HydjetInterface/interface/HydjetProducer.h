#ifndef HydjetProducer_h
#define HydjetProducer_h

// $Id: HydjetProducer.h,v 1.13 2008/01/21 12:03:08 yilmaz Exp $

/** \class HydjetProducer
*
* Generates HYDJET ==> HepMC events
*
* Camelia Mironov
*   for the Generator Interface. April 2007
*********************************************/

#define PYCOMP pycomp_

#include "FWCore/Framework/interface/EDProducer.h"
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
namespace CLHEP {
class HepRandomEngine;
}

namespace edm
{
  class HydjetProducer : public EDProducer {
  public:
    HydjetProducer(const ParameterSet &);
    virtual ~HydjetProducer();

  private:
    void					add_heavy_ion_rec(HepMC::GenEvent *evt);
    HepMC::GenParticle*	                        build_hyjet( int index, int barcode );	
    HepMC::GenVertex*                           build_hyjet_vertex(int i, int id);
    bool					call_pygive(const std::string& iParm);
    void					clear();
    bool					get_hard_particles(HepMC::GenEvent* evt, std::vector<SubEvent>& subs);
    bool                                        get_soft_particles(HepMC::GenEvent* evt, std::vector<SubEvent>& subs);
    bool                                        call_hyinit(double energy, double a, int ifb, double bmin,
							    double bmax,double bfix,int nh);
    bool					hydjet_init(const ParameterSet &pset);
    bool					hyjpythia_init(const ParameterSet &pset);
    inline double			        nuclear_radius() const;
    virtual void                                produce(Event & e, const EventSetup & es);
    
    HepMC::GenEvent   *evt;
    double            abeamtarget_;           // beam/target atomic mass number 
    double            bfixed_;                // fixed impact param (fm); valid only if cflag_=0
    double            bmax_;                  // max impact param; 
                                              // units of nucl radius
    double            bmin_;                  // min impact param; 
                                              // units of nucl radius
    int               cflag_;                 // centrality flag 
                                              // =  0 fixed impact param, 
                                              // <> 0 between bmin and bmax
    double            comenergy;              // collision energy   
    bool              doradiativeenloss_;     //! DEFAULT = true
    bool              docollisionalenloss_;   //! DEFAULT = true   
    double            fracsoftmult_;          // fraction of soft hydro induced hadronic multiplicity
                                              // proportional to no of nucleon participants
                                              // (1-fracsoftmult_)--- fraction of soft 
                                              // multiplicity proportional to the numebr 
                                              // of nucleon-nucleon binary collisions
                                              // DEFAULT=1., allowed range [0.01,1]
    double            hadfreeztemp_;          // hadron freez-out temperature
                                              // DEFAULT=0.14MeV, allowed ranges [0.08,0.2]MeV
    std::string       hymode_;                // Hydjet running mode
    unsigned int      maxEventsToPrint_;      // Events to print if verbosity  
    double            maxlongy_;              // max longitudinal collective rapidity: 
                                              // controls width of eta-spectra
                                              // DEFAULT=4, allowed range [0.01,7.0]
    double            maxtrany_;              // max transverse collective rapidity: 
                                              // controls slope of low-pt spectra
                                              // DEFAULT=1.5, allowed range [0.01,3.0]
    int               nsub_;
    int               nhard_;                 // multiplicity of PYTHIA(+PYQUEN)-induced particles in event              
    int               nmultiplicity_;         // mean soft multiplicity in central PbPb
                                              // automatically calculated for other centralitie and beams         
    int               nsoft_;                 // multiplicity of HYDRO-induced particles in event 
    unsigned int      nquarkflavor_;          //! number of active quark flavors in qgp
                                              //! DEFAULT=0; allowed values: 0,1,2,3.    
    unsigned int      pythiaPylistVerbosity_; // pythia verbosity; def=1 
    double            qgpt0_;                 // initial temperature of QGP
                                              // DEFAULT = 1GeV; allowed range [0.2,2.0]GeV; 
    double            qgptau0_;               // proper time of QGP formation
                                              // DEFAULT = 0.1 fm/c; allowed range [0.01,10.0]fm/
    unsigned int      shadowingswitch_;       // shadowing switcher
                                              // 1-ON, 0-OFF
    double            signn_;                 // inelastic nucleon nucleon cross section [mb]
                                              // DEFAULT= 58 mb
    CLHEP::HepRandomEngine* fRandomEngine;
    unsigned int      eventNumber_;
  };

double HydjetProducer::nuclear_radius() const
{
  // Return the nuclear radius derived from the 
  // beam/target atomic mass number.

  return 1.15 * pow((double)abeamtarget_, 1./3.);
}

} /*end namespace*/

#endif
