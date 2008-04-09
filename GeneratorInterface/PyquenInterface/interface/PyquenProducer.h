#ifndef Pyquen_Producer_h
#define Pyquen_Producer_h

/** \class PyquenProducer
 *
 * Generates PYTHIA+PYQUEN ==> HepMC events
 * $Id: PyquenProducer.h,v 1.6 2007/10/05 15:21:52 loizides Exp $
 *
 * Camelia Mironov                                  
 *   for the Generator Interface. March 2007
 ***************************************/

#define PYCOMP pycomp_

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <map>
#include <string>
#include "HepMC/GenEvent.h"

namespace CLHEP {
class HepRandomEngine;
}

namespace edm
{
  class PyquenProducer : public EDProducer {
  public:

    /// Constructor
    PyquenProducer(const ParameterSet &);
    /// Destructor
    virtual ~PyquenProducer();

  private:
    void	     add_heavy_ion_rec(HepMC::GenEvent *evt);
    bool             call_pygive(const std::string& iParm );
    void             clear();
    virtual void     produce(Event & e, const EventSetup& es);
    bool	     pyqpythia_init(const ParameterSet &pset);
    bool	     pyquen_init(const ParameterSet &pset);

    HepMC::GenEvent *evt;
    double           abeamtarget_;            //! beam/target atomic mass number 
    unsigned int     angularspecselector_;    //! angular emitted gluon  spectrum selection 
                                              //! DEFAULT= 0 -- small angular emitted gluon spectrum
                                              //!        = 1 -- broad angular emitted gluon spectrum
                                              //!        = 2 -- collinear angular emitted gluon spectrum
    double           bfixed_;                 //! fixed impact param (fm); valid only if cflag_=0
    int              cflag_;                  //! centrality flag =0 fixed impact param, <>0 minbias
    double           comenergy;               //! collision energy  
    bool             doquench_;               //! if true perform quenching (default = true)
    bool             doradiativeenloss_;      //! DEFAULT = true
    bool             docollisionalenloss_;    //! DEFAULT = true       
    unsigned int     nquarkflavor_;           //! number of active quark flavors in qgp
                                              //! DEFAULT=0; allowed values: 0,1,2,3.
    double           qgpt0_;                  //! initial temperature of QGP
                                              //! DEFAULT = 1GeV; allowed range [0.2,2.0]GeV; 
    double           qgptau0_;                //! proper time of QGP formation
                                              //! DEFAULT = 0.1 fm/c; allowed range [0.01,10.0]fm/c;
    unsigned int     maxEventsToPrint_;       //! Events to print if verbosity  
    bool             pythiaHepMCVerbosity_;   //! HepMC verbosity flag
    unsigned int     pythiaPylistVerbosity_;  //! Pythia PYLIST Verbosity flag 
    int 	     eventNumber_;

    CLHEP::HepRandomEngine* fRandomEngine;
  };
} /*end namespace*/ 

#endif
