#ifndef Pyquen_Source_h
#define Pyquen_Source_h

/** \class PyquenSource
 *
 * Generates PYTHIA+PYQUEN ==> HepMC events
 * $Id: PyquenSource.h,v 1.8 2008/12/01 12:40:26 yilmaz Exp $
 *
 * Camelia Mironov                                  
 *   for the Generator Interface. March 2007
 ***************************************/

#define PYCOMP pycomp_

#include "FWCore/Framework/interface/GeneratedInputSource.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <map>
#include <string>
#include "HepMC/GenEvent.h"

namespace edm
{
  class PyquenSource : public GeneratedInputSource {
  public:

    /// Constructor
    PyquenSource(const ParameterSet &, const InputSourceDescription &);
    /// Destructor
    virtual ~PyquenSource();

  private:
    void	     add_heavy_ion_rec(HepMC::GenEvent *evt);
    bool             call_pygive(const std::string& iParm );
    void             clear();
    virtual bool     produce(Event & e);
    bool	     pyqpythia_init(const ParameterSet &pset);
    bool	     pyquen_init(const ParameterSet &pset);

    HepMC::GenEvent *evt;
    double           abeamtarget_;            //! beam/target atomic mass number 
    unsigned int     angularspecselector_;    //! angular emitted gluon  spectrum selection 
                                              //! DEFAULT= 0 -- small angular emitted gluon spectrum
                                              //!        = 1 -- broad angular emitted gluon spectrum
                                              //!        = 2 -- collinear angular emitted gluon spectrum
    double           bmin_;                   //! min impact param (fm); valid only if cflag_!=0       
    double           bmax_;                   //! max impact param (fm); valid only if cflag_!=0       
    double           bfixed_;                 //! fixed impact param (fm); valid only if cflag_=0
    int              cflag_;                  //! centrality flag =0 fixed impact param, <>0 minbias
    double           comenergy;               //! collision energy  
    bool             doquench_;               //! if true perform quenching (default = true)
    bool             doradiativeenloss_;      //! DEFAULT = true
    bool             docollisionalenloss_;    //! DEFAULT = true       
    bool             doIsospin_;              //! Run n&p with proper ratios; if false, only p+p collisions
    unsigned int     nquarkflavor_;           //! number of active quark flavors in qgp
                                              //! DEFAULT=0; allowed values: 0,1,2,3.
    double           qgpt0_;                  //! initial temperature of QGP
                                              //! DEFAULT = 1GeV; allowed range [0.2,2.0]GeV; 
    double           qgptau0_;                //! proper time of QGP formation
                                              //! DEFAULT = 0.1 fm/c; allowed range [0.01,10.0]fm/c;
    unsigned int     maxEventsToPrint_;       //! Events to print if verbosity  
    bool             pythiaHepMCVerbosity_;   //! HepMC verbosity flag
    unsigned int     pythiaPylistVerbosity_;  //! Pythia PYLIST Verbosity flag 
  };
} /*end namespace*/ 

#endif
