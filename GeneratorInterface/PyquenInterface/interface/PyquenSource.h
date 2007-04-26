#ifndef Pyquen_Source_h
#define Pyquen_Source_h

/** \class PyquenSource
 *
 * Generates PYTHIA+PYQUEN ==> HepMC events
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

    /// Interface to the PYGIVE pythia routine, with add'l protections  
    bool call_pygive(const std::string& iParm );
    void clear();
    virtual bool produce(Event & e);
    
    HepMC::GenEvent *evt;
    int              abeamtarget_;            //! beam/target atomic mass number 
    double           bfixed_;                 //! fixed impact param (fm); valid only if cflag_=0
   
    int              cflag_;                  //! centrality flag =0 fixed impact param, <>0 minbias
    double           comenergy;               //! collision energy            
    unsigned int     maxEventsToPrint_;       //! Events to print if verbosity  
    bool             pythiaHepMCVerbosity_;   //  HepMC verbosity flag
    unsigned int     pythiaPylistVerbosity_;  //! Pythia PYLIST Verbosity flag

  
  };
} 

#endif
