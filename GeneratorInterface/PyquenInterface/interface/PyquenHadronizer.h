#ifndef Pyquen_Hadronizer_h
#define Pyquen_Hadronizer_h

/** \class PyquenHadronizer
 *
 * Generates PYTHIA+PYQUEN ==> HepMC events
 * $Id: PyquenHadronizer.h,v 1.14 2013/05/23 14:45:21 gartung Exp $
 *
 * Camelia Mironov                                  
 *   for the Generator Interface. March 2007
 ***************************************/

#include "GeneratorInterface/Core/interface/BaseHadronizer.h"
#include "GeneratorInterface/HiGenCommon/interface/BaseHiGenEvtSelector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include <map>
#include <string>
#include "HepMC/GenEvent.h"

namespace gen
{
   class Pythia6Service;

  class PyquenHadronizer : public BaseHadronizer {
  public:

    PyquenHadronizer(const edm::ParameterSet &);
    virtual ~PyquenHadronizer();

    bool generatePartonsAndHadronize();
    bool hadronize();
    bool decay();
    bool residualDecay();
    bool readSettings( int );
    bool initializeForExternalPartons();
    bool initializeForInternalPartons();
    bool declareStableParticles( const std::vector<int>& );
    bool declareSpecialSettings( const std::vector<std::string>& ) { return true; }
    virtual bool select(HepMC::GenEvent* evtTry) const { return selector_->filter(evtTry); }
    void finalizeEvent();
    void statistics();
    const char* classname() const;

  private:
    void	     add_heavy_ion_rec(HepMC::GenEvent *evt);

    bool	     pyqpythia_init(const edm::ParameterSet &pset);
    bool	     pyquen_init(const edm::ParameterSet &pset);
    const char*      nucleon();
    void             rotateEvtPlane(HepMC::GenEvent* evt, double angle);

    edm::ParameterSet pset_;
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
    int              protonSide_;
    bool             embedding_;
    double           evtPlane_;
    double           pfrac_;                  //! Proton fraction in the nucleus

    unsigned int     nquarkflavor_;           //! number of active quark flavors in qgp
                                              //! DEFAULT=0; allowed values: 0,1,2,3.
    double           qgpt0_;                  //! initial temperature of QGP
                                              //! DEFAULT = 1GeV; allowed range [0.2,2.0]GeV; 
    double           qgptau0_;                //! proper time of QGP formation
                                              //! DEFAULT = 0.1 fm/c; allowed range [0.01,10.0]fm/c;
    unsigned int     maxEventsToPrint_;       //! Events to print if verbosity  
    bool             pythiaHepMCVerbosity_;   //! HepMC verbosity flag
    unsigned int     pythiaPylistVerbosity_;  //! Pythia PYLIST Verbosity flag 

    //    CLHEP::HepRandomEngine* fRandomEngine;
    edm::InputTag   src_;
    Pythia6Service* pythia6Service_;
    std::string      filterType_;
    BaseHiGenEvtSelector* selector_;
   };
} /*end namespace*/ 

#endif
