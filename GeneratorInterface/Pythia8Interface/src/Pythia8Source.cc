/*
 *  Mikhail Kirsanov
 *  04/12/08
 */

#include <iostream>
#include <string>
#include <map>
#include <time.h>

#include <CLHEP/Random/JamesRandom.h>

#include <HepMC/GenEvent.h>

#include <Basics.h>
#include <Pythia.h>
#include <HepMCInterface.h>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/GeneratedInputSource.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenInfoProduct.h"


using namespace edm;
using namespace std;

//used for defaults
static const unsigned long kNanoSecPerSec = 1000000000;
static const unsigned long kAveEventPerSec = 200;

/** \class Pythia8Source
 *
 * Generates Pythia8 HepMC events
 *
 * Hector Naves                                  
 *   for the Generator Interface. 26/10/05 
 * Patrick Janot
 *   read all possible cards for Pythia8 Setup. 26/02/06
 *   ( port from FAMOS )
 ***************************************/

namespace edm
{
  class Pythia8Source : public GeneratedInputSource {
  public:

    /// Constructor
    Pythia8Source(const ParameterSet &, const InputSourceDescription &);
    /// Destructor
    virtual ~Pythia8Source();

    void endRun( Run& r);

  private:

    virtual bool produce(Event & e);
    void clear();
    
    /// Pythia PYLIST Verbosity flag
    unsigned int pythiaPylistVerbosity_;
    /// HepMC verbosity flag
    bool pythiaHepMCVerbosity_;
    /// Events to print if verbosity
    unsigned int maxEventsToPrint_;    
    
    Pythia8::Pythia* pythia;
    Pythia8::Event* pythia8event;
    HepMC::I_Pythia8* ToHepMC;

    // external cross section and filter efficiency
    double extCrossSect;
    double extFilterEff;
 
    // for single particle generation in pythia
    int    particleID;
    bool   doubleParticle;
    double ptmin, ptmax;
    double etamin, etamax;
    double phimin, phimax;
    double comenergy;
    
    CLHEP::HepRandomEngine* fRandomEngine;

  };
} 


Pythia8Source::Pythia8Source( const ParameterSet & pset, 
			    InputSourceDescription const& desc ) :
  GeneratedInputSource(pset, desc),
  pythiaPylistVerbosity_ (pset.getUntrackedParameter<int>("pythiaPylistVerbosity",0)),
  pythiaHepMCVerbosity_ (pset.getUntrackedParameter<bool>("pythiaHepMCVerbosity",false)),
  maxEventsToPrint_ (pset.getUntrackedParameter<int>("maxEventsToPrint",1)),
  extCrossSect(pset.getUntrackedParameter<double>("crossSection", -1.)),
  extFilterEff(pset.getUntrackedParameter<double>("filterEfficiency", -1.)),
  comenergy(pset.getUntrackedParameter<double>("comEnergy",14000.))
  
{
  
  cout << "Pythia8Source: initializing Pythia. " << endl;
  
  
  // PYLIST Verbosity Level
  // Valid PYLIST arguments are: 1, 2, 3, 5, 7, 11, 12, 13
  pythiaPylistVerbosity_ = pset.getUntrackedParameter<int>("pythiaPylistVerbosity",0);
  cout << "Pythia PYLIST verbosity level = " << pythiaPylistVerbosity_ << endl;
  
  // HepMC event verbosity Level
  pythiaHepMCVerbosity_ = pset.getUntrackedParameter<bool>("pythiaHepMCVerbosity",false);
  cout << "Pythia HepMC verbosity = " << pythiaHepMCVerbosity_ << endl; 

  //Max number of events printed on verbosity level 
  maxEventsToPrint_ = pset.getUntrackedParameter<int>("maxEventsToPrint",0);
  cout << "Number of events to be printed = " << maxEventsToPrint_ << endl;


  //In the future, we will get the random number seed on each event and tell 
  // pythia to use that new seed
    cout << "----------------------------------------------" << endl;
    cout << "Setting Pythia8 random number seed " << endl;
    cout << "----------------------------------------------" << endl;
  edm::Service<RandomNumberGenerator> rng;
  uint32_t seed = rng->mySeed();
  Pythia8::Rndm::init(seed);

    // Generator. Process selection. LHC initialization. Histogram.
  pythia = new Pythia8::Pythia;
  pythia8event = &(pythia->event);

  // Set PYTHIA parameters in a single ParameterSet
  ParameterSet pythia_params =
    pset.getParameter<ParameterSet>("Pythiaarameters") ;

  // The parameter sets to be read (default, min bias, user ...) in the
  // proper order.
  vector<string> setNames =
    pythia_params.getParameter<vector<string> >("parameterSets");

  // Loop over the sets
  for ( unsigned i=0; i<setNames.size(); ++i ) {

    string mySet = setNames[i];

    // Read the PYTHIA parameters for each set of parameters
    vector<string> pars =
      pythia_params.getParameter<vector<string> >(mySet);

    if (mySet != "CSAParameters") {
      cout << "----------------------------------------------" << endl;
      cout << "Read PYTHIA parameter set " << mySet << endl;
      cout << "----------------------------------------------" << endl;

      // Loop over all parameters and stop in case of mistake
      for( vector<string>::const_iterator
           itPar = pars.begin(); itPar != pars.end(); ++itPar ) {

        if ( ! pythia->readString(*itPar) ) {
          throw edm::Exception(edm::errors::Configuration,"PythiaError") 
          <<" pythia8 did not accept the following \""<<*itPar<<"\"";
        }

      }

    } else {
      cout << " mySet == CSAParameters, not valid for Pythia8 " << endl;
      exit(1);
    }

  }

  pythia->init( 2212, 2212, comenergy);

  pythia->settings.listChanged();

  ToHepMC = new HepMC::I_Pythia8;
//  ToHepMC->set_crash_on_problem();

  cout << endl; // Stetically add for the output
  //********                                      
  
  produces<HepMCProduct>();
  produces<GenInfoProduct, edm::InRun>();

  cout << "Pythia8Source: starting event generation ... " << endl;

}


Pythia8Source::~Pythia8Source(){
  cout << "PythiaSource: event generation done. " << endl;
  pythia->statistics();
  delete pythia;
  delete ToHepMC;
  clear();
}

void Pythia8Source::clear() {
 
}


void Pythia8Source::endRun(Run & r) {

 double cs = pythia->info.sigmaGen(); // cross section in mb
 auto_ptr<GenInfoProduct> giprod (new GenInfoProduct());
 giprod->set_cross_section(cs);
 giprod->set_external_cross_section(extCrossSect);
 giprod->set_filter_efficiency(extFilterEff);
 r.put(giprod);

}


bool Pythia8Source::produce(Event & e) {

    auto_ptr<HepMCProduct> bare_product(new HepMCProduct());  
    //cout << "
    //********                                         
    //
    
    if (!pythia->next()) return false;  // generate one event with Pythia

    HepMC::GenEvent* hepmcevt = new HepMC::GenEvent();
    ToHepMC->fill_next_event( *pythia8event, hepmcevt );

    
//    hepmcevt->set_signal_process_id(pypars.msti[0]);
    hepmcevt->set_signal_process_id(pythia->info.code());
//    hepmcevt->set_event_scale(pypars.pari[16]);
    hepmcevt->set_event_scale(pythia->info.pTHat());
    hepmcevt->set_event_number(numberEventsInRun() - remainingEvents() - 1);

    int id1 = pythia->info.id1();
    int id2 = pythia->info.id2();
    if ( id1 == 21 ) id1 = 0;
    if ( id2 == 21 ) id2 = 0;
    double x1 = pythia->info.x1();
    double x2 = pythia->info.x2();
    double Q  = pythia->info.QRen();
    double pdf1 = pythia->info.pdf1()/pythia->info.x1();
    double pdf2 = pythia->info.pdf2()/pythia->info.x2();
    hepmcevt->set_pdf_info( HepMC::PdfInfo(id1,id2,x1,x2,Q,pdf1,pdf2) ) ;

    hepmcevt->weights().push_back( pythia->info.weight() );

    //******** Verbosity ********
    
    if(event() <= maxEventsToPrint_ &&
       (pythiaPylistVerbosity_ || pythiaHepMCVerbosity_) ) {

      // Prints PYLIST info
      if(pythiaPylistVerbosity_) {
//      call_pylist(pythiaPylistVerbosity_);
        pythia->info.list();
        pythia->event.list();
    }
      
      // Prints HepMC event
      if(pythiaHepMCVerbosity_) {
//  cout << "Event process = " << pypars.msti[0] << endl 
    cout << "Event process = " << pythia->info.code() << endl
	<< "----------------------" << endl;
	hepmcevt->print();
      }
    }
    

    if(hepmcevt)  bare_product->addHepMCData(hepmcevt);

    e.put(bare_product);

    return true;
}

DEFINE_ANOTHER_FWK_INPUT_SOURCE(Pythia8Source);
