/*
 *
 * Generates PYQUEN HepMC events
 *
 * Original Author: Camelia Mironov
*/

#include <iostream>
#include "time.h"

#include "GeneratorInterface/PyquenInterface/interface/PyquenSource.h"
#include "GeneratorInterface/PyquenInterface/interface/PyquenWrapper.h"
#include "GeneratorInterface/CommonInterface/interface/PythiaCMS.h"

#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

#include "HepMC/GenEvent.h"
#include "HepMC/HeavyIon.h"
#include "HepMC/IO_HEPEVT.h"
#include "HepMC/PythiaWrapper.h"

using namespace edm;
using namespace std;

HepMC::IO_HEPEVT hepevtio;

PyquenSource :: PyquenSource(const ParameterSet & pset, InputSourceDescription const& desc):
GeneratedInputSource(pset, desc), evt(0), 
abeamtarget_(pset.getParameter<double>("aBeamTarget")),
bfixed_(pset.getParameter<double>("bFixed")),
cflag_(pset.getParameter<int>("cFlag")),
comenergy(pset.getParameter<double>("comEnergy")),
maxEventsToPrint_(pset.getUntrackedParameter<int>("maxEventsToPrint",1)),
pythiaHepMCVerbosity_(pset.getUntrackedParameter<bool>("pythiaHepMCVerbosity",false)),
pythiaPylistVerbosity_(pset.getUntrackedParameter<int>("pythiaPylistVerbosity",0))
{
  // Default constructor

  // Verbosity Level
  // Valid PYLIST arguments are: 1, 2, 3, 5, 7, 11, 12, 13
  pythiaPylistVerbosity_ = pset.getUntrackedParameter<int>("pythiaPylistVerbosity",0);
   LogDebug("PYLISTverbosity") << "Pythia PYLIST verbosity level = " << pythiaPylistVerbosity_ << endl;
  
  // HepMC event verbosity Level
  pythiaHepMCVerbosity_ = pset.getUntrackedParameter<bool>("pythiaHepMCVerbosity",false);
  LogDebug("HepMCverbosity")  << "Pythia HepMC verbosity = " << pythiaHepMCVerbosity_ << endl; 

  //Max number of events printed on verbosity level 
  maxEventsToPrint_ = pset.getUntrackedParameter<int>("maxEventsToPrint",0);
  LogDebug("Events2Print") << "Number of events to be printed = " << maxEventsToPrint_ << endl;

  //initialize pythia
  pyqpythia_init(pset);

  // Call PYTHIA
  call_pyinit("CMS", "p", "p", comenergy);  
  
  cout<<endl;

  produces<HepMCProduct>();
}


//_____________________________________________________________________
PyquenSource::~PyquenSource()
{
  // distructor

  call_pystat(1);

  clear();
}


//_____________________________________________________________________
void PyquenSource::add_heavy_ion_rec(HepMC::GenEvent *evt)
{
  HepMC::HeavyIon *hi = new HepMC::HeavyIon(
    -1,                                 // Ncoll_hard
    -1,                                 // Npart_proj
    -1,                                 // Npart_targ
    -1,                                 // Ncoll
    -1,                                 // spectator_neutrons
    -1,                                 // spectator_protons
    -1,                                 // N_Nwounded_collisions
    -1,                                 // Nwounded_N_collisions
    -1,                                 // Nwounded_Nwounded_collisions
    plfpar.bgen,                        // impact_parameter in [fm]
    0,                                  // event_plane_angle
    0,                                  // eccentricity
    -1                                  // sigma_inel_NN
  );

  evt->set_heavy_ion(*hi);
}


//______________________________________________________________________
bool PyquenSource::call_pygive(const std::string& iParm ) 
{
  // Set Pythia parameters

  int numWarn = pydat1.mstu[26]; //# warnings
  int numErr = pydat1.mstu[22];// # errors
  // call the fortran routine pygive with a fortran string
  PYGIVE( iParm.c_str(), iParm.length() );  
  // if an error or warning happens it is problem
  return pydat1.mstu[26] == numWarn && pydat1.mstu[22] == numErr;   
}


//____________________________________________________________________
void PyquenSource::clear()
{

}


//_____________________________________________________________________
bool PyquenSource::produce(Event & e)
{

  edm::LogInfo("PYQUENabeamtarget") << "abeamtarget_ =  " << abeamtarget_;
  edm::LogInfo("PYQUENcflag") << "cflag_ = " << cflag_;
  edm::LogInfo("PYQUENbfixed") << "bfixed_ = " << bfixed_;
  edm::LogInfo("PYQUENinAction") << "##### Calling PYQUEN(abeamtarget_,cflag_,bfixed_) ####";

  // Generate PYQUEN event
  // generate single partonic PYTHIA jet event
  call_pyevnt();

  // call PYQUEN to apply parton rescattering and energy loss
  PYQUEN(abeamtarget_,cflag_,bfixed_);

  // call PYTHIA to finish the hadronization
  PYEXEC();

  // fill the HEPEVT with the PYJETS event record
  call_pyhepc(1);

  // event information
  HepMC::GenEvent* evt = hepevtio.read_next_event();
  evt->set_signal_process_id(pypars.msti[0]);      // type of the process
  evt->set_event_scale(pypars.pari[16]);           // Q^2
  evt->set_event_number(numberEventsInRun() - remainingEvents() - 1);

  add_heavy_ion_rec(evt);

  auto_ptr<HepMCProduct> bare_product(new HepMCProduct());
  bare_product->addHepMCData(evt );
  e.put(bare_product); 

  // verbosity
  if( event() <= maxEventsToPrint_ && ( pythiaPylistVerbosity_ || pythiaHepMCVerbosity_ )){ 
      // Prints PYLIST info
      if( pythiaPylistVerbosity_ ){
	  call_pylist(pythiaPylistVerbosity_);
      }
      
      // Prints HepMC event
      if( pythiaHepMCVerbosity_ ){
	  cout << "Event process = " << pypars.msti[0] << endl; 
	//	evt->print();
      }
  }
    
  return true;
}


//_____________________________________________________________________
bool PyquenSource::pyqpythia_init(const ParameterSet & pset)
{
  //initialize PYTHIA

  //random number seed
  edm::Service<RandomNumberGenerator> rng;
  uint32_t seed = rng->mySeed();
  ostringstream sRandomSet;
  sRandomSet << "MRPY(1)=" << seed;
  call_pygive(sRandomSet.str());

    // Set PYTHIA parameters in a single ParameterSet  
  ParameterSet pythia_params = pset.getParameter<ParameterSet>("PythiaParameters") ;
  
  // The parameter sets to be read
  vector<string> setNames = pythia_params.getParameter<vector<string> >("parameterSets");

    // Loop over the sets
  for ( unsigned i=0; i<setNames.size(); ++i ) {
    string mySet = setNames[i];
    
    // Read the PYTHIA parameters for each set of parameters
    vector<string> pars = pythia_params.getParameter<vector<string> >(mySet);
    
    cout << "----------------------------------------------" << endl;
    cout << "Read PYTHIA parameter set " << mySet << endl;
    cout << "----------------------------------------------" << endl;
    
    // Loop over all parameters and stop in case of mistake
    for( vector<string>::const_iterator itPar = pars.begin(); itPar != pars.end(); ++itPar ) {
      static string sRandomValueSetting("MRPY(1)");
      if( 0 == itPar->compare(0,sRandomValueSetting.size(),sRandomValueSetting) ) {
	throw edm::Exception(edm::errors::Configuration,"PythiaError")
	  <<" Attempted to set random number using 'MRPY(1)'. NOT ALLOWED! \n Use RandomNumberGeneratorService to set the random number seed.";
      }
      if( !call_pygive(*itPar) ) {
	throw edm::Exception(edm::errors::Configuration,"PythiaError") 
	  <<"PYTHIA did not accept \""<<*itPar<<"\"";
      }
    }
  }

  return true;
}

