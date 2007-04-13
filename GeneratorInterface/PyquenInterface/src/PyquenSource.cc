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
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

#include "CLHEP/HepMC/include/PythiaWrapper6_2.h"
#include "CLHEP/HepMC/ConvertHEPEVT.h"
#include "CLHEP/HepMC/CBhepevt.h"

using namespace edm;
using namespace std;

HepMC::ConvertHEPEVT conv;

PyquenSource :: PyquenSource(const ParameterSet & pset, 
			     InputSourceDescription const& desc):
GeneratedInputSource(pset, desc), evt(0), 
abeamtarget_(pset.getUntrackedParameter<int>("aBeamTarget",207)),
bfixed_(pset.getUntrackedParameter<double>("bFixed",0.)),
cflag_(pset.getUntrackedParameter<int>("cFlag",0)),
comenergy(pset.getUntrackedParameter<double>("comEnergy",5500.)),
maxEventsToPrint_ (pset.getUntrackedParameter<int>("maxEventsToPrint",1)),
pythiaHepMCVerbosity_ (pset.getUntrackedParameter<bool>("pythiaHepMCVerbosity",false)),
pythiaPylistVerbosity_ (pset.getUntrackedParameter<int>("pythiaPylistVerbosity",0))
{
  // Default constructor

  // ------------------
  // Verbosity Level

  // Valid PYLIST arguments are: 1, 2, 3, 5, 7, 11, 12, 13
  pythiaPylistVerbosity_ = pset.getUntrackedParameter<int>("pythiaPylistVerbosity",0);
  cout << "Pythia PYLIST verbosity level = " << pythiaPylistVerbosity_ << endl;
  
  // HepMC event verbosity Level
  pythiaHepMCVerbosity_ = pset.getUntrackedParameter<bool>("pythiaHepMCVerbosity",false);
  cout << "Pythia HepMC verbosity = " << pythiaHepMCVerbosity_ << endl; 

  //Max number of events printed on verbosity level 
  maxEventsToPrint_ = pset.getUntrackedParameter<int>("maxEventsToPrint",0);
  cout << "Number of events to be printed = " << maxEventsToPrint_ << endl;


  // -------------------
  // PYTHIA parameters

  // Set PYTHIA parameters in a single ParameterSet  
  ParameterSet pythia_params = 
    pset.getParameter<ParameterSet>("PythiaParameters") ;
  
  // Read the PYTHIA parameters from the set
  vector<string> pars = 
    pythia_params.getParameter<vector<string> >("pythia");
    
  // Loop over all parameters and stop in case of mistake
  for( vector<string>::const_iterator  
	 itPar = pars.begin(); itPar != pars.end(); ++itPar ) 
    {
      static string sRandomValueSetting("MRPY(1)");
      if( 0 == itPar->compare(0,sRandomValueSetting.size(),sRandomValueSetting) ) 
	{
	  throw edm::Exception(edm::errors::Configuration,"PythiaError")
	    <<" attempted to set random number seed. Not Allowed! Use RandomNumberGeneratorService";
	}
      if( ! call_pygive(*itPar) ) 
	{
	  throw edm::Exception(edm::errors::Configuration,"PythiaError") 
	    <<" pythia did not accept the following \""<<*itPar<<"\"";
	}
    }


  //-----------------------------
  // PYTHIA settings

  //random number seed

  edm::Service<RandomNumberGenerator> rng;
  uint32_t seed = rng->mySeed();
  ostringstream sRandomSet;
  sRandomSet <<"MRPY(1)="<<seed;
  call_pygive(sRandomSet.str());

  // QCD dijet production
  call_pygive("MSEL=1");

  // to avoid stopping run
  call_pygive("MSTU(21)=1");

  // tolerance parameter to adjust fragmentation
  call_pygive("PARU(14)=1.");

  // pp multiple scattering off
  call_pygive("MSTP(81)=0");

  // hadronization off
  call_pygive("MSTP(111)=0");


  //----------------------------
  // Initialize PYTHIA
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
  // Generate PYQUEN event
  auto_ptr<HepMCProduct> bare_product(new HepMCProduct());

  // generate single partonic PYTHIA jet event
  call_pyevnt();

  // call PYQUEN to apply parton rescattering and energy loss
  PYQUEN(abeamtarget_,cflag_,bfixed_);

  // call PYTHIA to finish the hadronization
  PYEXEC();

  // fill the HEPEVT with the PYJETS event record
  call_pyhepc(1);

  // event information
  HepMC::GenEvent* evt = conv.getGenEventfromHEPEVT();
  evt->set_signal_process_id(pypars.msti[0]);      // type of the process
  evt->set_event_scale(pypars.pari[16]);           // Q^2
  evt->set_event_number(numberEventsInRun() - remainingEvents() - 1);

  // verbosity
    
  if(event() <= maxEventsToPrint_ &&
     (pythiaPylistVerbosity_ || pythiaHepMCVerbosity_)) 
    {

      // Prints PYLIST info
      if(pythiaPylistVerbosity_) 
	{
	  call_pylist(pythiaPylistVerbosity_);
	}
      
      // Prints HepMC event
      if(pythiaHepMCVerbosity_) 
	{
	  cout << "Event process = " << pypars.msti[0] << endl; 
	//	evt->print();
	}
    }
    

  if(evt)  bare_product->addHepMCData(evt );

  e.put(bare_product);
  
  return true;
}


//_____________________________________________________________________
