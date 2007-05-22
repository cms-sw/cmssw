/*
 *  $Date: 2007/05/22 13:39:22 $
 *  $Revision: 1.9 $
 *  
 *  Filip Moorgat & Hector Naves 
 *  26/10/05
 * 
 *  Patrick Janot : added the PYTHIA card reading
 *
 *  Sasha Nikitenko : added single/double particle gun
 *
 *  Holger Pieta : added FileInPath for SLHA
 *
 */


#include "GeneratorInterface/Pythia6Interface/interface/PythiaSource.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "SimDataFormats/HepMCProduct/interface/GenInfoProduct.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "Utilities/General/interface/FileInPath.h"


#include <iostream>
#include "time.h"

using namespace edm;
using namespace std;


#include "HepMC/PythiaWrapper6_2.h"
#include "HepMC/IO_HEPEVT.h"

#define PYGIVE pygive_
extern "C" {
  void PYGIVE(const char*,int length);
}

#define PY1ENT py1ent_
extern "C" {
  void PY1ENT(int& ip, int& kf, double& pe, double& the, double& phi);
}

#define PYMASS pymass_
extern "C" {
  double PYMASS(int& kf);
}

#define PYEXEC pyexec_
extern "C" {
  void PYEXEC();
}

#define TXGIVE txgive_
extern "C" {
  void TXGIVE(const char*,int length);
}

#define TXGIVE_INIT txgive_init_
extern "C" {
  void TXGIVE_INIT();
}

#define SLHAGIVE slhagive_
 extern "C" {
   void SLHAGIVE(const char*,int length);
}
  	 
#define SLHA_INIT slha_init_
 extern "C" {
   void SLHA_INIT();
}


//used for defaults
  static const unsigned long kNanoSecPerSec = 1000000000;
  static const unsigned long kAveEventPerSec = 200;

PythiaSource::PythiaSource( const ParameterSet & pset, 
			    InputSourceDescription const& desc ) :
  GeneratedInputSource(pset, desc), evt(0), 
  pythiaPylistVerbosity_ (pset.getUntrackedParameter<int>("pythiaPylistVerbosity",0)),
  pythiaHepMCVerbosity_ (pset.getUntrackedParameter<bool>("pythiaHepMCVerbosity",false)),
  maxEventsToPrint_ (pset.getUntrackedParameter<int>("maxEventsToPrint",1)),
  comenergy(pset.getUntrackedParameter<double>("comEnergy",14000.)),
  extCrossSect(pset.getUntrackedParameter<double>("crossSection", -1.)),
  extFilterEff(pset.getUntrackedParameter<double>("filterEfficiency", -1.))
  
{
  
  cout << "PythiaSource: initializing Pythia. " << endl;
  
 
  
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

  
  particleID = pset.getUntrackedParameter<int>("ParticleID", 0);
  if(particleID) {

    cout <<" Particle ID = " << particleID << endl; 

    doubleParticle = pset.getUntrackedParameter<bool>("DoubleParticle",true);
    cout <<" double back-to-back " << doubleParticle << endl; 

    ptmin = pset.getUntrackedParameter<double>("Ptmin",20.);
    ptmax = pset.getUntrackedParameter<double>("Ptmax",420.);
    cout <<" ptmin = " << ptmin <<" ptmax = " << ptmax << endl;

    etamin = pset.getUntrackedParameter<double>("Etamin",0.);
    etamax = pset.getUntrackedParameter<double>("Etamax",2.2);
    cout <<" etamin = " << etamin <<" etamax = " << etamax << endl;

    phimin = pset.getUntrackedParameter<double>("Phimin",0.);
    phimax = pset.getUntrackedParameter<double>("Phimax",360.);
    cout <<" phimin = " << phimin <<" phimax = " << phimax << endl;

    Service<RandomNumberGenerator> rng;
    long seed = (long)(rng->mySeed());
    cout << " seed= " << seed << endl ;
    fRandomEngine = new CLHEP::HepJamesRandom(seed) ;
    fRandomGenerator = new CLHEP::RandFlat(fRandomEngine) ;
    cout << "Internal BaseFlatGunSource is initialzed" << endl ;
 
  }
  // Set PYTHIA parameters in a single ParameterSet
  ParameterSet pythia_params = 
    pset.getParameter<ParameterSet>("PythiaParameters") ;
  
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
    
    if (mySet != "SLHAParameters" && mySet != "CSAParameters"){
    cout << "----------------------------------------------" << endl;
    cout << "Read PYTHIA parameter set " << mySet << endl;
    cout << "----------------------------------------------" << endl;
    
    // Loop over all parameters and stop in case of mistake
    for( vector<string>::const_iterator  
	   itPar = pars.begin(); itPar != pars.end(); ++itPar ) {
      static string sRandomValueSetting("MRPY(1)");
      if( 0 == itPar->compare(0,sRandomValueSetting.size(),sRandomValueSetting) ) {
	throw edm::Exception(edm::errors::Configuration,"PythiaError")
	  <<" attempted to set random number using pythia command 'MRPY(1)' this is not allowed.\n  Please use the RandomNumberGeneratorService to set the random number seed.";
      }
      if( ! call_pygive(*itPar) ) {
	throw edm::Exception(edm::errors::Configuration,"PythiaError") 
	  <<" pythia did not accept the following \""<<*itPar<<"\"";
      }
    }
    } else if(mySet == "CSAParameters"){   

   // Read CSA parameter
  
   pars = pythia_params.getParameter<vector<string> >("CSAParameters");

   cout << "----------------------------------------------" << endl; 
   cout << "Reading CSA parameter settings. " << endl;
   cout << "----------------------------------------------" << endl;                                                                           

   call_txgive_init();
  
  
   // Loop over all parameters and stop in case of a mistake
    for (vector<string>::const_iterator 
            itPar = pars.begin(); itPar != pars.end(); ++itPar) {
      call_txgive(*itPar); 
     
         } 
     
   } else if(mySet == "SLHAParameters"){   

   // Read SLHA parameter
  
   pars = pythia_params.getParameter<vector<string> >("SLHAParameters");

   cout << "----------------------------------------------" << endl; 
   cout << "Reading SLHA parameters. " << endl;
   cout << "----------------------------------------------" << endl;                                                                           

  
  
   // Loop over all parameters and stop in case of a mistake
    for (vector<string>::const_iterator 
            itPar = pars.begin(); itPar != pars.end(); ++itPar) {
      call_slhagive(*itPar); 
     
         } 
 
    call_slha_init(); 
  
  }
  }
  //In the future, we will get the random number seed on each event and tell 
  // pythia to use that new seed
    cout << "----------------------------------------------" << endl;
    cout << "Setting Pythia random number seed " << endl;
    cout << "----------------------------------------------" << endl;
  edm::Service<RandomNumberGenerator> rng;
  uint32_t seed = rng->mySeed();
  ostringstream sRandomSet;
  sRandomSet <<"MRPY(1)="<<seed;
  call_pygive(sRandomSet.str());

  if(particleID) 
    {
      call_pyinit( "NONE", "p", "p", comenergy );
    } else {
      call_pyinit( "CMS", "p", "p", comenergy );
    }

  cout << endl; // Stetically add for the output
  //********                                      
  
  produces<HepMCProduct>();
  produces<GenInfoProduct, edm::InRun>();
  cout << "PythiaSource: starting event generation ... " << endl;
}


PythiaSource::~PythiaSource(){
  cout << "PythiaSource: event generation done. " << endl;
  call_pystat(1);
  clear(); 
}

void PythiaSource::clear() {
 
}

void PythiaSource::endRun(Run & r) {
 
 double cs = pypars.pari[0]; // cross section in mb
 auto_ptr<GenInfoProduct> giprod (new GenInfoProduct());
 giprod->set_cross_section(cs);
 giprod->set_external_cross_section(extCrossSect);
 giprod->set_filter_efficiency(extFilterEff);
 r.put(giprod);

}

bool PythiaSource::produce(Event & e) {

    auto_ptr<HepMCProduct> bare_product(new HepMCProduct());  
    //cout << "PythiaSource: Generating event ...  " << endl;

    //********                                         
    //
    if(particleID) 
      {
	int ip = 1;
	double pt  = fRandomGenerator->fire(ptmin, ptmax);
	double eta = fRandomGenerator->fire(etamin, etamax);
	double the = 2.*atan(exp(-eta));
	double phi = fRandomGenerator->fire(phimin, phimax);
	double pmass = PYMASS(particleID);
	double pe = pt/sin(the);
	double ee = sqrt(pe*pe+pmass*pmass);

	/*
	cout <<" pt = " << pt 
	     <<" eta = " << eta 
	     <<" the = " << the 
	     <<" pe = " << pe 
	     <<" phi = " << phi 
	     <<" pmass = " << pmass 
	     <<" ee = " << ee << endl;
	*/

	phi = phi * (3.1415927/180.);

	PY1ENT(ip, particleID, ee, the, phi);

	if(doubleParticle)
	  {
	    ip = ip + 1;
	    int particleID2 = -1 * particleID;
	    the = 2.*atan(exp(eta));
	    phi  = phi + 3.1415927;
	    if (phi > 2.* 3.1415927) {phi = phi - 2.* 3.1415927;}         
	    PY1ENT(ip, particleID2, ee, the, phi);
	  }
	PYEXEC();
      } else {
	call_pyevnt();      // generate one event with Pythia
      }

    call_pyhepc( 1 );


    HepMC::IO_HEPEVT conv;
    //HepMC::GenEvent* evt = conv.getGenEventfromHEPEVT();
    HepMC::GenEvent* evt = conv.read_next_event();
    evt->set_signal_process_id(pypars.msti[0]);
    evt->set_event_scale(pypars.pari[16]);
    evt->set_event_number(numberEventsInRun() - remainingEvents() - 1);
    

    //******** Verbosity ********
    
    if(event() <= maxEventsToPrint_ &&
       (pythiaPylistVerbosity_ || pythiaHepMCVerbosity_)) {

      // Prints PYLIST info
      if(pythiaPylistVerbosity_) {
	call_pylist(pythiaPylistVerbosity_);
      }
      
      // Prints HepMC event
      if(pythiaHepMCVerbosity_) {
	cout << "Event process = " << pypars.msti[0] << endl 
	<< "----------------------" << endl;
	evt->print();
      }
    }
    
    
    //evt = reader_->fillCurrentEventData(); 
    //********                                      

    if(evt)  bare_product->addHepMCData(evt );

    e.put(bare_product);

    return true;
}

bool 
PythiaSource::call_pygive(const std::string& iParm ) {

  int numWarn = pydat1.mstu[26]; //# warnings
  int numErr = pydat1.mstu[22];// # errors
  
//call the fortran routine pygive with a fortran string
  PYGIVE( iParm.c_str(), iParm.length() );  
  //  PYGIVE( iParm );  
//if an error or warning happens it is problem
  return pydat1.mstu[26] == numWarn && pydat1.mstu[22] == numErr;   
 
}

bool 
PythiaSource::call_txgive(const std::string& iParm ) {
  
   TXGIVE( iParm.c_str(), iParm.length() );
   cout << "     " <<  iParm.c_str() << endl; 

	return 1;  
}

bool 
PythiaSource::call_txgive_init() {
  
   TXGIVE_INIT();
   cout << "  Setting CSA reweighting parameters.   "   << endl; 
   
	return 1;  
}

bool
PythiaSource::call_slhagive(const std::string& iParm ) {
	if( iParm.find( "SLHAFILE", 0 ) != string::npos ) {
		string::size_type start = iParm.find_first_of( "=" ) + 1;
		string::size_type end = iParm.length() - 1;
		string::size_type temp = iParm.find_first_of( "'", start );
		if( temp != string::npos ) {
			start = temp + 1;
			end = iParm.find_last_of( "'" ) - 1;
		}
		start = iParm.find_first_not_of( " ", start );
		end = iParm.find_last_not_of( " ", end );
		string shortfile = iParm.substr( start, end - start + 1 );
		string file;
		if( shortfile[0] == '/' ) {
			cout << "SLHA file given with absolut path." << endl;
			file = shortfile;
		} else {
			try {
				FileInPath f1( shortfile );
				file = f1.fullPath();
			} catch(...) {
				cout << "SLHA file not in path. Trying anyway." << endl;
				file = shortfile;
			}
		}
		file = "SLHAFILE = '" + file + "'";
		SLHAGIVE( file.c_str(), file.length() );
		cout << "     " <<  file.c_str() << endl;
		
	} else {
		SLHAGIVE( iParm.c_str(), iParm.length() );
		cout << "     " <<  iParm.c_str() << endl; 
	}
	return 1;
}


bool 
PythiaSource::call_slha_init() {
  
   SLHA_INIT();
   cout << "  Opening the SLHA spectrum file.   "   << endl; 
   
	return 1;  
}
