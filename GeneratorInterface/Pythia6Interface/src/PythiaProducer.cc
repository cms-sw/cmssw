/*
 *  $Date: 2008/04/10 22:16:14 $
 *  $Revision: 1.2 $
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


#include "GeneratorInterface/Pythia6Interface/interface/PythiaProducer.h"
#include "GeneratorInterface/Pythia6Interface/interface/PYR.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "SimDataFormats/HepMCProduct/interface/GenInfoProduct.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "CLHEP/Random/RandFlat.h"



#include <iostream>
#include "time.h"

using namespace edm;
using namespace std;


#include "HepMC/PythiaWrapper6_2.h"
#include "HepMC/IO_HEPEVT.h"

// #include "GeneratorInterface/CommonInterface/interface/Txgive.h"

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

#define PYCOMP pycomp_
extern "C" {
   int PYCOMP(int& ip);
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
#define PYGLFR pyglfr_
  extern "C" {
    void PYGLFR();
}

#define PYGLRHAD pyglrhad_
  extern "C" {
    void PYGLRHAD();
}

#define PYSTFR pyglfr_
  extern "C" {
    void PYSTLFR();
}

#define PYSTRHAD pystrhad_
  extern "C" {
    void PYSTRHAD();
}

namespace {
  HepRandomEngine& getEngineReference()
  {

   Service<RandomNumberGenerator> rng;
   if(!rng.isAvailable()) {
    throw cms::Exception("Configuration")
       << "The RandomNumberProducer module requires the RandomNumberGeneratorService\n"
          "which appears to be absent.  Please add that service to your configuration\n"
          "or remove the modules that require it.";
   }

// The Service has already instantiated an engine.  Make contact with it.
   return (rng->getEngine());
  }
}

HepMC::IO_HEPEVT conv2;

//used for defaults
  static const unsigned long kNanoSecPerSec = 1000000000;
  static const unsigned long kAveEventPerSec = 200;

PythiaProducer::PythiaProducer( const ParameterSet & pset) :
  EDProducer(), evt(0), 
  pythiaPylistVerbosity_ (pset.getUntrackedParameter<int>("pythiaPylistVerbosity",0)),
  pythiaHepMCVerbosity_ (pset.getUntrackedParameter<bool>("pythiaHepMCVerbosity",false)),
  imposeProperTimes_ (pset.getUntrackedParameter<bool>("imposeProperTimes",false)),
  maxEventsToPrint_ (pset.getUntrackedParameter<int>("maxEventsToPrint",1)),
  extCrossSect(pset.getUntrackedParameter<double>("crossSection", -1.)),
  extFilterEff(pset.getUntrackedParameter<double>("filterEfficiency", -1.)),
  comenergy(pset.getUntrackedParameter<double>("comEnergy",14000.)),
  useExternalGenerators_(false),
  useTauola_(false),
  useTauolaPolarization_(false),
  stopHadronsEnabled(false), gluinoHadronsEnabled(false),
  fRandomEngine(getEngineReference()),
  eventNumber_(0)
{
  
  // PYLIST Verbosity Level
  // Valid PYLIST arguments are: 1, 2, 3, 5, 7, 11, 12, 13
  pythiaPylistVerbosity_ = pset.getUntrackedParameter<int>("pythiaPylistVerbosity",0);
  
  // HepMC event verbosity Level
  pythiaHepMCVerbosity_ = pset.getUntrackedParameter<bool>("pythiaHepMCVerbosity",false);

  //Max number of events printed on verbosity level 
  maxEventsToPrint_ = pset.getUntrackedParameter<int>("maxEventsToPrint",0);
  
  particleID = pset.getUntrackedParameter<int>("ParticleID", 0);

// Initialize the random engine unconditionally
  randomEngine = &fRandomEngine;
  fRandomGenerator = new CLHEP::RandFlat(fRandomEngine) ;

  if(particleID) {

    cout <<" Particle ID = " << particleID << endl; 

    doubleParticle = pset.getUntrackedParameter<bool>("DoubleParticle",true);
    cout <<" double back-to-back " << doubleParticle << endl; 

    kinedata = pset.getUntrackedParameter<string>("kinematicsFile","");

    ptmin = pset.getUntrackedParameter<double>("Ptmin",20.);
    ptmax = pset.getUntrackedParameter<double>("Ptmax",420.);
    cout <<" ptmin = " << ptmin <<" ptmax = " << ptmax << endl;
  
    emin = pset.getUntrackedParameter<double>("Emin",-1);
    emax = pset.getUntrackedParameter<double>("Emax",-1);
    if ( emin > 0 && emax > 0 ) {
      cout <<" emin = " << emin <<" emax = " << emax << endl;
    }

    if(kinedata.size() < 1){
      etamin = pset.getUntrackedParameter<double>("Etamin",0.);
      etamax = pset.getUntrackedParameter<double>("Etamax",2.2);
      cout <<" etamin = " << etamin <<" etamax = " << etamax << endl;
    }else{
      ymin = pset.getUntrackedParameter<double>("ymin",0.);
      ymax = pset.getUntrackedParameter<double>("ymax",10.);
      cout <<" ymin = " << ymin <<" ymax = " << ymax << endl;
    }

    phimin = pset.getUntrackedParameter<double>("Phimin",0.);
    phimax = pset.getUntrackedParameter<double>("Phimax",360.);
    cout <<" phimin = " << phimin <<" phimax = " << phimax << endl;

    if(kinedata.size() > 0)
       fPtYGenerator = new PtYDistributor(kinedata, fRandomEngine);
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

   call_txgive_init();
  
  
   // Loop over all parameters and stop in case of a mistake
    for (vector<string>::const_iterator 
            itPar = pars.begin(); itPar != pars.end(); ++itPar) {
      call_txgive(*itPar); 
     
         } 
     
   } else if(mySet == "SLHAParameters"){   

   // Read SLHA parameter
  
   pars = pythia_params.getParameter<vector<string> >("SLHAParameters");

   // Loop over all parameters and stop in case of a mistake
    for (vector<string>::const_iterator 
            itPar = pars.begin(); itPar != pars.end(); ++itPar) {
      call_slhagive(*itPar); 
     
         } 
 
    call_slha_init(); 
  
  }
  }

   stopHadronsEnabled = pset.getUntrackedParameter<bool>("stopHadrons",false);
   gluinoHadronsEnabled = pset.getUntrackedParameter<bool>("gluinoHadrons",false);

  //Init names and pdg code of r-hadrons
   if(stopHadronsEnabled)  PYSTRHAD();
   if(gluinoHadronsEnabled)  PYGLRHAD();

#ifdef NOTYET
  //In the future, we will get the random number seed on each event and tell 
  // pythia to use that new seed
// The random engine has already been initialized.  DO NOT do it again!
  edm::Service<RandomNumberGenerator> rng;
  uint32_t seed = rng->mySeed();
  ostringstream sRandomSet;
  sRandomSet <<"MRPY(1)="<<seed;
  call_pygive(sRandomSet.str());
#endif
  
  if(particleID) 
    {
      call_pyinit( "NONE", "p", "p", comenergy );
    } else {
      call_pyinit( "CMS", "p", "p", comenergy );
    }

  // TAUOLA, etc.
  //
  useExternalGenerators_ = pset.getUntrackedParameter<bool>("UseExternalGenerators",false);
//  useTauola_ = pset.getUntrackedParameter<bool>("UseTauola", false);
//  useTauolaPolarization_ = pset.getUntrackedParameter<bool>("UseTauolaPolarization", false);
  
  if ( useExternalGenerators_ ) {
 // read External Generator parameters
    ParameterSet ext_gen_params =
       pset.getParameter<ParameterSet>("ExternalGenerators") ;
    vector<string> extGenNames =
       ext_gen_params.getParameter< vector<string> >("parameterSets");
    for (unsigned int ip=0; ip<extGenNames.size(); ++ip )
    {
      string curSet = extGenNames[ip];
      ParameterSet gen_par_set =
         ext_gen_params.getUntrackedParameter< ParameterSet >(curSet);
/*
     cout << "----------------------------------------------" << endl;
     cout << "Read External Generator parameter set "  << endl;
     cout << "----------------------------------------------" << endl;
*/
     if ( curSet == "Tauola" )
     {
        useTauola_ = true;
        if ( useTauola_ ) {
           cout << "--> use TAUOLA" << endl;
        } 
	useTauolaPolarization_ = gen_par_set.getParameter<bool>("UseTauolaPolarization");
        if ( useTauolaPolarization_ ) 
	{
           cout << "(Polarization effects enabled)" << endl;
           tauola_.enablePolarizationEffects();
        } 
	else 
	{
           cout << "(Polarization effects disabled)" << endl;
           tauola_.disablePolarizationEffects();
        }
	vector<string> cards = gen_par_set.getParameter< vector<string> >("InputCards");
	cout << "----------------------------------------------" << endl;
        cout << "Initializing Tauola" << endl;
        for( vector<string>::const_iterator
                itPar = cards.begin(); itPar != cards.end(); ++itPar )
        {
           call_txgive(*itPar);
        }
        tauola_.initialize();
        //call_pretauola(-1); // initialize TAUOLA package for tau decays
     }
    }
    // cout << "----------------------------------------------" << endl;
  }


  cout << endl; // Statically add for the output
  //********                                      
  
  produces<HepMCProduct>();
  produces<GenInfoProduct, edm::InRun>();
}


PythiaProducer::~PythiaProducer(){
  clear(); 
}

void PythiaProducer::clear() {
 
}

void PythiaProducer::endRun(Run & r, const EventSetup & es) {
 
 double cs = pypars.pari[0]; // cross section in mb
 auto_ptr<GenInfoProduct> giprod (new GenInfoProduct());
 giprod->set_cross_section(cs);
 giprod->set_external_cross_section(extCrossSect);
 giprod->set_filter_efficiency(extFilterEff);
 r.put(giprod);

 call_pystat(1);
  if ( useTauola_ ) {
    tauola_.print();
    //call_pretauola(1); // print TAUOLA decay statistics output
  }

}

void PythiaProducer::produce(Event & e, const EventSetup& es) {

    auto_ptr<HepMCProduct> bare_product(new HepMCProduct());  

    //********                                         
    //	
   if(particleID) 
      {    
        int dum;
        double pi = 3.1415927;
        int ip = 1;
        double ee=0,the=0,eta=0;
        double pmass = PYMASS(particleID);
        double phi = (phimax-phimin)*pyr_(&dum)+phimin; 

        if(kinedata.size() < 1){  // no kinematics input specified, use flat distribution, pt and eta
	  double pt  = (ptmax-ptmin)*pyr_(&dum)+ptmin;
	  double e   = (emax-emin)*pyr_(&dum)+emin;
 	  eta = (etamax-etamin)*pyr_(&dum)+etamin;
	  the = 2.*atan(exp(-eta));
	  if ( emin > pmass && emax > pmass ) { // generate single particle distribution flat in energy
	    ee = e;
	} else { // generate single particle distribution flat in pt
	  double pe = pt/sin(the);
	  ee = sqrt(pe*pe+pmass*pmass);
	}
      } else { // kinematics from input file, pt and y
     double pt  = fPtYGenerator->firePt(ptmin, ptmax);
     double y = fPtYGenerator->fireY(ymin, ymax);
     double u = exp(y);
     ee = 0.5*sqrt(pmass*pmass+pt*pt)*(u*u+1)/u;
     double pz = sqrt(ee*ee-pt*pt-pmass*pmass);
     if(y<0) pz = -pz;
       the = atan(pt/pz);
     if(pz < 0) the = pi + the;
       eta = -log(tan(the/2));
   }
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
	    //int particleID2 = -1 * particleID;
            int pythiaCode = PYCOMP(particleID);
            int has_antipart = pydat2.kchg[3-1][pythiaCode-1];
            int particleID2 = has_antipart ? -1 * particleID : particleID;
	    the = 2.*atan(exp(eta));
	    phi  = phi + 3.1415927;
	    if (phi > 2.* 3.1415927) {phi = phi - 2.* 3.1415927;}         
	    PY1ENT(ip, particleID2, ee, the, phi);
	  }
	PYEXEC();
      } else {
          if(!gluinoHadronsEnabled && !stopHadronsEnabled)
          {
             call_pyevnt();      // generate one event with Pythia
          }
          else
          {
             call_pygive("MSTJ(14)=-1");
             call_pyevnt();      // generate one event with Pythia
             call_pygive("MSTJ(14)=1");
             if(gluinoHadronsEnabled)  PYGLFR();
             if(stopHadronsEnabled)  PYSTFR();
          }

      }

    if ( useTauola_ ) {
      tauola_.processEvent();
      //call_pretauola(0); // generate tau decays with TAUOLA
    }

    // convert to stdhep format
    //
    call_pyhepc( 1 );
    
    // convert stdhep (hepevt) to hepmc
    //
    //HepMC::GenEvent* evt = conv2.getGenEventfromHEPEVT();
    HepMC::GenEvent* evt = conv2.read_next_event();

    // fix for 1-part events
    if ( particleID ) evt->set_beam_particles(0,0);

    evt->set_signal_process_id(pypars.msti[0]);
    evt->set_event_scale(pypars.pari[16]);
    ++eventNumber_;
    evt->set_event_number(eventNumber_);

    // int id1 = pypars.msti[14];
    // int id2 = pypars.msti[15];
    int id1 = pyint1.mint[14];
    int id2 = pyint1.mint[15];
    if ( id1 == 21 ) id1 = 0;
    if ( id2 == 21 ) id2 = 0; 
    double x1 = pyint1.vint[40];
    double x2 = pyint1.vint[41];  
    double Q  = pyint1.vint[50];
    double pdf1 = pyint1.vint[38];
    pdf1 /= x1 ;
    double pdf2 = pyint1.vint[39];
    pdf2 /= x2 ;
    evt->set_pdf_info( HepMC::PdfInfo(id1,id2,x1,x2,Q,pdf1,pdf2) ) ;
    
    evt->weights().push_back( pyint1.vint[96] );

 if (imposeProperTimes_) {
      int dumm;
      HepMC::GenEvent::vertex_const_iterator vbegin = evt->vertices_begin();
      HepMC::GenEvent::vertex_const_iterator vend = evt->vertices_end();
      HepMC::GenEvent::vertex_const_iterator vitr = vbegin;
      for (; vitr != vend; ++vitr ) {
            HepMC::GenVertex::particle_iterator pbegin = (*vitr)->particles_begin(HepMC::children);
            HepMC::GenVertex::particle_iterator pend = (*vitr)->particles_end(HepMC::children);
            HepMC::GenVertex::particle_iterator pitr = pbegin;
            for (; pitr != pend; ++pitr) {
                  if ((*pitr)->end_vertex()) continue;
                  if ((*pitr)->status()!=1) continue;
                  int pdgcode= abs((*pitr)->pdg_id());
                  if (pdgcode!=211 && pdgcode!=321) continue;
                  double ctau = pydat2.pmas[3][PYCOMP(pdgcode)-1];

                  double unif_rand = pyr_(&dumm);
                  // Value of 0 is excluded, so log(unif_rand) should be OK
                  double proper_length = - ctau * log(unif_rand);
                  HepMC::FourVector mom = (*pitr)->momentum();
                  double factor = proper_length/mom.m();
                  HepMC::FourVector vin = (*vitr)->position();
                  double x = vin.x() + factor * mom.px();
                  double y = vin.y() + factor * mom.py();
                  double z = vin.z() + factor * mom.pz();
                  double t = vin.t() + factor * mom.e();

                  HepMC::GenVertex* vdec = new HepMC::GenVertex(HepMC::FourVector(x,y,z,t));
                  evt->add_vertex(vdec);
                  vdec->add_particle_in((*pitr));
            }
      }
    }

    //******** Verbosity ********
    
    if(e.id().event() <= maxEventsToPrint_ &&
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

    return;
}

bool 
PythiaProducer::call_pygive(const std::string& iParm ) {

  int numWarn = pydat1.mstu[26]; //# warnings
  int numErr = pydat1.mstu[22];// # errors
  
//call the fortran routine pygive with a fortran string
  PYGIVE( iParm.c_str(), iParm.length() );  
  //  PYGIVE( iParm );  
//if an error or warning happens it is problem
  return pydat1.mstu[26] == numWarn && pydat1.mstu[22] == numErr;   
 
}

bool 
PythiaProducer::call_txgive(const std::string& iParm ) {
  
   TXGIVE( iParm.c_str(), iParm.length() );
   cout << "     " <<  iParm.c_str() << endl; 
   return 1;  
}

bool 
PythiaProducer::call_txgive_init() {
  
   TXGIVE_INIT();
   return 1;  
}

bool
PythiaProducer::call_slhagive(const std::string& iParm ) {
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
		//	try {
				FileInPath f1( shortfile );
				file = f1.fullPath();
		//	} catch(...) {
		//		cout << "SLHA file not in path. Trying anyway." << endl;
		//		file = shortfile;
		//	}
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
PythiaProducer::call_slha_init() {
  
   SLHA_INIT();
   return 1;  
}
