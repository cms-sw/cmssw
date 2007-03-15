/*
 *  Modification fro Exhume 
 *  02/07
 * 
 */


#include "GeneratorInterface/ExhumeInterface/interface/ExhumeSource.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "CLHEP/Random/JamesRandom.h"
#include "CLHEP/Random/RandFlat.h"

#include <iostream>
#include "time.h"

using namespace edm;
using namespace std;


// Generator modifications
// ***********************
//#include "CLHEP/HepMC/include/PythiaWrapper6_2.h"
#include "CLHEP/HepMC/ConvertHEPEVT.h"
#include "CLHEP/HepMC/CBhepevt.h"


void call_pylist(int);
void call_pystat(int);
/*
#define my_pythia_init my_pythia_init_
extern "C" {
	void my_pythia_init();
}
#define pydata pydata_
extern "C" {
    void pydata(void);
}
*/
extern struct {
	int mstp[200];
	double parp[200];
	int msti[200];
	double pari[200];
} pypars_;
#define pypars pypars_


/*
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
*/
HepMC::ConvertHEPEVT conv;
// ***********************


//used for defaults
  static const unsigned long kNanoSecPerSec = 1000000000;
  static const unsigned long kAveEventPerSec = 200;

ExhumeSource::ExhumeSource( const ParameterSet & pset, 
			    InputSourceDescription const& desc ) :
  GeneratedInputSource(pset, desc), evt(0), 
  pythiaPylistVerbosity_ (pset.getUntrackedParameter<int>("pythiaPylistVerbosity",0)),
  pythiaHepMCVerbosity_ (pset.getUntrackedParameter<bool>("pythiaHepMCVerbosity",false)),
  maxEventsToPrint_ (pset.getUntrackedParameter<int>("maxEventsToPrint",1)),
  comenergy(pset.getUntrackedParameter<double>("comEnergy",14000.)),
  ProcessType(pset.getParameter<string>("ProcessType")),
  HiggsDecay(pset.getParameter<int>("HiggsDecay")),
  QuarkType(pset.getParameter<int>("QuarkType")),
  ThetaMin(pset.getParameter<double>("ThetaMin")),
  MassRangeLow(pset.getParameter<double>("MassRangeLow")),
  MassRangeHigh(pset.getParameter<double>("MassRangeHigh"))		
{
  
  cout << "ExhumeSource: initializing Exhume/Pythia. " << endl;
  
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

  //Exhume Initialization
  if(ProcessType == "Higgs"){
	ExhumeProcess = new Exhume::Higgs(pset);
	((Exhume::Higgs*)ExhumeProcess)->SetHiggsDecay(HiggsDecay);
	sigID = 100 + HiggsDecay;
  } else if(ProcessType == "QQ"){	
	ExhumeProcess = new Exhume::QQ(pset);
	((Exhume::QQ*)ExhumeProcess)->SetQuarkType(QuarkType);
	((Exhume::QQ*)ExhumeProcess)->SetThetaMin(ThetaMin);
	sigID = 200 + QuarkType;
  } else if(ProcessType == "GG"){
	ExhumeProcess = new Exhume::GG(pset);
	((Exhume::GG*)ExhumeProcess)->SetThetaMin(ThetaMin);
	sigID = 300;
  } else{
	sigID = -1;
	throw edm::Exception(edm::errors::Configuration,"ExhumeError") <<" No valid Exhume Process";
  }

  edm::Service<RandomNumberGenerator> rng;
  uint32_t seed = rng->mySeed();
  ExhumeEvent = new Exhume::Event(*ExhumeProcess,seed);

  ExhumeEvent->SetMassRange(MassRangeLow,MassRangeHigh);
  ExhumeEvent->SetParameterSpace();

  /*
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

   // Loop over all parameters and stop in case of a mistake
    for (vector<string>::const_iterator 
            itPar = pars.begin(); itPar != pars.end(); ++itPar) {
      call_txgive(*itPar); 
     
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
  
  call_pyinit( "CMS", "p", "p", comenergy );
  */
  //my_pythia_init();
  //pydata();

  cout << endl; // Stetically add for the output
  //********                                      
  
  produces<HepMCProduct>();
  cout << "ExhumeSource: starting event generation ... " << endl;
}


ExhumeSource::~ExhumeSource(){
  cout << "ExhumeSource: event generation done. " << endl;
  clear(); 
}

void ExhumeSource::clear() {
  double XS = ExhumeEvent->CrossSectionCalculation();
  double Eff = ExhumeEvent->GetEfficiency();
  string Name = ExhumeProcess->GetName();

  cout<<endl<<"   You have just been ExHuMEd."<<endl<<endl;;
  cout<<"   The cross section for process "<<Name
            <<" is "<<XS<<" fb"<<endl<<endl;
  cout<<"   The efficiency of event generation was "<<Eff<<"%"<<endl<<endl;

  delete ExhumeEvent;
  delete ExhumeProcess;
}


bool ExhumeSource::produce(Event & e) {

    auto_ptr<HepMCProduct> bare_product(new HepMCProduct());  
    cout << "ExhumeSource: Generating event ...  " << endl;

    //********                                         
    //
    ExhumeEvent->Generate();
    ExhumeProcess->Hadronise();

    HepMC::GenEvent* evt = conv.getGenEventfromHEPEVT();
    //evt->set_signal_process_id(pypars.msti[0]);
    //int signalid = 100 + HiggsDecay; 	
    evt->set_signal_process_id(sigID);	
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
	cout << "Event process = " << ExhumeProcess->GetName() << endl 
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

/*bool 
ExhumeSource::call_pygive(const std::string& iParm ) {

  int numWarn = pydat1.mstu[26]; //# warnings
  int numErr = pydat1.mstu[22];// # errors
  
//call the fortran routine pygive with a fortran string
  PYGIVE( iParm.c_str(), iParm.length() );  
  //  PYGIVE( iParm );  
//if an error or warning happens it is problem
  return pydat1.mstu[26] == numWarn && pydat1.mstu[22] == numErr;   
 
}

bool 
ExhumeSource::call_txgive(const std::string& iParm ) {
  
   TXGIVE( iParm.c_str(), iParm.length() );
   cout << "     " <<  iParm.c_str() << endl; 

	return 1;  
}*/
