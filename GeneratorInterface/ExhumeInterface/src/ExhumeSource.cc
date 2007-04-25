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
//#include "HepMC/include/PythiaWrapper6_2.h"
//#include "HepMC/ConvertHEPEVT.h"
//#include "HepMC/CBhepevt.h"
#include "HepMC/IO_HEPEVT.h"

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

//HepMC::ConvertHEPEVT conv;
HepMC::IO_HEPEVT conv;

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

    //HepMC::GenEvent* evt = conv.getGenEventfromHEPEVT();
    HepMC::GenEvent* evt = conv.read_next_event();
    //evt->set_signal_process_id(pypars.msti[0]);
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

