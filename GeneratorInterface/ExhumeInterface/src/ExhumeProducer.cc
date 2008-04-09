#include "GeneratorInterface/ExhumeInterface/interface/ExhumeProducer.h"
#include "GeneratorInterface/ExhumeInterface/interface/PYR.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "SimDataFormats/HepMCProduct/interface/GenInfoProduct.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "Utilities/General/interface/FileInPath.h"

#include "CLHEP/Random/RandomEngine.h"
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

#define pylist pylist_
extern "C" {
        void pylist(int*);
}
inline void call_pylist( int mode ){ pylist( &mode ); }

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
//HepMC::IO_HEPEVT conv;

// ***********************


//used for defaults
  static const unsigned long kNanoSecPerSec = 1000000000;
  static const unsigned long kAveEventPerSec = 200;

ExhumeProducer::ExhumeProducer( const ParameterSet & pset) :
  EDProducer(),
  evt(0), 
  pythiaPylistVerbosity_ (pset.getUntrackedParameter<int>("pythiaPylistVerbosity",0)),
  pythiaHepMCVerbosity_ (pset.getUntrackedParameter<bool>("pythiaHepMCVerbosity",false)),
  maxEventsToPrint_ (pset.getUntrackedParameter<int>("maxEventsToPrint",1)),
  comenergy(pset.getUntrackedParameter<double>("comEnergy",14000.)),
  extCrossSect(pset.getUntrackedParameter<double>("crossSection", -1.)),
// JMM change
//extFilterEff(pset.getUntrackedParameter<double>("filterEfficiency", -1.))
  extFilterEff(pset.getUntrackedParameter<double>("filterEfficiency", -1.)),
  eventNumber_(0)
{
  std::ostringstream header_str;

  header_str << "ExhumeProducer: initializing Exhume/Pythia.\n";
  
  // PYLIST Verbosity Level
  // Valid PYLIST arguments are: 1, 2, 3, 5, 7, 11, 12, 13
  pythiaPylistVerbosity_ = pset.getUntrackedParameter<int>("pythiaPylistVerbosity",0);
  header_str << "Pythia PYLIST verbosity level = " << pythiaPylistVerbosity_ << "\n";
  
  // HepMC event verbosity Level
  pythiaHepMCVerbosity_ = pset.getUntrackedParameter<bool>("pythiaHepMCVerbosity",false);
  header_str << "Pythia HepMC verbosity = " << pythiaHepMCVerbosity_ << "\n"; 

  //Max number of events printed on verbosity level 
  maxEventsToPrint_ = pset.getUntrackedParameter<int>("maxEventsToPrint",0);
  header_str << "Number of events to be printed = " << maxEventsToPrint_ << "\n";

  //Exhume Initialization
  ParameterSet process_pset = pset.getParameter<ParameterSet>("ExhumeProcess") ;
  ProcessType = process_pset.getParameter<string>("ProcessType");
  if(ProcessType == "Higgs"){
	ExhumeProcess = new Exhume::Higgs(pset);
	HiggsDecay = process_pset.getParameter<int>("HiggsDecay");
	double m_higgs = pset.getUntrackedParameter<double>("HiggsMass",120.);
	MassRangeLow = process_pset.getUntrackedParameter<double>("MassRangeLow",(m_higgs - 5.));
	MassRangeHigh = process_pset.getUntrackedParameter<double>("MassRangeHigh",(m_higgs + 5.));
	((Exhume::Higgs*)ExhumeProcess)->SetHiggsDecay(HiggsDecay);
	sigID = 100 + HiggsDecay;
  } else if(ProcessType == "QQ"){	
	ExhumeProcess = new Exhume::QQ(pset);
	QuarkType = process_pset.getParameter<int>("QuarkType");
	ThetaMin = process_pset.getUntrackedParameter<double>("ThetaMin");
	MassRangeLow = process_pset.getUntrackedParameter<double>("MassRangeLow",10.);
        MassRangeHigh = process_pset.getUntrackedParameter<double>("MassRangeHigh",200.);
	((Exhume::QQ*)ExhumeProcess)->SetQuarkType(QuarkType);
	((Exhume::QQ*)ExhumeProcess)->SetThetaMin(ThetaMin);
	sigID = 200 + QuarkType;
  } else if(ProcessType == "GG"){
	ExhumeProcess = new Exhume::GG(pset);
	ThetaMin = process_pset.getUntrackedParameter<double>("ThetaMin");
	MassRangeLow = process_pset.getUntrackedParameter<double>("MassRangeLow",10.);
        MassRangeHigh = process_pset.getUntrackedParameter<double>("MassRangeHigh",200.);
	((Exhume::GG*)ExhumeProcess)->SetThetaMin(ThetaMin);
	sigID = 300;
  } else{
	sigID = -1;
	throw edm::Exception(edm::errors::Configuration,"ExhumeError") <<" No valid Exhume Process";
  }

  edm::Service<RandomNumberGenerator> rng;
  uint32_t seed = rng->mySeed();
  fRandomEngine = &(rng->getEngine());
  randomEngine = fRandomEngine;
  fRandomGenerator = new CLHEP::RandFlat(fRandomEngine) ;
  ExhumeEvent = new Exhume::Event(*ExhumeProcess,seed);

  ExhumeEvent->SetMassRange(MassRangeLow,MassRangeHigh);
  ExhumeEvent->SetParameterSpace();

  //my_pythia_init();
  //pydata();

  header_str << "\n"; // Stetically add for the output
  //********                                      
  
  produces<HepMCProduct>();
  produces<GenInfoProduct, edm::InRun>();

  header_str << "ExhumeProducer: starting event generation ...\n";

  edm::LogInfo("")<<header_str.str();
}


ExhumeProducer::~ExhumeProducer(){
  std::ostringstream footer_str;
  footer_str << "ExhumeProducer: event generation done.\n";

  edm::LogInfo("") << footer_str.str();

  clear();
}

void ExhumeProducer::clear() {
  delete ExhumeEvent;
  delete ExhumeProcess;
}

void ExhumeProducer::endRun(Run & r) {
 std::ostringstream footer_str;

 double cs = ExhumeEvent->CrossSectionCalculation();
 double eff = ExhumeEvent->GetEfficiency();
 string name = ExhumeProcess->GetName();

 footer_str << "\n" <<"   You have just been ExHuMEd." << "\n" << "\n";
 footer_str << "   The cross section for process " << name
            << " is " << cs << " fb" << "\n" << "\n";
 footer_str << "   The efficiency of event generation was " << eff << "%" << "\n" << "\n";

 edm::LogInfo("") << footer_str.str();

 auto_ptr<GenInfoProduct> giprod (new GenInfoProduct());
 giprod->set_cross_section(cs);
 giprod->set_external_cross_section(extCrossSect);
 giprod->set_filter_efficiency(extFilterEff);
 r.put(giprod);
}

void ExhumeProducer::produce(Event & e, const EventSetup& es) {

    auto_ptr<HepMCProduct> bare_product(new HepMCProduct());  
    edm::LogInfo("") << "ExhumeProducer: Generating event ...\n";

    //********                                         
    //
    ExhumeEvent->Generate();
    ExhumeProcess->Hadronise();

    HepMC::IO_HEPEVT conv;	
    //HepMC::GenEvent* evt = conv.getGenEventfromHEPEVT();
    HepMC::GenEvent* evt = conv.read_next_event();
    //evt->set_signal_process_id(pypars.msti[0]);
    evt->set_signal_process_id(sigID);	
    evt->set_event_scale(pypars.pari[16]);
// JMM change
//  evt->set_event_number(numberEventsInRun() - remainingEvents() - 1);
    ++eventNumber_;
    evt->set_event_number(eventNumber_);
    

    //******** Verbosity ********
    
//  if(event() <= maxEventsToPrint_ &&
    if(e.id().event() <= maxEventsToPrint_ &&
       (pythiaPylistVerbosity_ || pythiaHepMCVerbosity_)) {

      // Prints PYLIST info
      if(pythiaPylistVerbosity_) {
	call_pylist(pythiaPylistVerbosity_);
      }
      
      // Prints HepMC event
      if(pythiaHepMCVerbosity_) {
	edm::LogInfo("") << "Event process = " << ExhumeProcess->GetName() << "\n" 
	<< "----------------------" << "\n";
	evt->print();
      }
    }
    
    
    //evt = reader_->fillCurrentEventData(); 
    //********                                      

    if(evt)  bare_product->addHepMCData(evt );

    e.put(bare_product);
}

