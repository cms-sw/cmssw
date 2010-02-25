#include "ExhumeProducer.h"
#include "PYR.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenRunInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "Utilities/General/interface/FileInPath.h"

#include "CLHEP/Random/RandomEngine.h"
#include "CLHEP/Random/RandFlat.h"

//ExHuME headers
#include "GeneratorInterface/ExhumeInterface/interface/Event.h"
#include "GeneratorInterface/ExhumeInterface/interface/QQ.h"
#include "GeneratorInterface/ExhumeInterface/interface/GG.h"
#include "GeneratorInterface/ExhumeInterface/interface/Higgs.h"

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
HepMC::IO_HEPEVT conv;

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
  comEnergy_(pset.getParameter<double>("comEnergy")),
  extCrossSect_(pset.getUntrackedParameter<double>("crossSection", -1.)),
  extFilterEff_(pset.getUntrackedParameter<double>("filterEfficiency", -1.)),
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
  string processType = process_pset.getParameter<string>("ProcessType");
  if(processType == "Higgs"){
	exhumeProcess_ = new Exhume::Higgs(pset);
	int higgsDecay = process_pset.getParameter<int>("HiggsDecay");
	((Exhume::Higgs*)exhumeProcess_)->SetHiggsDecay(higgsDecay);
	sigID_ = 100 + higgsDecay;
  } else if(processType == "QQ"){	
	exhumeProcess_ = new Exhume::QQ(pset);
	int quarkType = process_pset.getParameter<int>("QuarkType");
	double thetaMin = process_pset.getParameter<double>("ThetaMin");
	((Exhume::QQ*)exhumeProcess_)->SetQuarkType(quarkType);
	((Exhume::QQ*)exhumeProcess_)->SetThetaMin(thetaMin);
	sigID_ = 200 + quarkType;
  } else if(processType == "GG"){
	exhumeProcess_ = new Exhume::GG(pset);
	double thetaMin = process_pset.getParameter<double>("ThetaMin");
	((Exhume::GG*)exhumeProcess_)->SetThetaMin(thetaMin);
	sigID_ = 300;
  } else{
	sigID_ = -1;
	throw edm::Exception(edm::errors::Configuration,"ExhumeError") <<" No valid Exhume Process";
  }

  edm::Service<RandomNumberGenerator> rng;
  //uint32_t seed = rng->mySeed();
  fRandomEngine = &(rng->getEngine());
  randomEngine = fRandomEngine;
  fRandomGenerator = new CLHEP::RandFlat(fRandomEngine) ;
  //exhumeEvent_ = new Exhume::Event(*exhumeProcess_,seed);
  exhumeEvent_ = new Exhume::Event(*exhumeProcess_,randomEngine);

  double massRangeLow = process_pset.getParameter<double>("MassRangeLow");
  double massRangeHigh = process_pset.getParameter<double>("MassRangeHigh");
  exhumeEvent_->SetMassRange(massRangeLow,massRangeHigh);
  exhumeEvent_->SetParameterSpace();

  header_str << "\n"; // Stetically add for the output
  //********                                      
  
  produces<HepMCProduct>();
  produces<GenEventInfoProduct>();
  produces<GenRunInfoProduct, edm::InRun>();

  header_str << "ExhumeProducer: starting event generation ...\n";

  edm::LogInfo("") << header_str.str();
}


ExhumeProducer::~ExhumeProducer(){
  std::ostringstream footer_str;
  footer_str << "ExhumeProducer: event generation done.\n";

  edm::LogInfo("") << footer_str.str();

  clear();
}

void ExhumeProducer::clear() {
  delete exhumeEvent_;
  delete exhumeProcess_;
}

void ExhumeProducer::endRun(Run & run) {
  std::ostringstream footer_str;

  double cs = exhumeEvent_->CrossSectionCalculation();
  double eff = exhumeEvent_->GetEfficiency();
  string name = exhumeProcess_->GetName();

  footer_str << "\n" <<"   You have just been ExHuMEd." << "\n" << "\n";
  footer_str << "   The cross section for process " << name
            << " is " << cs << " fb" << "\n" << "\n";
  footer_str << "   The efficiency of event generation was " << eff << "%" << "\n" << "\n";

  edm::LogInfo("") << footer_str.str();

  auto_ptr<GenRunInfoProduct> genRunInfoProd (new GenRunInfoProduct());
  genRunInfoProd->setInternalXSec(cs);
  genRunInfoProd->setFilterEfficiency(extFilterEff_);
  genRunInfoProd->setExternalXSecLO(extCrossSect_);
  genRunInfoProd->setExternalXSecNLO(-1.);   

  run.put(genRunInfoProd);
}

void ExhumeProducer::produce(Event & event, const EventSetup& setup) {

  auto_ptr<HepMCProduct> bare_product(new HepMCProduct());  
  edm::LogInfo("") << "ExhumeProducer: Generating event ...\n";

  exhumeEvent_->Generate();
  exhumeProcess_->Hadronise();

  HepMC::GenEvent* genEvt = conv.read_next_event();
  genEvt->set_signal_process_id(sigID_);	
  genEvt->set_event_scale(pypars.pari[16]);
// JMM change
//  genEvt->set_event_number(numberEventsInRun() - remainingEvents() - 1);
  ++eventNumber_;
  genEvt->set_event_number(eventNumber_);
    
  //******** Verbosity ********
    
//  if(event() <= maxEventsToPrint_ &&
  if(event.id().event() <= maxEventsToPrint_ &&
     (pythiaPylistVerbosity_ || pythiaHepMCVerbosity_)) {

    // Prints PYLIST info
    if(pythiaPylistVerbosity_) {
       call_pylist(pythiaPylistVerbosity_);
    }
      
    // Prints HepMC event
    if(pythiaHepMCVerbosity_) {
       edm::LogInfo("") << "Event process = " << exhumeProcess_->GetName() << "\n" 
	                << "----------------------" << "\n";
       genEvt->print();
    }
  }
    
  if(genEvt)  bare_product->addHepMCData(genEvt);
  event.put(bare_product);
 
  std::auto_ptr<GenEventInfoProduct> genEventInfo(new GenEventInfoProduct(genEvt));
  event.put(genEventInfo);
}

DEFINE_FWK_MODULE(ExhumeProducer);
