#include <iostream>
#include <time.h>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenRunInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"

#include "GeneratorInterface/BeamHaloGenerator/interface/BeamHaloProducer.h"
#include "GeneratorInterface/BeamHaloGenerator/interface/PYR.h"

using namespace edm;
using namespace std;

#include "HepMC/IO_HEPEVT.h"
#include "HepMC/HEPEVT_Wrapper.h"
// #include "HepMC/ConvertHEPEVT.h"
// #include "HepMC/CBhepevt.h"
#include "HepMC/WeightContainer.h"


#define KI_BHG_INIT ki_bhg_init_
extern "C" {
  void KI_BHG_INIT(long& seed);
}

#define BHSETPARAM bhsetparam_
extern "C" {
  void BHSETPARAM(int* iparam, float* fparam, const char* cparam, int length);
}

#define KI_BHG_FILL ki_bhg_fill_
extern "C" {
    void KI_BHG_FILL(int& iret, float& weight);
}

#define KI_BHG_STAT ki_bhg_stat_
extern "C" {
    void KI_BHG_STAT(int &iret);
}


// HepMC::ConvertHEPEVT conv;
//include "HepMC/HEPEVT_Wrapper.h"
static HepMC::HEPEVT_Wrapper wrapper;
static HepMC::IO_HEPEVT conv;


BeamHaloProducer::~BeamHaloProducer() {
	int iret=0;
	call_ki_bhg_stat(iret);
}


BeamHaloProducer::BeamHaloProducer( const ParameterSet & pset) :
	evt(0)
{

   int iparam[8];
   float fparam[4];
   std::string cparam;
 // -- from bhgctrl.inc
   iparam[0]  = pset.getUntrackedParameter<int>("GENMOD");
   iparam[1]  = pset.getUntrackedParameter<int>("LHC_B1");
   iparam[2]  = pset.getUntrackedParameter<int>("LHC_B2");
   iparam[3]  = pset.getUntrackedParameter<int>("IW_MUO");
   iparam[4]  = pset.getUntrackedParameter<int>("IW_HAD");
   iparam[5]  = 9999999;
   iparam[6]  = pset.getUntrackedParameter<int>("OFFSET",0);
   iparam[7]  = pset.getUntrackedParameter<int>("shift_bx");
   
   fparam[0]  = (float)pset.getUntrackedParameter<double>("EG_MIN");
   fparam[1]  = (float)pset.getUntrackedParameter<double>("EG_MAX");

   fparam[2] = (float)pset.getUntrackedParameter<double>("BXNS");
   fparam[3]  = (float)pset.getUntrackedParameter<double>("W0",1.0);

   cparam     = pset.getUntrackedParameter<std::string>("G3FNAME","input.txt");
   call_bh_set_parameters(iparam,fparam,cparam);


// -- Seed for randomnumbers
    Service<RandomNumberGenerator> rng;
    _BeamHalo_randomEngine = &(rng->getEngine());
    long seed = (long)(rng->mySeed());


// -- initialisation
   call_ki_bhg_init(seed);


  produces<HepMCProduct>();
  produces<GenEventInfoProduct>();
  produces<GenRunInfoProduct, InRun>();

  cout << "BeamHaloProducer: starting event generation ... " << endl;
}


void BeamHaloProducer::clear()
{
}

void BeamHaloProducer::produce(Event & e, const EventSetup & es) {
	// cout << "in produce " << endl;

  //    	auto_ptr<HepMCProduct> bare_product(new HepMCProduct());

	// cout << "apres autoptr " << endl;

	int iret=0;
        float weight = 0;
    	call_ki_bhg_fill(iret, weight);

// Throw an exception if call_ki_bhg_fill(...) fails.  Use the EventCorruption
// exception since it maps onto SkipEvent which is what we want to do here.

        if( iret < 0 )
          throw edm::Exception(edm::errors::EventCorruption)
            << "BeamHaloProducer: function call_ki_bhg_fill returned " << iret << endl;

	// cout << "apres fortran " << endl;


    	// HepMC::GenEvent* evt = conv.getGenEventfromHEPEVT();
	//	HepMC::GenEvent* evt = conv.read_next_event();  seems to be broken (?)
  evt = new HepMC::GenEvent();

  for (int theindex = 1; theindex<=wrapper.number_entries(); theindex++) {
  HepMC::GenVertex* Vtx = new  HepMC::GenVertex(HepMC::FourVector(wrapper.x(theindex),wrapper.y(theindex),wrapper.z(theindex),wrapper.t(theindex)));
  HepMC::FourVector p(wrapper.px(theindex),wrapper.py(theindex),wrapper.pz(theindex),wrapper.e(theindex));
  HepMC::GenParticle* Part = 
    new HepMC::GenParticle(p,wrapper.id(theindex),wrapper.status(theindex));
  Vtx->add_particle_out(Part); 
  evt->add_vertex(Vtx);
  }

  evt->set_event_number(e.id().event());

	HepMC::WeightContainer& weights = evt -> weights();
	weights.push_back(weight);
	//	evt->print();
  std::auto_ptr<HepMCProduct> CMProduct(new HepMCProduct());
  if (evt) CMProduct->addHepMCData(evt);
  e.put(CMProduct);

  auto_ptr<GenEventInfoProduct> genEventInfo(new GenEventInfoProduct(evt));
  e.put(genEventInfo);
}

void BeamHaloProducer::endRunProduce( Run &run, const EventSetup& es )
{
   // just create an empty product
   // to keep the EventContent definitions happy
   // later on we might put the info into the run info that this is a PGun
   auto_ptr<GenRunInfoProduct> genRunInfo( new GenRunInfoProduct() );
   run.put( genRunInfo );
}

bool BeamHaloProducer::call_bh_set_parameters(int* ival, float* fval, const std::string cval_string) {
  BHSETPARAM(ival,fval,cval_string.c_str(),cval_string.length());
	return true;
}

bool BeamHaloProducer::call_ki_bhg_init(long& seed) {
	KI_BHG_INIT(seed);
	return true;
}

bool BeamHaloProducer::call_ki_bhg_fill(int& iret, float& weight) {
	KI_BHG_FILL(iret,weight);
	return true;
}

bool BeamHaloProducer::call_ki_bhg_stat(int& iret) {
	KI_BHG_STAT(iret);
	return true;
}
