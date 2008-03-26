
#include "GeneratorInterface/BeamHaloGenerator/interface/BeamHaloSource.h"

#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

#include <iostream>
#include "time.h"

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
HepMC::HEPEVT_Wrapper wrapper;
HepMC::IO_HEPEVT conv;


BeamHaloSource::~BeamHaloSource() {
	int iret=0;
	call_ki_bhg_stat(iret);
}


BeamHaloSource::BeamHaloSource( const ParameterSet & pset,
                            InputSourceDescription const& desc ) :
	GeneratedInputSource(pset, desc), evt(0)
{

   int iparam[6];
   float fparam[3];
   std::string cparam;
 // -- from bhgctrl.inc
   iparam[0]  = pset.getUntrackedParameter<int>("GENMOD");
   iparam[1]  = pset.getUntrackedParameter<int>("LHC_B1");
   iparam[2]  = pset.getUntrackedParameter<int>("LHC_B2");
   iparam[3]  = pset.getUntrackedParameter<int>("IW_MUO");
   iparam[4]  = pset.getUntrackedParameter<int>("IW_HAD");
   iparam[5]  = numberEventsInRun();
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
    long seed = (long)(rng->mySeed());


// -- initialisation
   call_ki_bhg_init(seed);


  produces<HepMCProduct>();
  cout << "BeamHaloSource: starting event generation ... " << endl;

}


void BeamHaloSource::clear()
{
}

bool BeamHaloSource::produce(Event & e) {
	// cout << "in produce " << endl;

  //    	auto_ptr<HepMCProduct> bare_product(new HepMCProduct());

	// cout << "apres autoptr " << endl;

	int iret=0;
        float weight = 0;
    	call_ki_bhg_fill(iret, weight);

	if( iret < 0 ) return false;

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

  evt->set_event_number(event());

	HepMC::WeightContainer& weights = evt -> weights();
	weights.push_back(weight);
	//	evt->print();
  std::auto_ptr<HepMCProduct> CMProduct(new HepMCProduct());
  if (evt) CMProduct->addHepMCData(evt );
  e.put(CMProduct);
    
    return true;
}



bool BeamHaloSource::call_bh_set_parameters(int* ival, float* fval, const std::string cval_string) {
  BHSETPARAM(ival,fval,cval_string.c_str(),cval_string.length());
	return true;
}

bool BeamHaloSource::call_ki_bhg_init(long& seed) {
	KI_BHG_INIT(seed);
	return true;
}

bool BeamHaloSource::call_ki_bhg_fill(int& iret, float& weight) {
	KI_BHG_FILL(iret,weight);
	return true;
}

bool BeamHaloSource::call_ki_bhg_stat(int& iret) {
	KI_BHG_STAT(iret);
	return true;
}





