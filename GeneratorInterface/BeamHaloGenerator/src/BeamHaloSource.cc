
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

// #include "HepMC/ConvertHEPEVT.h"
// #include "HepMC/CBhepevt.h"
#include "HepMC/WeightContainer.h"


#define KI_BHG_INIT ki_bhg_init_
extern "C" {
  void KI_BHG_INIT(long& seed);
}

#define BHSETPARAM bhsetparam_
extern "C" {
   void BHSETPARAM(int* iparam, float* fparam);
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

 // -- from bhgctrl.inc
   iparam[0]  = pset.getUntrackedParameter<int>("GENMOD");
   iparam[1]  = pset.getUntrackedParameter<int>("LHC_B1");
   iparam[2]  = pset.getUntrackedParameter<int>("LHC_B2");
   iparam[3]  = pset.getUntrackedParameter<int>("IW_MUO");
   iparam[4]  = pset.getUntrackedParameter<int>("IW_HAD");

   iparam[5]  = pset.getUntrackedParameter<int>("shift_bx");

   
   fparam[0]  = (float)pset.getUntrackedParameter<double>("EG_MIN");
   fparam[1]  = (float)pset.getUntrackedParameter<double>("EG_MAX");

   fparam[2] = (float)pset.getUntrackedParameter<double>("BXNS");


    call_bh_set_parameters(iparam,fparam);


// -- Seed for randomnumbers
    Service<RandomNumberGenerator> rng;
    long seed = (long)(rng->mySeed());


// -- initialisationa
   call_ki_bhg_init(seed);


  produces<HepMCProduct>();
  cout << "BeamHaloSource: starting event generation ... " << endl;

}


void BeamHaloSource::clear()
{
}

bool BeamHaloSource::produce(Event & e) {

	// cout << "in produce " << endl;

    	auto_ptr<HepMCProduct> bare_product(new HepMCProduct());

	// cout << "apres autoptr " << endl;

	int iret=0;
        float weight = 0;
    	call_ki_bhg_fill(iret, weight);

	// cout << "apres fortran " << endl;


    	// HepMC::GenEvent* evt = conv.getGenEventfromHEPEVT();
	HepMC::GenEvent* evt = conv.read_next_event();

	HepMC::WeightContainer& weights = evt -> weights();
	weights.push_back(weight);

	if(evt)  bare_product->addHepMCData(evt );

    	e.put(bare_product);

    return true;
}



bool BeamHaloSource::call_bh_set_parameters(int* ival, float* fval) {
	BHSETPARAM(ival,fval);
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




