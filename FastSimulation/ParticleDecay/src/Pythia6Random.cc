#include "FastSimulation/ParticleDecay/interface/Pythia6Random.h"

#include <string>
#include <iostream>

#define PYDATR pydatr_
extern "C" void *getAddr(void *item)
{
        return item;
}

extern 
struct {
	int mrpy[6];
	double rrpy[100];
} PYDATR;

struct Pythia6Random::_pythia6random * Pythia6Random::__pythia6random = 0;

double Pythia6Random::dDummy = 0.0;
int Pythia6Random::nDummy = 0;

Pythia6Random::Pythia6Random(int seed) {

  myPythia6Random[0] = new _pythia6random;
  myPythia6Random[1] = new _pythia6random;

  // Initialize current state of the decay random generation
  myPythia6Random[1]->mrpy[0] = seed;
  myPythia6Random[1]->mrpy[1] = 0;
  myPythia6Random[1]->mrpy[2] = 0;
  myPythia6Random[1]->mrpy[3] = 0;
  myPythia6Random[1]->mrpy[4] = 0;
  myPythia6Random[1]->mrpy[5] = 0;
  for ( int j=0; j<100; ++j ) myPythia6Random[1]->rrpy[j] = 0.;

}

Pythia6Random::~Pythia6Random(void) {

  delete myPythia6Random[0];
  delete myPythia6Random[1];

}

void 
Pythia6Random::init(void)
{
  __pythia6random = 
    static_cast<struct Pythia6Random::_pythia6random *>(getAddr(&PYDATR.mrpy[0]));
}

int&
Pythia6Random::mrpy(int i)
{
  if (__pythia6random == 0) init();
  if ( i<1 || i>m_length ) {
    nDummy = -999;
    return nDummy;
  }
  return __pythia6random->mrpy[i-1];
}

double&
Pythia6Random::rrpy(int i)
{
  if (__pythia6random == 0) init();
  if ( i<1 || i>r_length ) {
    dDummy = -999.0;
    return dDummy;
  }
  return __pythia6random->rrpy[i-1];
}

void
Pythia6Random::save(int i) {

  for ( int j=1; j<7; ++j ) {
    myPythia6Random[i]->mrpy[j-1] =  mrpy(j);
    //    std::cout << "Save " << i << " mrpy(" << j << ") = " << mrpy(j) << std::endl;
  }
  for ( int j=1; j<101; ++j ) {
    myPythia6Random[i]->rrpy[j-1] =  rrpy(j);
    //    std::cout << "Save " << i << " rrpy(" << j << ") = " << rrpy(j) << std::endl;
  }

}

void
Pythia6Random::get(int i) {

  for ( int j=1; j<7; ++j ) {
    mrpy(j) = myPythia6Random[i]->mrpy[j-1];
    //    std::cout << "Get " << i << " mrpy(" << j << ") = " << mrpy(j) << std::endl;
  }
  for ( int j=1; j<101; ++j ) { 
    rrpy(j) = myPythia6Random[i]->rrpy[j-1];
    //    std::cout << "Get " << i << " rrpy(" << j << ") = " << rrpy(j) << std::endl;
  }

}
