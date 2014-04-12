#include "FastSimulation/ParticleDecay/interface/Pythia6jets.h"

#define PYJETS pyjets_
extern "C" void *getaddr(void *item)
{
        return item;
}

extern 
struct _Pyjets {
	int n;
	int npad;
	int k[5][4000];
	double p[5][4000];
	double v[5][4000];
} PYJETS;

struct Pythia6jets::_pythia6jets * Pythia6jets::__pythia6jets = 0;

double Pythia6jets::dDummy = 0.0;
int Pythia6jets::nDummy = 0;

Pythia6jets::Pythia6jets(void)
{
}

Pythia6jets::~Pythia6jets(void)
{
}

void Pythia6jets::init(void)
{
  __pythia6jets = 
    static_cast<struct Pythia6jets::_pythia6jets *>(getaddr(&PYJETS.n));
}

int &Pythia6jets::n(void)
{
  if (__pythia6jets == 0) init();
  return __pythia6jets->n;
}

int &Pythia6jets::npad(void)
{
  if (__pythia6jets == 0) init();
  return __pythia6jets->npad;
}

int &Pythia6jets::k(int i,int j)
{
  if (__pythia6jets == 0) init();
  if ((i<1)||(i>_depth)||
      (j<1)||(j>_length))
    {
      nDummy = -999;
      return nDummy;
    }
  return __pythia6jets->k[j-1][i-1];
}

double &Pythia6jets::p(int i,int j)
{
  if (__pythia6jets == 0) init();
  if ((i<1)||(i>_depth)||
      (j<1)||(j>_length))
    {
      dDummy = -999.0;
      return dDummy;
    }
  return __pythia6jets->p[j-1][i-1];
}

double &Pythia6jets::v(int i,int j)
{
  if (__pythia6jets == 0) init();
  if ((i<1)||(i>_depth)||
      (j<1)||(j>_length))
    {
      dDummy = -999.0;
      return dDummy;
    }
  return __pythia6jets->v[j-1][i-1];
}
