// This will force the symbols below to be kept, even in the case tauola 
// is an archive library.
#include "GeneratorInterface/TauolaInterface/interface/TauolaWrapper.h"

extern "C" void phyfix_(void );
extern "C" void spinhiggs_(void );
extern "C" void taupi0_(void);
extern "C" void inietc_(void);
extern "C" void inimas_(void);
extern "C" void iniphx_(void);
extern "C" void initdk_(void);
extern "C" void taupi0_(void);
extern "C" void dekay_(void);
extern "C" void plzapx_(void);
__attribute__((visibility("hidden"))) void dummy()
{
  float dummyFloat = 0;
  int dummyInt = 0;
  dexay_(&dummyInt, &dummyFloat);
  phyfix_();
  spinhiggs_();
  taupi0_();
  taupi0_();
  inietc_();
  inimas_();
  iniphx_();
  initdk_();
  taupi0_();
  dekay_();
  plzapx_();
}
