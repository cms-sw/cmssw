/*                                                                            
                                                                            
Nikolai Amelin, Ludmila Malinina, Timur Pocheptsov (C) JINR/Dubna
amelin@sunhe.jinr.ru, malinina@sunhe.jinr.ru, pocheptsov@sunhe.jinr.ru 
November. 2, 2005                                

*/

#ifndef PARTICLETABLE_INCLUDED
#define PARTICLETABLE_INCLUDED

#include <map>
#include <Rtypes.h>

struct ParticleInfo {
  int fBaryonNumber;
  int fStrangeness;
  int fIsospin;
  int fSpin;
  int fCharge;

  ParticleInfo(int bN, int s, int s1, int s2, int c) {
    fBaryonNumber = bN;
    fStrangeness = s;
    fIsospin = s1; //2S
    fSpin = s2; //2I
    fCharge = c; //fCharge = 2 * I3
  }
};

extern const std::map<const int, ParticleInfo> gParticleTable;
typedef std::map<const int, ParticleInfo>::const_iterator MapIt_t;

#endif
