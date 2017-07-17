/*
  Copyright   : The FASTMC and SPHMC Collaboration
  Author      : Ionut Cristian Arsene 
  Affiliation : Oslo University, Norway & Institute for Space Sciences, Bucharest, Romania
  e-mail      : i.c.arsene@fys.uio.no
  Date        : 2007/05/30
  
  This class is using the particle and decay lists provided by the 
  THERMINATOR (Computer Physics Communications 174 669 (2006)) and
  SHARE (Computer Physics Communications 167 229 (2005)) collaborations.
*/

#ifndef DECAY_CHANNEL
#include "GeneratorInterface/Hydjet2Interface/interface/DecayChannel.h"
#endif
#include <iostream>

using namespace std;

DecayChannel::DecayChannel() {
  fMotherPDG = kNonsensePDG;
  fBranchingRatio = 0.0;
  fNDaughters = 0;
  for(int i=0; i<kMaxDaughters; i++)
    fDaughtersPDG[i] = kNonsensePDG;
}
          
DecayChannel::DecayChannel(const DecayChannel &copy) {
  fMotherPDG = copy.fMotherPDG;
  fBranchingRatio = copy.fBranchingRatio;
  fNDaughters = copy.fNDaughters;
  for(int i=0; i<fNDaughters; i++)
    fDaughtersPDG[i] = copy.fDaughtersPDG[i];
}
                    
DecayChannel::DecayChannel(int mother, double branching, int nDaughters, int *daughters) {
  fMotherPDG = mother;
  fBranchingRatio = branching;
  fNDaughters = 0;
  for(int i=0; i<nDaughters; i++) {
    if(i >= kMaxDaughters) {
      edm::LogError("DecayChannel")<<"From explicit constructor: Number of daughters bigger than the maximum allowed one (" << kMaxDaughters << ") !!";
    }
    fDaughtersPDG[fNDaughters++] = *(daughters+i);
  }
}
                              
void DecayChannel::SetDaughters(int *daughters, int n) {
  for(int i=0; i<n; i++) {
    if(i >= kMaxDaughters) {
      edm::LogError("DecayChannel")<<"From SetDaughters(): Number of daughters bigger than the maximum allowed one (" << kMaxDaughters << ") !!";
    }
    fDaughtersPDG[fNDaughters++] = *(daughters+i);
  }
}
                                  
void DecayChannel::AddDaughter(int pdg) {
  if(fNDaughters >= kMaxDaughters) {
    edm::LogError("DecayChannel")<<"From AddDaughter(): Number of daughters is already >= than the maximum allowed one (" << kMaxDaughters << ") !!";
  }
  fDaughtersPDG[fNDaughters++] = pdg;
}
                                        
int DecayChannel::GetDaughterPDG(int i) {
  if((i >= fNDaughters) || (i<0)) {
    edm::LogError("DecayChannel")<<"From GetDaughterPDG(): Daughter index required is too big or less than zero!! There are only " << fNDaughters << " secondaries in this channel !!";
    return kNonsensePDG;
  }
  return fDaughtersPDG[i];
}

