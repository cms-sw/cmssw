/*
  Copyright   : The FASTMC and SPHMC Collaboration
  Author      : Ionut Cristian Arsene 
  Affiliation : Oslo University, Norway & Institute for Space Sciences, Bucharest, Romania
  e-mail      : i.c.arsene@fys.uio.no
  Date        : 2007/05/30

  This class is using the particle and decays lists provided by the 
  THERMINATOR (Computer Physics Communications 174 669 (2006)) and
  SHARE (Computer Physics Communications 167 229 (2005)) collaborations.
*/

#ifndef DECAY_CHANNEL
#define DECAY_CHANNEL

#include "Rtypes.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

const int kMaxDaughters = 3;
const int kNonsensePDG = 1000000000;

class DecayChannel {
 private:
  int    fMotherPDG;
  double fBranchingRatio;
  int    fNDaughters;
  int    fDaughtersPDG[kMaxDaughters];
  
 public:
  DecayChannel();                                                                           // default constructor
  DecayChannel(const DecayChannel &copy);                                                   // copy constructor
  DecayChannel(int mother, double branching, int nDaughters, int *daughters);       // explicit constructor
  ~DecayChannel() {};                                                                       // destructor
  
  void     SetMotherPDG(int value)              {fMotherPDG = value;}
  void     SetBranching(double value)           {fBranchingRatio = value;}
  void     SetDaughters(int *values, int n);
  void     AddDaughter(int pdg);
  int    GetMotherPDG()                         {return fMotherPDG;}
  double GetBranching()                         {return fBranchingRatio;}
  int    GetNDaughters()                        {return fNDaughters;}
  int*   GetDaughters()                         {return fDaughtersPDG;}
  int    GetDaughterPDG(int i);                                                         // i --> must be the zero-based index of daughter
};

#endif
