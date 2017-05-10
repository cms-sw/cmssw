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

#ifndef PARTICLE_PDG
#define PARTICLE_PDG

#include "Rtypes.h"

#ifndef DECAY_CHANNEL
#include "DecayChannel.h"
#endif

const int kMaxDecayChannels = 100;

class ParticlePDG {
 private:
  char        fName[9];
  int         fPDG;
  double      fMass;
  double      fWidth;
  double      fSpin;                         // J
  double      fIsospin;                      // I
  double      fIsospinZ;                     // I3
  double      fLightQuarkNumber;             // u, d quark number
  double      fAntiLightQuarkNumber;         // u-, d- quark number
  double      fStrangeQuarkNumber;           // s quark number
  double      fAntiStrangeQuarkNumber;       // s- quark number
  double      fCharmQuarkNumber;             // c quark number
  double      fAntiCharmQuarkNumber;         // c- quark number
  int         fNDecayChannels;
  DecayChannel* fDecayChannels[kMaxDecayChannels];
   
 public:
  ParticlePDG();
  ParticlePDG(char* name, int pdg, double mass, double width);
  ~ParticlePDG();
  
  void AddChannel(DecayChannel &channel);
  void SetName(char* name) {
    for(int i=0; i<9; i++)
      if(*(name+i) != '\0') fName[i] = *(name+i);
      else break;
  }
  void SetPDG(int value) {fPDG = value;}
  void SetMass(double value) {fMass = value;}
  void SetWidth(double value) {fWidth = value;}
  void SetSpin(double value) {fSpin = value;}
  void SetIsospin(double value) {fIsospin = value;}
  void SetIsospinZ(double value) {fIsospinZ = value;}
  void SetLightQNumber(double value) {fLightQuarkNumber = value;}
  void SetLightAQNumber(double value) {fAntiLightQuarkNumber = value;}
  void SetStrangeQNumber(double value) {fStrangeQuarkNumber = value;}
  void SetStrangeAQNumber(double value) {fAntiStrangeQuarkNumber = value;}
  void SetCharmQNumber(double value) {fCharmQuarkNumber = value;}
  void SetCharmAQNumber(double value) {fAntiCharmQuarkNumber = value;}
  
  char* GetName() {return fName;}
  int GetPDG() {return fPDG;}
  double GetMass() {return fMass;}
  double GetWidth() {return fWidth;}
  int GetNDecayChannels() {return fNDecayChannels;}
  double GetSpin() {return fSpin;}
  double GetIsospin() {return fIsospin;}
  double GetIsospinZ() {return fIsospinZ;}
  double GetLightQNumber() {return fLightQuarkNumber;}
  double GetLightAQNumber() {return fAntiLightQuarkNumber;}
  double GetStrangeQNumber() {return fStrangeQuarkNumber;}
  double GetStrangeAQNumber() {return fAntiStrangeQuarkNumber;}
  double GetCharmQNumber() {return fCharmQuarkNumber;}
  double GetCharmAQNumber() {return fAntiCharmQuarkNumber;}
  double GetBaryonNumber() {return (fLightQuarkNumber     + fStrangeQuarkNumber     + fCharmQuarkNumber -  
                                    fAntiLightQuarkNumber - fAntiStrangeQuarkNumber - fAntiCharmQuarkNumber)/3.;}
  double GetStrangeness() {return (fAntiStrangeQuarkNumber - fStrangeQuarkNumber);}
  double GetCharmness() {return (fCharmQuarkNumber - fAntiCharmQuarkNumber);}
  double GetElectricCharge() {return fIsospinZ + (GetBaryonNumber()+GetStrangeness()+GetCharmness())/2.;}
  
  double GetFullBranching();
  DecayChannel* GetDecayChannel(int i) {
    if(0<=i && i<fNDecayChannels) 
      return fDecayChannels[i];
    else
      return 0x0;
  }
};

#endif
