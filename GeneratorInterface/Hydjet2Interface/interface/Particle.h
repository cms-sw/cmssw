/*                                                                            
                                                                            
Nikolai Amelin, Ludmila Malinina, Timur Pocheptsov (C) JINR/Dubna
amelin@sunhe.jinr.ru, malinina@sunhe.jinr.ru, pocheptsov@sunhe.jinr.ru 
November. 2, 2005                                

*/

#ifndef PARTICLE_INCLUDED
#define PARTICLE_INCLUDED

#include <list>

#include <TLorentzRotation.h>
#include <TLorentzVector.h>

#include <TVector3.h>
#include "ParticlePDG.h"
#include <iostream>


class Particle {
 protected:
  TLorentzVector   fPosition;
  TLorentzVector   fMomentum;
  TLorentzVector   fLastMotherDecayCoor;
  TLorentzVector   fLastMotherDecayMom;
  ParticlePDG     *fParticleProperties;
  double         fLastInteractionTime;
  int            fInteractionNumber;
  int            fPythiaStatusCode;
  int            fLastMotherPdg;
  int            fType; //0-hydro, 1-jets
  int            fIndex;                    // index (0 based) of particle in the final particle list which will contain both primaries and secondaries
  int            fMotherIndex;              // index of the mother (-1 if its a primary particle)
  int            fNDaughters;               // number of daughter particles (0 if the particle had not decayed)
  int            fFirstDaughterIndex;       // index for the first daughter particle (-1 if non-existing)
  int            fLastDaughterIndex;        // index for the last daughter particle (-1 if non-existing)
  static int     fLastIndex;                // the last index assigned
  bool           fDecayed;                  // true if the decay procedure already applied

 public:
  Particle(ParticlePDG *pdg = 0);
  Particle(ParticlePDG *pdg, const TLorentzVector &pos, const TLorentzVector &mom,
	   double lastInterTime = 0., int lastInterNum = 0, int type=0);
  Particle(ParticlePDG *pdg, const TLorentzVector &pos, const TLorentzVector &mom,
	   double lastInterTime, int lastInterNum, int type, int motherPdg, 
	   const TLorentzVector &motherPos, const TLorentzVector &motherMom);

  double X()const{return fPosition.X();}
  double X(double val){fPosition.SetX(val); return val;}
  double Y()const{return fPosition.Y();}
  double Y(double val){fPosition.SetY(val); return val;}
  double Z()const{return fPosition.Z();}
  double Z(double val){fPosition.SetZ(val); return val;}
  double T()const{return fPosition.T();}
  double T(double val){fPosition.SetT(val); return val;}
  double Px()const{return fMomentum.Px();}
  double Px(double val){fMomentum.SetPx(val); return val;}
  double Py()const{return fMomentum.Py();}
  double Py(double val){fMomentum.SetPy(val); return val;}
  double Pz()const{return fMomentum.Pz();}
  double Pz(double val){fMomentum.SetPz(val); return val;}
  double E()const{return fMomentum.E();}
  double E(double val){fMomentum.SetE(val); return val;}

  TLorentzVector &Pos(){return fPosition;}
  const TLorentzVector &Pos()const{return fPosition;}
  TLorentzVector &Pos(const TLorentzVector &val){return fPosition = val;}
  TLorentzVector &Mom(){return fMomentum;}
  const TLorentzVector &Mom()const{return fMomentum;}
  TLorentzVector &Mom(const TLorentzVector &val){return fMomentum = val;}

  void SetDecayed() {fDecayed = kTRUE;}
  bool GetDecayed() const {return fDecayed;}

  void Boost(const TVector3 &val){fMomentum.Boost(val);}
  void Boost(const TLorentzVector &val){fMomentum.Boost(val.BoostVector());}
  void TransformMomentum(const TRotation &rotator){fMomentum *= rotator;}
  void TransformPosition(const TRotation &rotator){fPosition *= rotator;}
  void Shift(const TVector3 &val){fPosition += TLorentzVector(val, 0.);}

  //Pseudorapidity
  double Eta ()const;
  //Rapidity
  double Rapidity()const;
  double Phi()const;
  double Theta()const;
  double Pt()const;

  int Encoding() const;
  double TableMass() const;
  ParticlePDG *Def() const {return fParticleProperties;}
  ParticlePDG *Def(ParticlePDG *newProp) {return fParticleProperties = newProp;}
  //mother   
  void SetLastMotherPdg(int value){fLastMotherPdg = value;}
  int GetLastMotherPdg() const {return fLastMotherPdg;}

  // aic(2008/08/08): functions added in order to enable tracking of mother/daughter particles by a unique index
  // The index coincides with the position of the particle in the secondaries list.
  int SetIndex() {fIndex = ++fLastIndex; return fIndex;}
  int GetIndex() {return fIndex;}
  static int GetLastIndex() {return fLastIndex;}
  static void InitIndexing() {
    fLastIndex = -1;
  }
  void SetMother(int value) {fMotherIndex = value;}
  int GetMother() {return fMotherIndex;}
  void SetFirstDaughterIndex(int index) {fFirstDaughterIndex = index;}
  void SetLastDaughterIndex(int index) {fLastDaughterIndex = index;}
  void SetPythiaStatusCode(int code) {fPythiaStatusCode = code;}
  int GetPythiaStatusCode() {return fPythiaStatusCode;}
  int GetNDaughters() {
    if(fFirstDaughterIndex==-1 || fLastDaughterIndex==-1) 
      return 0;
    else
      return fLastDaughterIndex-fFirstDaughterIndex+1;
  }
  int GetFirstDaughterIndex() {return fFirstDaughterIndex;}
  int GetLastDaughterIndex() {return fLastDaughterIndex;}
  
  TLorentzVector &SetLastMotherDecayCoor(const TLorentzVector &val){return fLastMotherDecayCoor = val;}
  const TLorentzVector &GetLastMotherDecayCoor()const{return fLastMotherDecayCoor;}
  TLorentzVector &SetLastMotherDecayMom(const TLorentzVector &val){return fLastMotherDecayMom = val;}
  const TLorentzVector &GetLastMotherDecayMom()const{return fLastMotherDecayMom;}

  void SetLastInterTime(double value){fLastInteractionTime = value;}
  double GetLastInterTime()const{return fLastInteractionTime;}
  void SetLastInterNumber(int value){fInteractionNumber = value;}
  int GetLastInterNumber()const{return fInteractionNumber;}
  void IncInter(){++fInteractionNumber;}

  void SetType(int value){fType = value;}
  int GetType()const{return fType;}


};

double S(const TLorentzVector &, const TLorentzVector &);
double T(const TLorentzVector &, const TLorentzVector &);

typedef std::list<Particle> List_t;
typedef std::list<Particle>::iterator LPIT_t;

class ParticleAllocator {
 public:
  void AddParticle(const Particle & particle, List_t & list);
  void FreeListNode(List_t & list, LPIT_t it);
  void FreeList(List_t & list);

 private:
  List_t fFreeNodes;
};

#endif
