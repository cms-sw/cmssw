/*                                                                           
                                                                            
Nikolai Amelin, Ludmila Malinina, Timur Pocheptsov (C) JINR/Dubna
amelin@sunhe.jinr.ru, malinina@sunhe.jinr.ru, pocheptsov@sunhe.jinr.ru 
November. 2, 2005                                

*/

#include <TMath.h>
#include "GeneratorInterface/Hydjet2Interface/interface/ParticlePDG.h"
#include "GeneratorInterface/Hydjet2Interface/interface/Particle.h"

Particle::Particle(ParticlePDG *prop)
  : fParticleProperties(prop),
    fLastInteractionTime(0.),
    fInteractionNumber(0),
    fPythiaStatusCode(-1),
    fLastMotherPdg(0),
    fType(0),
    fIndex(-1),
    fMotherIndex(-1),
    fNDaughters(0),
    fFirstDaughterIndex(-1),
    fLastDaughterIndex(-1),
    fDecayed(kFALSE)
{}

Particle::Particle(ParticlePDG *prop, const TLorentzVector &pos, 
		   const TLorentzVector &mom, double lit, int lin, int type)
  : fPosition(pos), 
    fMomentum(mom), 
    fPythiaStatusCode(-1),
    fIndex(-1), 
    fMotherIndex(-1), 
    fNDaughters(0), 
    fFirstDaughterIndex(-1),
    fLastDaughterIndex(-1),
    fDecayed(kFALSE)
{
  fLastInteractionTime = lit;
  fInteractionNumber = lin;
  fParticleProperties = prop;
  fType =  type;
}

Particle::Particle(ParticlePDG *prop, const TLorentzVector &pos, const TLorentzVector &mom,
                   double t, int n, int ty, int motherPdg, const TLorentzVector &mPos, 
		   const TLorentzVector &mMom)
  : fPosition(pos), fMomentum(mom),
    fLastMotherDecayCoor(mPos),
    fLastMotherDecayMom(mMom),
    fPythiaStatusCode(-1),
    fIndex(-1), 
    fMotherIndex(-1), 
    fNDaughters(0),
    fFirstDaughterIndex(-1),
    fLastDaughterIndex(-1),
    fDecayed(kFALSE)
{
  fParticleProperties = prop;
  fLastInteractionTime = t;
  fInteractionNumber = n;
  fType = ty;
  fLastMotherPdg = motherPdg;
}

int Particle::Encoding() const {
  return fParticleProperties->GetPDG();
}

double Particle::TableMass() const {
  return fParticleProperties->GetMass();
}

double Particle::Eta() const {
  if(fMomentum.P() != fMomentum.Pz())
    return 0.5 * TMath::Log((fMomentum.P() + fMomentum.Pz()) / (fMomentum.P()-fMomentum.Pz()));
  else return 1.e30;
}

double Particle::Rapidity() const {
  if (fMomentum.E() != fMomentum.Pz())
    return 0.5 * TMath::Log((fMomentum.E() + fMomentum.Pz()) / (fMomentum.E() - fMomentum.Pz()));
  else return 1.e30;
}

double Particle::Phi() const {
  return TMath::Pi()+TMath::ATan2(-fMomentum.Py(), -fMomentum.Px());
}

double Particle::Theta() const {
  return !fMomentum.Pz() ? TMath::Pi() / 2 : TMath::ACos(fMomentum.Pz() / fMomentum.P());
}

double Particle::Pt() const {
  return TMath::Sqrt(fMomentum.Px() * fMomentum.Px() + fMomentum.Py() * fMomentum.Py());
}

double S(const TLorentzVector &v1, const TLorentzVector &v2) {
  return TMath::Power(v1.T() + v2.T(), 2) - TMath::Power(v1.X() + v2.X(), 2) -
    TMath::Power(v1.Y() + v2.Y(), 2) - TMath::Power(v1.Z() + v2.Z(), 2);
}

double T(const TLorentzVector & v1, const TLorentzVector & v2) {
  return TMath::Power(v1.T() - v2.T(), 2) - TMath::Power(v1.X() - v2.X(), 2) - 
    TMath::Power(v1.Y() - v2.Y(), 2) - TMath::Power(v1.Z() - v2.Z(), 2);
}

void ParticleAllocator::AddParticle(const Particle & p, List_t &list) {
  if(fFreeNodes.empty())
    list.push_back(p);
  else {
    list.splice(list.end(), fFreeNodes, fFreeNodes.begin());
    list.back() = p;
  }
}

void ParticleAllocator::FreeListNode(List_t & list, LPIT_t it) {
  fFreeNodes.splice(fFreeNodes.end(), list, it);      
}

void ParticleAllocator::FreeList(List_t & list) {
  fFreeNodes.splice(fFreeNodes.end(), list);
}
