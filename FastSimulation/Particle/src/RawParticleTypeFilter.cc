// -----------------------------------------------------------------------------
//  Prototype for a particle class
// -----------------------------------------------------------------------------
//  $Date: 2006/03/10 21:23:56 $
//  $Revision: 1.1 $
// -----------------------------------------------------------------------------
//  Author: Stephan Wynhoff - RWTH-Aachen (Email: Stephan.Wynhoff@cern.ch)
// -----------------------------------------------------------------------------
/** \file GeneratorInterface/Particle/src/RawParticle.cc
 * <A HREF=../../../../GeneratorInterface/Particle/src/RawParticle.cc>Source code</A>
 */

#include "FastSimulation/Particle/interface/RawParticleTypeFilter.h"

using namespace std;

RawParticleTypeFilter::RawParticleTypeFilter(const string& particleName) {
  HepLorentzVector one;
  RawParticle tmp(particleName,one);
//    cout << tmp.pid() << endl;
  myAcceptIDs.push_back(tmp.pid());
}

RawParticleTypeFilter::RawParticleTypeFilter(const string& particleName1, 
					     const string& particleName2) {
  HepLorentzVector one;
  RawParticle tmp1(particleName1,one),tmp2(particleName2,one);
//    cout << tmp1.pid() << endl;
//    cout << tmp2.pid() << endl;
  myAcceptIDs.push_back(tmp1.pid());
  myAcceptIDs.push_back(tmp2.pid());
}

RawParticleTypeFilter::RawParticleTypeFilter(const int pid) {
  myAcceptIDs.push_back(pid);
}

RawParticleTypeFilter::RawParticleTypeFilter(const int pid1, const int pid2) {
  myAcceptIDs.push_back(pid1);
  myAcceptIDs.push_back(pid2);
}

void RawParticleTypeFilter::addAccept(const int pid) {
  myAcceptIDs.push_back(pid);
  myRejectIDs.clear();
}

void RawParticleTypeFilter::addAccept(const string& name) {
  HepLorentzVector one;
  RawParticle tmp(name,one);
  this->addAccept(tmp.pid());
}

void RawParticleTypeFilter::addReject(const int pid) {
  myRejectIDs.push_back(pid);
  myAcceptIDs.clear();
}

void RawParticleTypeFilter::addReject(const string& name) {
  HepLorentzVector one;
  RawParticle tmp(name,one);
  this->addReject(tmp.pid());
}

bool RawParticleTypeFilter::isOKForMe(const RawParticle *p) const
{
  bool acceptThis = true;
  if (myAcceptIDs.size() > 0) {
    acceptThis = this->isAcceptable(p->pid());
  } 
  if (myRejectIDs.size() > 0) {
    acceptThis = ! this->isRejectable(p->pid());
  }
  return acceptThis;
}

bool RawParticleTypeFilter::isAcceptable(const int id) const
{
  bool acceptThis = false;

  vector<int>::const_iterator myAcceptIDsItr;
  for (myAcceptIDsItr = myAcceptIDs.begin(); myAcceptIDsItr != myAcceptIDs.end();
       myAcceptIDsItr++) {
    if (id == (*myAcceptIDsItr)) {
      acceptThis = true;
    }
  }
  return acceptThis;
}

bool RawParticleTypeFilter::isRejectable(const int id) const
{
  bool acceptThis = false;

  vector<int>::const_iterator myRejectIDsItr;
  for (myRejectIDsItr = myRejectIDs.begin(); myRejectIDsItr != myRejectIDs.end();
       myRejectIDsItr++) {
    if (id == (*myRejectIDsItr)) {
      acceptThis = true;
    }
  }
  return acceptThis;
}
