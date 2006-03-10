// -----------------------------------------------------------------------------
//  Prototype for a particle class
// -----------------------------------------------------------------------------
//  $Date: 2004/02/03 14:24:29 $
//  $Revision: 1.11 $
// -----------------------------------------------------------------------------
//  Author: Stephan Wynhoff - RWTH-Aachen (Email: Stephan.Wynhoff@cern.ch)
// -----------------------------------------------------------------------------
#include "CLHEP/HepMC/GenVertex.h"
/** \file GeneratorInterface/Particle/src/RawParticle.cc */
//#include "GeneratorInterface/HepPDT/interface/HepPDT.h"
//#include "GeneratorInterface/HepPDT/interface/HepPDTable.h"
//#include "GeneratorInterface/HepPDT/interface/HepDecayMode.h"
//#include "GeneratorInterface/HepPDT/interface/HepJetsetDummyHandler.h"
//#include "GeneratorInterface/HepPDT/interface/HepParticleData.h"
#include "FastSimulation/Particle/interface/RawParticle.h"
//#include "Utilities/GenUtil/interface/ConfigurationDictionary.h"
//#include "Utilities/GenUtil/interface/FileInPath.h"
//#include "Utilities/GenUtil/interface/pidInfo.h"
//#include "Utilities/GenUtil/interface/CMSexception.h"

#include <fstream>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <algorithm>

// init static members
HepPDTable * RawParticle::tab = 0;
int RawParticle::nParticles = 0;
bool RawParticle::isfirst = true;

using namespace std;
using namespace HepMC;

RawParticle::RawParticle() {
  init();
  //  cout << "Create RawParticle as default" << nParticles << endl;
}

RawParticle::RawParticle(HepMC::GenParticle *p) : 
  HepLorentzVector(p->momentum()) {
  init();
  myId = p->pdg_id();
  myVertex = p->production_vertex()->position();
  myStatus = p->status();
  //  myCharge = ??;
  //  myMass   = ??;
}
  

RawParticle::RawParticle(HepLorentzVector p) 
  : HepLorentzVector(p) {
  init();
  //  cout << "Create RawParticle from LV" << nParticles << endl;
}

RawParticle::RawParticle(const int id, const HepLorentzVector p) 
  : HepLorentzVector(p) {
  this->init();
  this->setID(id);
  //  cout << "Create RawParticle from id,LV" << nParticles << endl;
}

RawParticle::RawParticle(const std::string name, const HepLorentzVector p) 
  : HepLorentzVector(p) {
  this->init();
  this->setID(name);
  //  cout << "Create RawParticle from id,LV" << nParticles << endl;
}

RawParticle::RawParticle(HepLorentzVector p, HepLorentzVector xStart) {
  init();
  this->setPx(p.px());
  this->setPy(p.py());
  this->setPz(p.pz());
  this->setE(p.e());
  myVertex = xStart;
  //    cout << this->vertex() << endl;
  //    { pidInfo a;}
  //  cout << "Create RawParticle from 2 LVs" << nParticles << endl;
}

RawParticle::RawParticle(HepDouble px, HepDouble py, HepDouble pz, HepDouble e) {
  init();
  this->setPx(px);
  this->setPy(py);
  this->setPz(pz);
  this->setE(e);
  //  cout << "Create RawParticle from px,py,pz,e " << nParticles << endl;
}

RawParticle::RawParticle(const RawParticle &right) : HepLorentzVector() {
  this->setPx(right.px());
  this->setPy(right.py());
  this->setPz(right.pz());
  this->setE(right.e());
  myId     = right.myId; 
  myStatus = right.myStatus;
  myUsed   = right.myUsed;
  myCharge = right.myCharge;
  myMass   = right.myMass;
  myVertex = (right.myVertex);
  //  cout << "Copy construct RawParticle " << nParticles << endl;
}

RawParticle::~RawParticle() {
  //    cout << "Delete RawParticle " << nParticles << endl;
  //    { pidInfo a;}
  //    this->print();
  

  nParticles--;
  //    { pidInfo a;}
}

RawParticle&  RawParticle::operator = (const RawParticle & right ) {
  //  cout << "Copy assignment " << endl;
  if (this != &right) { // don't copy into yourself
    this->setPx(right.px());
    this->setPy(right.py());
    this->setPz(right.pz());
    this->setE(right.e());
    myId     = right.myId; 
    myStatus = right.myStatus;
    myUsed   = right.myUsed;
    myCharge = right.myCharge;
    myMass   = right.myMass;
    myVertex = right.myVertex;
  }
  return *this;
}

void RawParticle::init() {
  myId=0;  
  myStatus=99;
  myUsed=0;
  myCharge=0.;
  myMass=0.;
  if (isfirst) {
    //    tab = & HepPDT::theTable();
    isfirst=false;
  }
  nParticles++;
}

void RawParticle::setID(const int id) {
  myId = id;
}

void RawParticle::setID(const std::string name) {
  //  if (tab->getParticleData(name) != 0) {
  //  myId = (tab->getParticleData(name))->id();
  //  } else {
    myId = 0;
  //}
}

void RawParticle::setStatus(int istat) {
  myStatus = istat;
}

void RawParticle::setMass(float m) {
  myMass = m;
}

void RawParticle::setCharge(float q) {
  myCharge = q;
}

void RawParticle::chargeConjugate() {
  //  if ((tab->getParticleData(myId))->CC()) {
  //    myId = (tab->getParticleData(myId))->CC()->id();
    myCharge = -1*myCharge;
  //  }
}

void RawParticle::setT(const double t) {
    myVertex.setT(t);
}

void RawParticle::setVertex(const HepLorentzVector vtx) {
  myVertex = vtx;
  //  cout << this->vertex() << endl;
  //    { pidInfo a;}
}

HepLorentzVector  RawParticle::vertex() const { 
    return   myVertex ; 
}

void RawParticle::rotate(HepDouble rphi, const Hep3Vector &raxis) {
  this->rotate(rphi, raxis);
  myVertex.rotate(rphi, raxis);
}

void RawParticle::rotateX(HepDouble rphi) {
  this->rotateX(rphi);
  myVertex.rotateX(rphi);
}

void RawParticle::rotateY(HepDouble rphi) {
  this->rotateY(rphi);
  myVertex.rotateY(rphi);
}

void RawParticle::rotateZ(HepDouble rphi) {
  this->rotateZ(rphi);
  myVertex.rotateZ(rphi);
}

std::string RawParticle::PDGname() const {
  std::string MyParticleName;
  //  HepString MyParticleName;
  //  if (tab->getParticleData(myId) != 0) {
  //    MyParticleName = (tab->getParticleData(myId))->name();
  //  } else {
  //    MyParticleName = "none";
  //  }
  return (std::string) MyParticleName;}


void RawParticle::printName() const {
  std::string MyParticleName;
  //  CLHep::HepString MyParticleName;
  //  if (tab->getParticleData(myId) != 0) {
  //    MyParticleName = (tab->getParticleData(myId))->name();
  //  }  if (MyParticleName.length() != 0) {
  //   cout <<  MyParticleName;
  //    for(unsigned int k=0;k<9-MyParticleName.length() && k<10; k++) 
  //      cout << " " ;
  //  }
  //  else
  //    cout << "unknown  ";
}

void RawParticle::print() const {
  printName();
  cout << setw(3) << status();
  cout.setf(ios::fixed, ios::floatfield);
  cout.setf(ios::right, ios::adjustfield);
  cout << setw(8) << setprecision(2) << px();
  cout << setw(8) << setprecision(2) << py();
  cout << setw(8) << setprecision(2) << pz();
  cout << setw(8) << setprecision(2) << e();
  cout << setw(8) << setprecision(2) << m();
  cout << setw(8) << setprecision(2) << mass();
  cout << setw(8) << setprecision(2) << charge();
  cout << setw(8) << setprecision(2) << x();
  cout << setw(8) << setprecision(2) << y();
  cout << setw(8) << setprecision(2) << z();
  cout << setw(8) << setprecision(2) << t();
  cout << setw(0) << endl;
}

ostream& operator <<(ostream& o , const RawParticle& p) {

  o << p.pid() << " (";
  o << p.status() << "): ";
  o << (HepLorentzVector) p << " ";
  o << p.vertex();
  return o;

} 

int RawParticle::ntot() const {
  return nParticles;
}

HepDouble RawParticle::PDGcharge() const { 
  HepDouble q=-99999;
  //  if (tab->getParticleData(myId)) {
  //    q=tab->getParticleData(myId)->charge();
  //  }
  return q;
}

HepDouble RawParticle::PDGmass() const  { 
  HepDouble m=-99999;
  //  if (tab->getParticleData(myId)) {
  //    m=tab->getParticleData(myId)->mass();
  //  }
  return m;
}

HepDouble RawParticle::PDGcTau() const { 
  HepDouble ct=-99999;
  //  if (tab->getParticleData(myId)) {
  //    ct=tab->getParticleData(myId)->cTau();
  //  }
  return ct;
}


HepDouble RawParticle::et() const { 
  double mypp, tmpEt=-1.;
  
  mypp = sqrt(this->px()*this->px() + this->py()*this->py() + this->pz()*this->pz());
  if (mypp != 0) {
    tmpEt=this->e() * this->perp()/mypp;
  }
  return tmpEt;
}
