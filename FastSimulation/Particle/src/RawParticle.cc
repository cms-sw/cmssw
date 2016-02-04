// -----------------------------------------------------------------------------
//  Prototype for a particle class
// -----------------------------------------------------------------------------
//  $Date: 2007/10/01 09:05:25 $
//  $Revision: 1.14 $
// -----------------------------------------------------------------------------
//  Author: Stephan Wynhoff - RWTH-Aachen (Email: Stephan.Wynhoff@cern.ch)
// -----------------------------------------------------------------------------
#include "FastSimulation/Particle/interface/ParticleTable.h"
#include "FastSimulation/Particle/interface/RawParticle.h"

#include <iostream>
#include <iomanip>
#include <cmath>

//using namespace HepPDT;

RawParticle::RawParticle() {
  init();
}

RawParticle::RawParticle(const XYZTLorentzVector& p) 
  : XYZTLorentzVector(p) {
  init();
}

RawParticle::RawParticle(const int id, 
			 const XYZTLorentzVector& p) 
  : XYZTLorentzVector(p) {
  this->init();
  this->setID(id);
}

RawParticle::RawParticle(const std::string name, 
			 const XYZTLorentzVector& p) 
  : XYZTLorentzVector(p) {
  this->init();
  this->setID(name);
}

RawParticle::RawParticle(const XYZTLorentzVector& p, 
			 const XYZTLorentzVector& xStart)  : 
  XYZTLorentzVector(p)
{
  init();
  myVertex = xStart;
}

RawParticle::RawParticle(double px, double py, double pz, double e) : 
  XYZTLorentzVector(px,py,pz,e)
{
  init();
}

RawParticle::RawParticle(const RawParticle &right) : 
  XYZTLorentzVector(right.Px(),right.Py(),right.Pz(),right.E()) 
{
  myId     = right.myId; 
  myStatus = right.myStatus;
  myUsed   = right.myUsed;
  myCharge = right.myCharge;
  myMass   = right.myMass;
  myVertex = (right.myVertex);
  tab = (right.tab);
  myInfo = (right.myInfo);
}

RawParticle::~RawParticle() {
  //  nParticles--;
}

RawParticle&  
RawParticle::operator = (const RawParticle & right ) {
  //  cout << "Copy assignment " << endl;
  if (this != &right) { // don't copy into yourself
    this->SetXYZT(right.Px(),right.Py(),right.Pz(),right.E());
    myId     = right.myId; 
    myStatus = right.myStatus;
    myUsed   = right.myUsed;
    myCharge = right.myCharge;
    myMass   = right.myMass;
    myVertex = right.myVertex;
    tab      = right.tab;
    myInfo   = right.myInfo;
  }
  return *this;
}

void 
RawParticle::init() {
  myId=0;  
  myStatus=99;
  myUsed=0;
  myCharge=0.;
  myMass=0.;
  tab = ParticleTable::instance();
  myInfo=0;
}

void 
RawParticle::setID(const int id) {
  myId = id;
  if ( tab ) {
    if ( !myInfo ) 
      myInfo = tab->theTable()->particle(HepPDT::ParticleID(myId));
    if ( myInfo ) { 
      myCharge = myInfo->charge();
      myMass   = myInfo->mass().value();
    }
  }
}

void 
RawParticle::setID(const std::string name) {
  if ( tab ) { 
    if ( !myInfo ) myInfo = tab->theTable()->particle(name);
    if ( myInfo ) { 
      myId = myInfo->pid();
      myCharge = myInfo->charge();
      myMass   = myInfo->mass().value();
    } else {
      myId = 0;
    }
  }
}

void 
RawParticle::setStatus(int istat) {
  myStatus = istat;
}

void 
RawParticle::setMass(float m) {
  myMass = m;
}

void 
RawParticle::setCharge(float q) {
  myCharge = q;
}

void 
RawParticle::chargeConjugate() {
  myId = -myId;
  myCharge = -1*myCharge;
}

void 
RawParticle::setT(const double t) {
  myVertex.SetE(t);
}

void 
RawParticle::rotate(double angle, const XYZVector& raxis) {
  Rotation r(raxis,angle);
  XYZVector v(r * Vect());
  SetXYZT(v.X(),v.Y(),v.Z(),E());
}

void 
RawParticle::rotateX(double rphi) {
  RotationX r(rphi);
  XYZVector v(r * Vect());
  SetXYZT(v.X(),v.Y(),v.Z(),E());
}

void 
RawParticle::rotateY(double rphi) {
  RotationY r(rphi);
  XYZVector v(r * Vect());
  SetXYZT(v.X(),v.Y(),v.Z(),E());
}

void 
RawParticle::rotateZ(double rphi) {
  RotationZ r(rphi);
  XYZVector v(r * Vect());
  SetXYZT(v.X(),v.Y(),v.Z(),E());
}

void 
RawParticle::boost(double betax, double betay, double betaz) {
  Boost b(betax,betay,betaz);
  XYZTLorentzVector p ( b * momentum() );
  SetXYZT(p.X(),p.Y(),p.Z(),p.T());
}

std::string RawParticle::PDGname() const {
  std::string MyParticleName;
  if ( tab && myInfo ) {
    MyParticleName = myInfo->name();
  } else {
    MyParticleName = "unknown  ";
  }
  return (std::string) MyParticleName;}


void 
RawParticle::printName() const {
  std::string MyParticleName = PDGname();
  if (MyParticleName.length() != 0) {
    std::cout <<  MyParticleName;
    for(unsigned int k=0;k<9-MyParticleName.length() && k<10; k++) 
      std::cout << " " ;
  } else {
    std::cout << "unknown  ";
  }
}

void 
RawParticle::print() const {
  printName();
  std::cout << std::setw(3) << status();
  std::cout.setf(std::ios::fixed, std::ios::floatfield);
  std::cout.setf(std::ios::right, std::ios::adjustfield);
  std::cout << std::setw(8) << std::setprecision(2) << Px();
  std::cout << std::setw(8) << std::setprecision(2) << Py();
  std::cout << std::setw(8) << std::setprecision(2) << Pz();
  std::cout << std::setw(8) << std::setprecision(2) << E();
  std::cout << std::setw(8) << std::setprecision(2) << M();
  std::cout << std::setw(8) << std::setprecision(2) << mass();
  std::cout << std::setw(8) << std::setprecision(2) << charge();
  std::cout << std::setw(8) << std::setprecision(2) << X();
  std::cout << std::setw(8) << std::setprecision(2) << Y();
  std::cout << std::setw(8) << std::setprecision(2) << Z();
  std::cout << std::setw(8) << std::setprecision(2) << T();
  std::cout << std::setw(0) << std::endl;
}

std::ostream& operator <<(std::ostream& o , const RawParticle& p) {

  o.setf(std::ios::fixed, std::ios::floatfield);
  o.setf(std::ios::right, std::ios::adjustfield);


  o << std::setw(4) << std::setprecision(2) << p.pid() << " (";
  o << std::setw(2) << std::setprecision(2) << p.status() << "): ";
  o << std::setw(10) << std::setprecision(4) << p.momentum() << " ";
  o << std::setw(10) << std::setprecision(4) << p.vertex();
  return o;

} 

double 
RawParticle::PDGcharge() const { 
  double q=-99999;
  if ( myInfo ) {
    q=myInfo->charge();
  }
  return q;
}

double 
RawParticle::PDGmass() const  { 
  double m=-99999;
  if ( myInfo ) {
    m = myInfo->mass().value();
  }
  return m;
}

double 
RawParticle::PDGcTau() const {
  double ct=1E99;
  if ( myInfo ) {

    // The lifetime is 0. in the Pythia Particle Data Table !
    //    ct=tab->theTable()->particle(ParticleID(myId))->lifetime().value();

    // Get it from the width (apparently Gamma/c!)
    double w = myInfo->totalWidth().value();
    if ( w != 0. && myId != 1000022 ) { 
      ct = 6.582119e-25 / w / 10.;   // ctau in cm 
    } else {
    // Temporary fix of a bug in the particle data table
      unsigned amyId = abs(myId);
      if ( amyId != 22 &&    // photon 
	   amyId != 11 &&    // e+/-
	   amyId != 10 &&    // nu_e
	   amyId != 12 &&    // nu_mu
	   amyId != 14 &&    // nu_tau
	   amyId != 1000022 && // Neutralino
	   amyId != 1000039 && // Gravitino
	   amyId != 2112 &&  // neutron/anti-neutron
	   amyId != 2212 &&  // proton/anti-proton
	   amyId != 101 &&   // Deutreron etc..
	   amyId != 102 &&   // Deutreron etc..
	   amyId != 103 &&   // Deutreron etc..
	   amyId != 104 ) {  // Deutreron etc.. 
	ct = 0.;
	/* */
      }
    }
  }

  /*
  std::cout << setw(20) << setprecision(18) 
       << "myId/ctau/width = " << myId << " " 
       << ct << " " << w << endl;  
  */

  return ct;
}

double 
RawParticle::et() const { 
  double mypp, tmpEt=-1.;
  
  mypp = std::sqrt(momentum().mag2());
  if ( mypp != 0 ) {
    tmpEt = E() * pt() / mypp;
  }
  return tmpEt;
}
