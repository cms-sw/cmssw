// Classname: TFitConstraintEp
// Author: Jan E. Sundermann, Verena Klose (TU Dresden)      


//________________________________________________________________
// 
// TFitConstraintEp::
// --------------------
//
// Fit constraint: energy and momentum conservation
//

#include "PhysicsTools/KinFitter/interface/TFitConstraintEp.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <iomanip>
#include "TClass.h"

//----------------
// Constructor --
//----------------



TFitConstraintEp::TFitConstraintEp()
  :TAbsFitConstraint()
  ,_particles(0)
  ,_constraint(0.)
  ,_component(TFitConstraintEp::pX)
{}

TFitConstraintEp::TFitConstraintEp(std::vector<TAbsFitParticle*>* particles, 
				   TFitConstraintEp::component thecomponent, 
				   Double_t constraint)
  :TAbsFitConstraint()
  ,_particles(0)
  ,_constraint(constraint)
  ,_component(thecomponent)
{
  // particles: vector containing pointer to TAbsFitParticle objects. 
  //            Energy or momentum conservation will be calculated for
  //            those particles.
  // thecomponent: conserved 4vector component ( pX, pY, pZ, E ). For
  //            full 4vector conservation four objects of type TFitConstraintEp
  //            are needed (four constraints)
  // constraint:value of energy or momentum constraint ( e.g. sum[ pX_i ] = constraint )

  if (particles) {
    _particles = (*particles);
  }
}

TFitConstraintEp::TFitConstraintEp(const TString &name, const TString &title,
				   std::vector<TAbsFitParticle*>* particles, 
				   TFitConstraintEp::component thecomponent, 
				   Double_t constraint)
  :TAbsFitConstraint(name, title)
  ,_particles(0)
  ,_constraint(constraint)
  ,_component(thecomponent)
{
  // particles: vector containing pointer to TAbsFitParticle objects. 
  //            Energy or momentum conservation will be calculated for
  //            those particles.
  // thecomponent: conserved 4vector component ( pX, pY, pZ, E ). For
  //            full 4vector conservation four objects of type TFitConstraintEp
  //            are needed (four constraints)
  // constraint:value of energy or momentum constraint ( e.g. sum[ pX_i ] = constraint )

  if (particles) {
    _particles = (*particles);
  }
}

//--------------
// Destructor --
//--------------
TFitConstraintEp::~TFitConstraintEp() {

}

void TFitConstraintEp::addParticle( TAbsFitParticle* particle ) {
  // Add one particles to list of constrained particles

  _particles.push_back( particle );

}

void TFitConstraintEp::addParticles( TAbsFitParticle* p1, TAbsFitParticle* p2, TAbsFitParticle* p3, TAbsFitParticle* p4,
				     TAbsFitParticle* p5, TAbsFitParticle* p6, TAbsFitParticle* p7, TAbsFitParticle* p8,
				     TAbsFitParticle* p9, TAbsFitParticle* p10) {
  // Add many particles to list of constrained particles

  if (p1) addParticle( p1 );
  if (p2) addParticle( p2 );
  if (p3) addParticle( p3 );
  if (p4) addParticle( p4 );
  if (p5) addParticle( p5 );
  if (p6) addParticle( p6 );
  if (p7) addParticle( p7 );
  if (p8) addParticle( p8 );
  if (p9) addParticle( p9 );
  if (p10) addParticle( p10 );

}

//--------------
// Operations --
//--------------
TMatrixD* TFitConstraintEp::getDerivative( TAbsFitParticle* particle ) {
  // returns derivative df/dP with P=(p,E) and f the constraint (f=0).
  // The matrix contains one row (df/dp, df/dE).

  TMatrixD* DerivativeMatrix = new TMatrixD(1,4);
  (*DerivativeMatrix) *= 0.;
  (*DerivativeMatrix)(0,(int) _component) = 1.;
  return DerivativeMatrix;
}


Double_t TFitConstraintEp::getInitValue() {
  // Get initial value of constraint (before the fit)

  Double_t InitValue(0) ; 
  UInt_t Npart = _particles.size();
  for (unsigned int i=0;i<Npart;i++) {
    const TLorentzVector* FourVec = _particles[i]->getIni4Vec();
    InitValue += (*FourVec)[(int) _component];
  }
  InitValue -= _constraint;
  return InitValue;
}

Double_t TFitConstraintEp::getCurrentValue() {
  // Get value of constraint after the fit

  Double_t CurrentValue(0);
  UInt_t Npart = _particles.size();
  for (unsigned int i=0;i<Npart;i++) {
    const TLorentzVector* FourVec = _particles[i]->getCurr4Vec();
    CurrentValue += (*FourVec)[(int) _component];
  }
  CurrentValue -= _constraint;
  return CurrentValue;
}

TString TFitConstraintEp::getInfoString() {
  // Collect information to be used for printout

  std::stringstream info;
  info << std::scientific << std::setprecision(6);

  info << "__________________________" << std::endl
       << std::endl;
  info << "OBJ: " << IsA()->GetName() << "\t" << GetName() << "\t" << GetTitle() << std::endl;

  info << "initial value: " << getInitValue() << std::endl;
  info << "current value: " << getCurrentValue() << std::endl;
  info << "component: " << _component << std::endl;
  info << "constraint: " << _constraint << std::endl;

  return info.str();

}

void TFitConstraintEp::print() {
  // Print constraint contents

  edm::LogVerbatim("KinFitter") << this->getInfoString();

}
