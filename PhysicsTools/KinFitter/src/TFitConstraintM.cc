// Classname: TFitConstraintM
// Author: Jan E. Sundermann, Verena Klose (TU Dresden)      


//________________________________________________________________
// 
// TFitConstraintM::
// --------------------
//
// Fit constraint: mass conservation ( m_i - m_j - MassConstraint == 0 )
//

#include <iostream>
#include <iomanip>
#include "PhysicsTools/KinFitter/interface/TFitConstraintM.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TLorentzVector.h"
#include "TClass.h"


//----------------
// Constructor --
//----------------
TFitConstraintM::TFitConstraintM()
  : TAbsFitConstraint() 
  ,_ParList1(0)
  ,_ParList2(0)
  ,_TheMassConstraint(0)
{

}

TFitConstraintM::TFitConstraintM(std::vector<TAbsFitParticle*>* ParList1,
				 std::vector<TAbsFitParticle*>* ParList2, Double_t Mass)
  : TAbsFitConstraint() 
  ,_ParList1(0)
  ,_ParList2(0)
{
  // ParList1: Vector containing first list of constrained particles 
  //           ( sum[ m_i ] - sum[ m_j ] - MassConstraint == 0 )
  // ParList2: Vector containing second list of constrained particles 
  //           ( sum[ m_i ] - sum[ m_j ]  - MassConstraint == 0 )

  if (ParList1) {
    _ParList1 = (*ParList1);
  }
  if (ParList2) {
    _ParList2 = (*ParList2);
  }
  if (Mass >= 0) {
    _TheMassConstraint = Mass;
  }
  else if(Mass < 0) {
    edm::LogWarning ("NegativeMassConstr")
      << "Mass constraint in TFitConstraintM cannot be set to a negative value, will be set to 0.";
    _TheMassConstraint = 0.;
  }
}

TFitConstraintM::TFitConstraintM(const TString &name, const TString &title,
				 std::vector<TAbsFitParticle*>* ParList1,
				 std::vector<TAbsFitParticle*>* ParList2, Double_t Mass)
  : TAbsFitConstraint(name, title) 
  ,_ParList1(0)
  ,_ParList2(0) 
{
  // ParList1: Vector containing first list of constrained particles 
  //           ( sum[ m_i ] - sum[ m_j ] - MassConstraint == 0 )
  // ParList2: Vector containing second list of constrained particles 
  //           ( sum[ m_i ] - sum[ m_j ] - MassConstraint == 0 )

  if (ParList1) {
    _ParList1 = (*ParList1);
  }
  if (ParList2) {
    _ParList2 = (*ParList2);
  }  
  if (Mass >= 0) {
    _TheMassConstraint = Mass;
  }
  else if(Mass < 0) {
    edm::LogWarning ("NegativeMassConstr")
      << "Mass constraint in TFitConstraintM cannot be set to a negative value, will be set to 0.";
    _TheMassConstraint = 0.;
  }
}
void TFitConstraintM::addParticle1( TAbsFitParticle* particle ) {
  // Add one constrained particle to first list of particles
  // ( sum[ m_i ] - sum[ m_j ] - MassConstraint == 0 )

  _ParList1.push_back( particle );

}

void TFitConstraintM::addParticle2( TAbsFitParticle* particle ) {
  // Add one constrained particle to second list of particles
  // ( sum[ m_i ] - sum[ m_j ] - MassConstraint == 0 )

  _ParList2.push_back( particle );

}

void TFitConstraintM::addParticles1( TAbsFitParticle* p1, TAbsFitParticle* p2, TAbsFitParticle* p3, TAbsFitParticle* p4,
				     TAbsFitParticle* p5, TAbsFitParticle* p6, TAbsFitParticle* p7, TAbsFitParticle* p8,
				     TAbsFitParticle* p9, TAbsFitParticle* p10) {
  // Add many constrained particle to first list of particles
  // ( sum[ m_i ] - sum[ m_j ] - MassConstraint == 0 )

  if (p1) addParticle1( p1 );
  if (p2) addParticle1( p2 );
  if (p3) addParticle1( p3 );
  if (p4) addParticle1( p4 );
  if (p5) addParticle1( p5 );
  if (p6) addParticle1( p6 );
  if (p7) addParticle1( p7 );
  if (p8) addParticle1( p8 );
  if (p9) addParticle1( p9 );
  if (p10) addParticle1( p10 );

}

void TFitConstraintM::addParticles2( TAbsFitParticle* p1, TAbsFitParticle* p2, TAbsFitParticle* p3, TAbsFitParticle* p4,
				     TAbsFitParticle* p5, TAbsFitParticle* p6, TAbsFitParticle* p7, TAbsFitParticle* p8,
				     TAbsFitParticle* p9, TAbsFitParticle* p10) {
  // Add many constrained particle to second list of particles
  // ( sum[ m_i ] - sum[ m_j ] - MassConstraint == 0 )

  if (p1) addParticle2( p1 );
  if (p2) addParticle2( p2 );
  if (p3) addParticle2( p3 );
  if (p4) addParticle2( p4 );
  if (p5) addParticle2( p5 );
  if (p6) addParticle2( p6 );
  if (p7) addParticle2( p7 );
  if (p8) addParticle2( p8 );
  if (p9) addParticle2( p9 );
  if (p10) addParticle2( p10 );

}


//--------------
// Destructor --
//--------------
TFitConstraintM::~TFitConstraintM() {

}

//--------------
// Operations --
//--------------
TMatrixD* TFitConstraintM::getDerivative( TAbsFitParticle* particle ) {
  // returns derivative df/dP with P=(p,E) and f the constraint (f=0).
  // The matrix contains one row (df/dp, df/dE).

  TMatrixD* DerivativeMatrix = new TMatrixD(1,4);
  (*DerivativeMatrix) *= 0.;

  // Pf[4] is the 4-Mom (p,E) of the sum of particles on 
  // the list particle is part of 
  
  Double_t Factor = 0.;
  TLorentzVector Pf(0., 0., 0., 0.);

  if( OnList( &_ParList1, particle) ) {
    UInt_t Npart = _ParList1.size();
    for (unsigned int i=0; i<Npart; i++) {
      const TLorentzVector* FourVec = (_ParList1[i])->getCurr4Vec();
      Pf += (*FourVec);
    }
    if( Pf.M() == 0. ) {
      edm::LogInfo ("KinFitter")
	<< "Division by zero in "
	<< IsA()->GetName() << " (named " << GetName() << ", titled " << GetTitle()
	<< ") will lead to Inf in derivative matrix for particle "
	<< particle->GetName() << ".";
    }
    Factor = 1./ Pf.M();
  } else if (OnList( &_ParList2, particle) ) {
    UInt_t Npart = _ParList2.size();
    for (unsigned int i=0; i<Npart; i++) {
      const TLorentzVector* FourVec = (_ParList2[i])->getCurr4Vec();
      Pf += (*FourVec);
    }
    if( Pf.M() == 0. ) {
      edm::LogInfo ("KinFitter")
	<< "Division by zero in "
	<< IsA()->GetName() << " (named " << GetName() << ", titled " << GetTitle()
	<< ") will lead to Inf in derivative matrix for particle "
	<< particle->GetName() << ".";
    }
    Factor = -1./Pf.M();
  } else {
    Factor = 0.; 
  }
  
  (*DerivativeMatrix)(0,0) = -Pf[0] ;
  (*DerivativeMatrix)(0,1) = -Pf[1];
  (*DerivativeMatrix)(0,2) = -Pf[2];
  (*DerivativeMatrix)(0,3) = +Pf[3];
  (*DerivativeMatrix) *= Factor;

  return DerivativeMatrix;

}


Double_t TFitConstraintM::getInitValue() {
  // Get initial value of constraint (before the fit)

  Double_t InitValue(0) ;   
  InitValue = CalcMass(&_ParList1,true) - CalcMass(&_ParList2,true) - _TheMassConstraint ;
  return InitValue;
}

Double_t TFitConstraintM::getCurrentValue() {
  // Get value of constraint after the fit

  Double_t CurrentValue(0);
  CurrentValue= CalcMass(&_ParList1,false) - CalcMass(&_ParList2,false) - _TheMassConstraint;
  return CurrentValue;
}
 

Bool_t TFitConstraintM::OnList(std::vector<TAbsFitParticle*>* List,
			       TAbsFitParticle* particle) {
  // Checks whether list contains given particle

  Bool_t ok(false);
  UInt_t Npart = List->size();
  for (unsigned int i=0;i<Npart;i++) {
    ok = (particle == (*List)[i]);
    if (ok) break;
  }
  return ok;
}

Double_t TFitConstraintM::CalcMass(std::vector<TAbsFitParticle*>* List, Bool_t IniVal) {
  // Calculates initial/current invariant mass of provided list of particles

  TLorentzVector P(0., 0., 0., 0.);
  UInt_t Npart = List->size();
  for (unsigned int i=0;i<Npart;i++) {
    const TLorentzVector* FourVec = 0;
    if (IniVal)
      FourVec = ((*List)[i])->getIni4Vec();
    else      
      FourVec = ((*List)[i])->getCurr4Vec();
    P += (*FourVec);
  }
  return P.M();
}

TString TFitConstraintM::getInfoString() {
  // Collect information to be used for printout

  std::stringstream info;
  info << std::scientific << std::setprecision(6);

  info << "__________________________" << std::endl
       << std::endl;
  info <<"OBJ: " << IsA()->GetName() << "\t" << GetName() << "\t" << GetTitle() << std::endl;

  info << "initial value: " << getInitValue() << std::endl;
  info << "current value: " << getCurrentValue() << std::endl;
  info << "mass: " << _TheMassConstraint << std::endl;

  return info.str();

}

void TFitConstraintM::print() {
  // Print constraint contents

  edm::LogVerbatim("KinFitter") << this->getInfoString();

}
