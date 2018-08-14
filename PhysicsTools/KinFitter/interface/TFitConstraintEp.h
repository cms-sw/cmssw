#ifndef TFitConstraintEp_hh
#define TFitConstraintEp_hh

#include "PhysicsTools/KinFitter/interface/TAbsFitConstraint.h"
#include "PhysicsTools/KinFitter/interface/TAbsFitParticle.h"
#include "TMatrixD.h"
#include <vector>

class TFitConstraintEp: public TAbsFitConstraint {

public :

  enum component {
    pX,
    pY,
    pZ,
    E
  };

  TFitConstraintEp( );

  TFitConstraintEp(  const TString &name, const TString &title,
                     TFitConstraintEp::component thecomponent,
                     Double_t constraint = 0.);

  TFitConstraintEp(  std::vector<TAbsFitParticle*>* particles, 
		     TFitConstraintEp::component thecomponent, 
		     Double_t constraint = 0.);

  TFitConstraintEp(  const TString &name, const TString &title,
		     std::vector<TAbsFitParticle*>* particles, 
		     TFitConstraintEp::component thecomponent, 
		     Double_t constraint = 0.);
  ~TFitConstraintEp() override;

  void addParticle( TAbsFitParticle* particle );
  void addParticles( TAbsFitParticle* p1, TAbsFitParticle* p2 = nullptr, TAbsFitParticle* p3 = nullptr, TAbsFitParticle* p4 = nullptr,
		     TAbsFitParticle* p5 = nullptr, TAbsFitParticle* p6 = nullptr, TAbsFitParticle* p7 = nullptr, TAbsFitParticle* p8 = nullptr,
		     TAbsFitParticle* p9 = nullptr, TAbsFitParticle* p10 = nullptr);
  void setConstraint(Double_t constraint){_constraint = constraint;};

  // returns derivative df/dP with P=(p,E) and f the constraint f=0.
  // The matrix contains one row (df/dp, df/dE).
  TMatrixD* getDerivative( TAbsFitParticle* particle ) override;
  Double_t getInitValue() override;
  Double_t getCurrentValue() override;

  TString getInfoString() override;
  void print() override; 
 
protected :


private:
  std::vector<TAbsFitParticle*> _particles;    // Vector containing constrained particles
  Double_t _constraint;                   // Value of constraint
  TFitConstraintEp::component _component; // 4vector component to be constrained

  ClassDefOverride(TFitConstraintEp, 0)
};

#endif
