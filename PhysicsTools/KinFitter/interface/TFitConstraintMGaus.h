#ifndef TFitConstraintMGaus_hh
#define TFitConstraintMGaus_hh

#include "PhysicsTools/KinFitter/interface/TFitConstraintM.h"

#include <vector>

class TAbsFitParticle;

class TFitConstraintMGaus: public TFitConstraintM {

public :

  TFitConstraintMGaus();
  TFitConstraintMGaus(std::vector<TAbsFitParticle*>* ParList1,
		      std::vector<TAbsFitParticle*>* ParList2,
		      Double_t Mass = 0, Double_t Width = 0);
  TFitConstraintMGaus(const TString &name, const TString &title,
		      std::vector<TAbsFitParticle*>* ParList1,
		      std::vector<TAbsFitParticle*>* ParList2,
		      Double_t Mass = 0, Double_t Width = 0);

  ~TFitConstraintMGaus() override;

  Double_t getInitValue() override;
  Double_t getCurrentValue() override;
  TMatrixD* getDerivativeAlpha() override;

  void setMassConstraint(Double_t Mass, Double_t Width);

  TString getInfoString() override;
  void print() override; 

protected :
  
  Double_t _width;

  void init();

private :

  ClassDefOverride(TFitConstraintMGaus, 0)
};

#endif

