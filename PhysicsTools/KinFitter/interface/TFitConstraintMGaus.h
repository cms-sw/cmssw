using namespace std;

#ifndef TFitConstraintMGaus_hh
#define TFitConstraintMGaus_hh

#include "PhysicsTools/KinFitter/interface/TFitConstraintM.h"

#include <vector>

//class TMatrixD;
class TAbsFitParticle;

class TFitConstraintMGaus: public TFitConstraintM {

public :

  TFitConstraintMGaus();
  TFitConstraintMGaus(vector<TAbsFitParticle*>* ParList1,
		      vector<TAbsFitParticle*>* ParList2,
		      Double_t Mass = 0, Double_t Width = 0);
  TFitConstraintMGaus(const TString &name, const TString &title,
		      vector<TAbsFitParticle*>* ParList1,
		      vector<TAbsFitParticle*>* ParList2,
		      Double_t Mass = 0, Double_t Width = 0);

  virtual ~TFitConstraintMGaus();

  virtual Double_t getInitValue();
  virtual Double_t getCurrentValue();
  virtual TMatrixD* getDerivativeAlpha();

  void setMassConstraint(Double_t Mass, Double_t Width);

  virtual void print(); 

protected :
  
  Double_t _width;

  void init();

  ClassDef(TFitConstraintMGaus, 1)   // Fit constraint: mass conservation

};

#endif

