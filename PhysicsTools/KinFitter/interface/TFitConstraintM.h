#ifndef TFitConstraintM_hh
#define TFitConstraintM_hh

#include "PhysicsTools/KinFitter/interface/TAbsFitConstraint.h"
#include <vector>

#include "TMatrixD.h"

class TAbsFitParticle;

class TFitConstraintM: public TAbsFitConstraint {

public :

  TFitConstraintM();
  TFitConstraintM(std::vector<TAbsFitParticle*>* ParList1,
		  std::vector<TAbsFitParticle*>* ParList2,
		  Double_t Mass = 0);
  TFitConstraintM(const TString &name, const TString &title,
		  std::vector<TAbsFitParticle*>* ParList1,
		  std::vector<TAbsFitParticle*>* ParList2,
		  Double_t Mass = 0);

  ~TFitConstraintM() override;

  void addParticle1( TAbsFitParticle* particle );
  void addParticle2( TAbsFitParticle* particle );
  void addParticles1( TAbsFitParticle* p1, TAbsFitParticle* p2 = nullptr, TAbsFitParticle* p3 = nullptr, TAbsFitParticle* p4 = nullptr,
		      TAbsFitParticle* p5 = nullptr, TAbsFitParticle* p6 = nullptr, TAbsFitParticle* p7 = nullptr, TAbsFitParticle* p8 = nullptr,
		      TAbsFitParticle* p9 = nullptr, TAbsFitParticle* p10 = nullptr);
  void addParticles2( TAbsFitParticle* p1, TAbsFitParticle* p2 = nullptr, TAbsFitParticle* p3 = nullptr, TAbsFitParticle* p4 = nullptr,
		      TAbsFitParticle* p5 = nullptr, TAbsFitParticle* p6 = nullptr, TAbsFitParticle* p7 = nullptr, TAbsFitParticle* p8 = nullptr,
		      TAbsFitParticle* p9 = nullptr, TAbsFitParticle* p10 = nullptr);
  void setMassConstraint(Double_t Mass) { _TheMassConstraint = Mass; }

  // returns derivative df/dP with P=(p,E) and f the constraint f=0 for 
  // one particle. The matrix contains one row (df/dp, df/dE).
  TMatrixD* getDerivative( TAbsFitParticle* particle ) override;
  Double_t getInitValue() override;
  Double_t getCurrentValue() override;

  Bool_t OnList(std::vector<TAbsFitParticle*>* List, TAbsFitParticle* particle);
  Double_t CalcMass(std::vector<TAbsFitParticle*>* List, Bool_t IniVal);

  TString getInfoString() override;
  void print() override; 

protected :
  
  std::vector<TAbsFitParticle*> _ParList1;   // Vector containing first list of constrained particles ( sum[ m_i ] - sum[ m_j ] == 0 )
  std::vector<TAbsFitParticle*> _ParList2;   // Vector containing second list of constrained particles ( sum[ m_i ] - sum[ m_j ] == 0 )
  Double_t _TheMassConstraint;

private :

  ClassDefOverride(TFitConstraintM, 0)
};

#endif

