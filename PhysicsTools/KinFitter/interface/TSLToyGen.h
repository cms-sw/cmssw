#ifndef TSLToyGen_hh
#define TSLToyGen_hh

#include <vector>
#include "TObject.h"
#include "TObjArray.h"
#include "PhysicsTools/KinFitter/interface/TKinFitter.h"
#include "TVector3.h"

class TAbsFitParticle;

class TSLToyGen : public TObject {

public :

  TSLToyGen( const TAbsFitParticle* bReco, const TAbsFitParticle* lepton, const TAbsFitParticle* X, const TAbsFitParticle* neutrino);
  ~TSLToyGen() override;
  Bool_t doToyExperiments( Int_t nbExperiments = 1000 );

  TH1D* _histStatus;
  TH1D* _histNIter;
  TH1D* _histPChi2;
  TH1D* _histChi2;

  TH1D* _histMBrecoTrue;
  TH1D* _histMBrecoSmear; 
  TH1D* _histMBrecoFit; 
  TH1D* _histMXTrue;
  TH1D* _histMXSmear;
  TH1D* _histMXFit;
  TH1D* _histMXlnuTrue;
  TH1D* _histMXlnuSmear;
  TH1D* _histMXlnuFit;

  TObjArray _histsParTrue;
  TObjArray _histsParSmear;
  TObjArray _histsParFit;

  TObjArray _histsPull1;
  TObjArray _histsError1;
  TObjArray _histsDiff1;
  TObjArray _histsPull2;
  TObjArray _histsError2;
  TObjArray _histsDiff2;
  
  void setprintPartIni(Bool_t value) { _printPartIni = value; }
  void setprintConsIni(Bool_t value) { _printConsIni = value; }
  void setprintSmearedPartBefore(Bool_t value) { _printSmearedPartBefore = value; }
  void setprintPartAfter(Bool_t value) { _printPartAfter = value; } 
  void setprintConsBefore(Bool_t value) { _printConsBefore = value; }
  void setprintConsAfter(Bool_t value) { _printConsAfter = value; }

  void setMassConstraint(Bool_t value) { _withMassConstraint = value; }
  void setMPDGCons(Bool_t value) { _withMPDGCons = value; }
  void setCheckConstraintsTruth(Bool_t value) { _doCheckConstraintsTruth = value; }

protected:

  void smearParticles();

  void createHists();

  void fillPull1();
  void fillPull2();
  void fillPar();
  void fillM();

private :
  
  std::vector<TAbsFitParticle*> _inimeasParticles;    // vector that contains all true measured particles
  std::vector<TAbsFitParticle*> _iniunmeasParticles;  // vector that contains all true unmeasured particles
  std::vector<TAbsFitParticle*> _measParticles;    // vector that contains all smeared measured particles
  std::vector<TAbsFitParticle*> _unmeasParticles;  // vector that contains all smeared unmeasured particles
  TVector3 _Y4S;
  
  TAbsFitParticle* _iniBreco;
  TAbsFitParticle* _iniLepton;
  TAbsFitParticle* _iniX;
  TAbsFitParticle* _iniNeutrino;
  TAbsFitParticle* _breco;
  TAbsFitParticle* _lepton;
  TAbsFitParticle* _X;
  TAbsFitParticle* _neutrino;

  Bool_t _printPartIni;
  Bool_t _printConsIni;
  Bool_t _printSmearedPartBefore ;
  Bool_t _printConsBefore;
  Bool_t _printConsAfter;
  Bool_t _printPartAfter;
  Bool_t _withMassConstraint;
  Bool_t _withMPDGCons;
  Bool_t _doCheckConstraintsTruth;

  ClassDefOverride(TSLToyGen, 0)
};

#endif
