//
// Original Author:  Fedor Ratnikov Dec 27, 2006
// $Id: SimpleZSPJPTJetCorrector.h,v 1.1 2012/10/18 08:46:42 eulisse Exp $
//
// MC Jet Corrector
//
#ifndef SimpleZSPJPTJetCorrector_h
#define SimpleZSPJPTJetCorrector_h
#include "CondFormats/JetMETObjects/interface/JetCorrectorParameters.h"
#include <map>
#include <string>
#include <TFormula.h>

/// classes declaration


class SimpleZSPJPTJetCorrector {
 public:
  SimpleZSPJPTJetCorrector ();
  SimpleZSPJPTJetCorrector (const std::string& fDataFile);
  virtual ~SimpleZSPJPTJetCorrector ();

  void init (const std::string& fDataFile);
  virtual double correctionPtEtaPhiE (double fPt, double fEta, double fPhi, double fE) const;
  virtual double correctionEtEtaPhiP (double fEt, double fEta, double fPhi, double fP) const;
  virtual double correctionPUEtEtaPhiP (double fEt, double fEta, double fPhi, double fP) const;

 private:
  SimpleZSPJPTJetCorrector (const SimpleZSPJPTJetCorrector&);
  SimpleZSPJPTJetCorrector& operator= (const SimpleZSPJPTJetCorrector&);
  JetCorrectorParameters* mParameters; 
  TFormula*               mFunc;   
};

#endif
