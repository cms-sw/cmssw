//
// Original Author:  Fedor Ratnikov Dec 27, 2006
// $Id: SimpleZSPJPTJetCorrector.h,v 1.1 2010/03/04 13:11:52 kodolova Exp $
//
// MC Jet Corrector
//
#ifndef SimpleZSPJPTJetCorrector_h
#define SimpleZSPJPTJetCorrector_h
#include "CondFormats/JetMETObjects/interface/JetCorrectorParameters.h"
#include <map>
#include <string>

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
};

#endif
