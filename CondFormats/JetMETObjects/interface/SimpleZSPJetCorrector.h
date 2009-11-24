//
// Original Author:  Fedor Ratnikov Dec 27, 2006
// $Id: SimpleZSPJetCorrector.h,v 1.2 2008/05/20 23:38:36 fedor Exp $
//
// MC Jet Corrector
//
#ifndef SimpleZSPJetCorrector_h
#define SimpleZSPJetCorrector_h

#include "CondFormats/JetMETObjects/interface/StandaloneJetCorrector.h"

#include <map>
#include <string>

/// classes declaration
namespace zsp {
  class ParametrizationZSPJet;
  typedef std::map <double, ParametrizationZSPJet*> ZSPParametersMap;
}

class SimpleZSPJetCorrector : public StandaloneJetCorrector {
 public:
  SimpleZSPJetCorrector ();
  SimpleZSPJetCorrector (const std::string& fDataFile);
  virtual ~SimpleZSPJetCorrector ();

  void init (const std::string& fDataFile);

  virtual double correctionXYZT (double fPx, double fPy, double fPz, double fE) const;
  virtual double correctionPtEtaPhiE (double fPt, double fEta, double fPhi, double fE) const;
  virtual double correctionEtEtaPhiP (double fEt, double fEta, double fPhi, double fP) const;

 private:
  SimpleZSPJetCorrector (const SimpleZSPJetCorrector&);
  SimpleZSPJetCorrector& operator= (const SimpleZSPJetCorrector&);
  zsp::ZSPParametersMap* mParametrization;
};

#endif
