//
// Original Author:  Fedor Ratnikov Dec 27, 2006
// $Id: SimplePUJetCorrector.h,v 1.4 2007/12/08 01:55:40 fedor Exp $
//
// MC Jet Corrector
//
#ifndef SimplePUJetCorrector_h
#define SimplePUJetCorrector_h

#include <map>
#include <string>

/// classes declaration
namespace {
  class ParametrizationPUJet;
  typedef std::map <double, ParametrizationPUJet*> ParametersMap;
}

class SimplePUJetCorrector {
 public:
  SimplePUJetCorrector ();
  SimplePUJetCorrector (const std::string& fDataFile);
  virtual ~SimplePUJetCorrector ();

  void init (const std::string& fDataFile);

  virtual double correctionXYZT (double fPx, double fPy, double fPz, double fE) const;
  virtual double correctionPtEtaPhiE (double fPt, double fEta, double fPhi, double fE) const;
  virtual double correctionEtEta (double fEt, double fEta) const;

 private:
  ParametersMap* mParametrization;
};

#endif
