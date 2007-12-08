//
// Original Author:  Fedor Ratnikov Dec 27, 2006
// $Id: SimpleMCJetCorrector.h,v 1.3 2007/03/31 18:24:16 fedor Exp $
//
// MC Jet Corrector
//
#ifndef SimpleMCJetCorrector_h
#define SimpleMCJetCorrector_h

#include <map>
#include <string>

/// classes declaration
namespace {
  class ParametrizationMCJet;
  typedef std::map <double, ParametrizationMCJet*> ParametersMap;
}

class SimpleMCJetCorrector {
 public:
  SimpleMCJetCorrector ();
  SimpleMCJetCorrector (const std::string& fDataFile);
  virtual ~SimpleMCJetCorrector ();

  void init (const std::string& fDataFile);

  virtual double correctionXYZT (double fPx, double fPy, double fPz, double fE) const;
  virtual double correctionPtEtaPhiE (double fPt, double fEta, double fPhi, double fE) const;
  virtual double correctionEtEta (double fEt, double fEta) const;

 private:
  ParametersMap* mParametrization;
};

#endif
