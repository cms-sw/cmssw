//
// Original Author:  Fedor Ratnikov Dec 27, 2006
// $Id: MCJetCorrector.h,v 1.2 2007/01/18 01:35:11 fedor Exp $
//
// MC Jet Corrector
//
#ifndef SimpleMCJetCorrector_h
#define SimpleMCJetCorrector_h

#include "CondFormats/JetMETObjects/interface/SimpleJetCorrector.h"

#include <map>
#include <string>

/// classes declaration
namespace {
  class ParametrizationMCJet;
}

class SimpleMCJetCorrector : public SimpleJetCorrector {
 public:
  SimpleMCJetCorrector () {};
  SimpleMCJetCorrector (const std::string& fDataFile);
  virtual ~SimpleMCJetCorrector ();

  void init (const std::string& fDataFile);

  virtual double correctionXYZT (double fPx, double fPy, double fPz, double fE, ) const;
  virtual double correctionPtEtaPhiE (double fPt, double fEta, double fPhi, double fE, ) const;

 private:
  typedef std::map <double, ParametrizationMCJet*> ParametersMap;
  ParametersMap mParametrization;
};

#endif
