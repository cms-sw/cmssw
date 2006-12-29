//
// Original Author:  Fedor Ratnikov Dec 27, 2006
// $Id: HcalHardcodeCalibrations.h,v 1.5 2006/01/10 19:29:40 fedor Exp $
//
// MC Jet Corrector
//
#ifndef MCJetCorrector_h
#define MCJetCorrector_h

#include "JetMETCorrections/Objects/interface/JetCorrector.h"

#include <map>
#include <string>

/// classes declaration
namespace edm {
  class ParameterSet;
}
namespace {
  class ParametrizationMCJet;
}


class MCJetCorrector : public JetCorrector {
 public:
  MCJetCorrector (const edm::ParameterSet& fParameters);
  virtual ~MCJetCorrector ();

  /// apply correction using Jet information only
  virtual double correction (const reco::Jet& fJet) const;

  /// if correction needs event information
  virtual bool eventRequired () const {return false;}

 private:
  void setParameters (const std::string& fType);
  typedef std::map <double, ParametrizationMCJet*> ParametersMap;
  ParametersMap mParametrization;
};

#endif
