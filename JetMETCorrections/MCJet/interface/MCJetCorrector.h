//
// Original Author:  Fedor Ratnikov Dec 27, 2006
// $Id: MCJetCorrector.h,v 1.1 2006/12/29 00:48:38 fedor Exp $
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
  virtual double correction (const LorentzVector& fJet) const;

  /// if correction needs event information
  virtual bool eventRequired () const {return false;}

 private:
  void setParameters (const std::string& fType);
  typedef std::map <double, ParametrizationMCJet*> ParametersMap;
  ParametersMap mParametrization;
};

#endif
