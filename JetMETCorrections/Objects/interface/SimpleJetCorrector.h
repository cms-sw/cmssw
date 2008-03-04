//
// Original Author:  Fedor Ratnikov Feb. 16, 2007
// $Id: JetCorrector.h,v 1.3 2007/01/18 01:35:13 fedor Exp $
//
// Simplest jet corrector scaling every jet by fixed factor
//
#ifndef SimpleJetCorrector_h
#define SimpleJetCorrector_h

#include "JetMETCorrections/Objects/interface/JetCorrector.h"

namespace edm {
  class ParameterSet;
}

class SimpleJetCorrector : public JetCorrector {
 public:
  SimpleJetCorrector (const edm::ParameterSet& fConfig);
  virtual ~SimpleJetCorrector ();
  /// apply correction using Jet information only
  virtual double correction (const LorentzVector& fJet) const {return mScale;}
  /// if correction needs event information
  virtual bool eventRequired () const {return false;}
 private:
  double mScale;
};

#endif
