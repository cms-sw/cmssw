//
// Original Author:  Fedor Ratnikov Dec 27, 2006
// $Id: MCJetCorrector.h,v 1.3 2007/03/30 23:47:55 fedor Exp $
//
// MC Jet Corrector
//
#ifndef MCJetCorrector_h
#define MCJetCorrector_h

#include "JetMETCorrections/Objects/interface/JetCorrector.h"
#include "CondFormats/JetMETObjects/interface/SimpleMCJetCorrector.h"


/// classes declaration
namespace edm {
  class ParameterSet;
}

class SimpleMCJetCorrector;

class MCJetCorrector : public JetCorrector {
 public:
  MCJetCorrector (const edm::ParameterSet& fParameters);
  virtual ~MCJetCorrector ();

  /// apply correction using Jet information only
  virtual double correction (const LorentzVector& fJet) const;

  /// if correction needs event information
  virtual bool eventRequired () const {return false;}

 private:
  SimpleMCJetCorrector* mSimpleCorrector;
};

#endif
