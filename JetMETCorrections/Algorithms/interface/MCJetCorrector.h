//
// Original Author:  Fedor Ratnikov Dec 27, 2006
// $Id: MCJetCorrector.h,v 1.1 2007/10/03 23:29:50 fedor Exp $
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
  /// apply correction using Jet information only
  virtual double correction (const reco::Jet& fJet) const;

  /// if correction needs event information
  virtual bool eventRequired () const {return false;}

 private:
  SimpleMCJetCorrector* mSimpleCorrector;
};

#endif
