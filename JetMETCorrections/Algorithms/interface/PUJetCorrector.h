//
// Original Author:  Fedor Ratnikov Dec 27, 2006
// $Id: PUJetCorrector.h,v 1.2 2007/11/01 21:52:53 fedor Exp $
//
// PileUp Subtraction Jet Corrector
//
#ifndef PUJetCorrector_h
#define PUJetCorrector_h

#include "JetMETCorrections/Objects/interface/JetCorrector.h"
#include "CondFormats/JetMETObjects/interface/SimplePUJetCorrector.h"


/// classes declaration
namespace edm {
  class ParameterSet;
}

class SimplePUJetCorrector;

class PUJetCorrector : public JetCorrector {
 public:
  PUJetCorrector (const edm::ParameterSet& fParameters);
  virtual ~PUJetCorrector ();

  /// apply correction using Jet information only
  virtual double correction (const LorentzVector& fJet) const;
  /// apply correction using Jet information only
  virtual double correction (const reco::Jet& fJet) const;

  /// if correction needs event information
  virtual bool eventRequired () const {return false;}

 private:
  SimplePUJetCorrector* mSimpleCorrector;
};

#endif
