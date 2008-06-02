//
// Original Author:  Fedor Ratnikov Dec 27, 2006
// $Id: ZSPJetCorrector.h,v 1.2 2007/11/01 21:52:53 fedor Exp $
//
// MC Jet Corrector
//
#ifndef ZSPJetCorrector_h
#define ZSPJetCorrector_h

#include "JetMETCorrections/Objects/interface/JetCorrector.h"
#include "CondFormats/JetMETObjects/interface/SimpleZSPJetCorrector.h"


/// classes declaration
namespace edm {
  class ParameterSet;
}

class SimpleZSPJetCorrector;

class ZSPJetCorrector : public JetCorrector {
 public:
  ZSPJetCorrector (const edm::ParameterSet& fParameters);
  virtual ~ZSPJetCorrector ();

  /// apply correction using Jet information only
  virtual double correction (const LorentzVector& fJet) const;
  /// apply correction using Jet information only
  virtual double correction (const reco::Jet& fJet) const;

  /// if correction needs event information
  virtual bool eventRequired () const {return false;}

 private:
  SimpleZSPJetCorrector* mSimpleCorrector;
};

#endif
