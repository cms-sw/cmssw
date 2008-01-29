//
// Original Author:  Fedor Ratnikov Nov 1, 2007
// $Id: MCJetCorrector3D.h,v 1.1 2007/11/01 21:52:53 fedor Exp $
//
// MC 3D monolithic Jet Corrector
//
#ifndef MCJetCorrector3D_h
#define MCJetCorrector3D_h

#include "JetMETCorrections/Objects/interface/JetCorrector.h"

/// classes declaration
namespace edm {
  class ParameterSet;
}

class Simple3DMCJetCorrector;

class MCJetCorrector3D : public JetCorrector {
 public:
  MCJetCorrector3D (const edm::ParameterSet& fParameters);
  virtual ~MCJetCorrector3D ();

  /// apply correction using Jet information only
  virtual double correction (const LorentzVector& fJet) const;

  /// apply correction using all event information !!!->this is huck to retrofit 1.6.X
   virtual double correction (const reco::Jet& fJet, const edm::Event& fEvent, const edm::EventSetup& fSetup) const;

  /// if correction needs event information
  virtual bool eventRequired () const {return false;}

 private:
  Simple3DMCJetCorrector* mSimpleCorrector;
};

#endif
