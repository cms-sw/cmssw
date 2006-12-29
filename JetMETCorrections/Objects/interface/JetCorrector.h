//
// Original Author:  Fedor Ratnikov Dec 27, 2006
// $Id: HcalHardcodeCalibrations.h,v 1.5 2006/01/10 19:29:40 fedor Exp $
//
// Generic interface for JetCorrection services
//
#ifndef JetCorrector_h
#define JetCorrector_h

/// classes declaration
namespace edm {
  class Event;
  class EventSetup;
}
namespace reco {
  class Jet;
}

class JetCorrector
{
 public:
  JetCorrector (){};
  virtual ~JetCorrector (){};

  /// apply correction using Jet information only
  virtual double correction (const reco::Jet& fJet) const;

  /// apply correction using all event information
  virtual double correction (const reco::Jet& fJet, const edm::Event& fEvent, const edm::EventSetup& fSetup) const;

  /// if correction needs event information
  virtual bool eventRequired () const = 0;
};

#endif
