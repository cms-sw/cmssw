//
// Original Author:  Fedor Ratnikov Dec 27, 2006
// $Id: JetCorrector.h,v 1.1 2006/12/29 00:48:42 fedor Exp $
//
// Generic interface for JetCorrection services
//
#ifndef JetCorrector_h
#define JetCorrector_h

#include <string>

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

  /// retrieve corrector from the event setup. troughs exception if something is missing
  const JetCorrector* getJetCorrector (const std::string& fName, const edm::EventSetup& fSetup); 
};

#endif
