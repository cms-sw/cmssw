//
// Original Author:  Fedor Ratnikov Dec 27, 2006
// $Id: JetCorrector.h,v 1.8 2011/04/28 14:05:21 kkousour Exp $
//
// Generic interface for JetCorrection services
//
#ifndef JetCorrector_h
#define JetCorrector_h

#include <string>
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/JetReco/interface/Jet.h"

/// classes declaration
namespace edm {
  class Event;
  class EventSetup;
}

class JetCorrector
{
 public:

  typedef reco::Particle::LorentzVector LorentzVector;
 
  JetCorrector (){};
  virtual ~JetCorrector (){};

  /// get correction using Jet information only
  virtual double correction (const LorentzVector& fJet) const = 0;

  /// apply correction using Jet information only
  virtual double correction (const reco::Jet& fJet) const = 0;

  /// apply correction using all event information
  virtual double correction (const reco::Jet& fJet,
			     const edm::Event& fEvent, 
			     const edm::EventSetup& fSetup) const;

  /// apply correction using all event information
  virtual double correction (const reco::Jet& fJet,
			     const edm::RefToBase<reco::Jet>& fJetRef,
			     const edm::Event& fEvent, 
			     const edm::EventSetup& fSetup) const;

  /// Apply vectorial correction using all event information
  virtual double correction ( const reco::Jet& fJet, 
			      const edm::RefToBase<reco::Jet>& fJetRef,
			      const edm::Event& fEvent, 
			      const edm::EventSetup& fSetup, 
			      LorentzVector& corrected ) const;
  
  /// if correction needs event information
  virtual bool eventRequired () const = 0;

  /// if correction needs the jet reference
  virtual bool refRequired () const = 0;
  
  /// if vectorial correction is provided
  inline virtual bool vectorialCorrection () const;

  /// retrieve corrector from the event setup. troughs exception if something is missing
  static const JetCorrector* getJetCorrector (const std::string& fName, const edm::EventSetup& fSetup); 
};

// inline method
inline bool JetCorrector::vectorialCorrection () const { return false; }

#endif
